"""Second-pass text evidence and actionable quality checks after fast grouping."""
from __future__ import annotations

import concurrent.futures
from collections import defaultdict
from functools import lru_cache
import time
from pathlib import Path

import cv2

from slide_regions import extract
from slide_ocr import local_numeric_evidence, read_text, text_evidence, signature
from slide_speakers import CHECK_SPEAKER, ENGINE, SPEAKER_ONLY, create_detector, is_protected_slide, speaker_status
from slide_transitions import ONE_SIDED_REVIEW_REASON

NUMERIC_REVIEW = 'check possible changed value'

def audit_speaker_frames(source, groups, by_id, args, decode, ocr_enabled):
    """Keep clear camera shots out of exports while retaining every original capture."""
    report = {'status': 'ok', 'engine': ENGINE, 'auto_exclusion_enabled': ocr_enabled,
              'checked_groups': 0, 'auto_excluded': [], 'needs_review': [],
              'representative_changes': [], 'failures': []}
    try:
        detector = create_detector()
    except (OSError, cv2.error) as exc:
        report.update(status='unavailable', auto_exclusion_enabled=False, error=str(exc))
        for group in groups:
            if not is_protected_slide(by_id[group['representative']]):
                group['speaker_review'] = True
                report['needs_review'].append({'group': group['id'], 'representative': group['representative']})
        return report

    cache = getattr(args, 'ocr_cache', Path(__file__).with_name('.cache') / 'ocr')
    for group in groups:
        original = by_id[group['representative']]
        if original.get('transition_review'):
            continue
        if is_protected_slide(original):
            continue
        report['checked_groups'] += 1
        try:
            slide = extract(decode(source.read(original['name'])), original)
            status = speaker_status(detector, slide, original)
            if status == 'speaker-only' and not ocr_enabled:
                status = 'review'
            replacement = None
            if status == 'speaker-only':
                for member in group['members']:
                    if member == original['id']:
                        continue
                    alternate = by_id[member]
                    if not is_protected_slide(alternate):
                        alternate_slide = extract(decode(source.read(alternate['name'])), alternate)
                        if ocr_enabled and 'ocr' not in alternate:
                            alternate['ocr'] = read_text(alternate_slide, cache)
                        if not is_protected_slide(alternate) and speaker_status(detector, alternate_slide, alternate) == 'speaker-only':
                            continue
                    if is_protected_slide(alternate):
                        replacement = alternate['id']
                        break
                    # An unconfirmed member may carry content absent from the current choice.
                    status = 'review'
        except (OSError, ValueError, RuntimeError, cv2.error) as exc:
            report['status'] = 'partial'
            report['failures'].append({'group': group['id'], 'representative': original['id'], 'error': str(exc)})
            status = 'review'
            replacement = None
        if replacement is not None:
            group['representative'] = replacement
            report['representative_changes'].append({'group': group['id'], 'from': original['id'], 'to': replacement})
        elif status == 'speaker-only':
            group['excluded'] = True
            group['auto_exclusion'] = 'speaker-only'
            report['auto_excluded'].append({'group': group['id'], 'representative': original['id']})
        elif status == 'review':
            group['speaker_review'] = True
            report['needs_review'].append({'group': group['id'], 'representative': original['id']})
    return report


def audit(source,items,groups,decisions,candidates,args,out):
    from slide_dedupe import compare,decode,better_frame
    by_id={i['id']:i for i in items}
    log={'ocr_enabled':getattr(args,'ocr','off')=='on','numeric_splits':[],
         'numeric_reviews':[], 'ocr_merges':[],'suppressed_distinct_pairs':0}
    numeric_reviews = {}
    start=time.perf_counter()
    def read_native(item):
        return extract(decode(source.read(item['name'])),item)

    @lru_cache(maxsize=3)
    def native_for_comparison(index):
        return read_native(by_id[index])

    @lru_cache(maxsize=64)
    def compare_ids(a, b, reread):
        return text_evidence(by_id[a], by_id[b], native_pair=lambda: (native_for_comparison(a), native_for_comparison(b)),
                             reread_numeric=reread)

    def compare_text(a, b, *, reread=False):
        return compare_ids(a['id'], b['id'], reread)

    def remember_numeric(a, b, evidence):
        if evidence.get('numeric_reviews'):
            key = tuple(sorted((a['id'], b['id'])))
            numeric_reviews.setdefault(key, {'a': a['id'], 'b': b['id'], 'cells': evidence['numeric_reviews']})

    def inspect_local_change(a, b, metric, evidence):
        if (evidence['numeric_conflict'] or metric['changed'] > .005
                or min(metric.get('coverage_a', 0), metric.get('coverage_b', 0)) < .98
                or a.get('sha256') == b.get('sha256')
                or (a['slide_width'], a['slide_height']) != (b['slide_width'], b['slide_height'])):
            return
        local_occlusions, unresolved = [], []
        changes = local_numeric_evidence(native_for_comparison(a['id']), native_for_comparison(b['id']),
                                         occlusions=local_occlusions, unresolved=unresolved)
        if changes:
            evidence.update(numeric_conflict=True, numeric_changes=changes, numeric_source='native region')
        elif unresolved:
            evidence.update(numeric_conflict=True, numeric_reviews=unresolved, numeric_source='native region')
        elif local_occlusions:
            evidence['numeric_occlusions'] = sorted(set(evidence.get('numeric_occlusions', []) + local_occlusions))

    def read_targets(targets):
        if not targets:
            return
        cache = getattr(args, 'ocr_cache', Path(__file__).with_name('.cache') / 'ocr')
        def work(i):
            try:
                return i, read_text(read_native(by_id[i]), cache)
            except Exception as exc:
                return i, {'status': 'error', 'error': str(exc)}
        with concurrent.futures.ThreadPoolExecutor(max_workers=getattr(args,'ocr_workers',4)) as pool:
            for n,(i,result) in enumerate(pool.map(work,sorted(targets)),1):
                by_id[i]['ocr']=result
                if n%25==0:
                    print(f'  OCR {n}/{len(targets)}',flush=True)

    def read_new_representatives(groups, phase):
        missing = {g['representative'] for g in groups
                   if not g.get('excluded') and 'ocr' not in by_id[g['representative']]}
        if missing:
            print(f'  Offline OCR: {len(missing)} newly selected frames ({phase})...', flush=True)
            read_targets(missing)
        return len(missing)

    if log['ocr_enabled']:
        targets={g['representative'] for g in groups}
        suspicious=[d for d in decisions if d['reason']=='build' or
                    (d['reason']=='duplicate' and (d['changed']>0 or d.get('local_change',0)>0))]
        targets.update(d[k] for d in suspicious for k in ('image','matched'))
        print(f'  Offline OCR: {len(targets)} representative / changed-content frames...',flush=True)
        read_targets(targets)
        failures=[i for i in targets if by_id[i]['ocr']['status']!='ok']
        if failures and len(failures)==len(targets):
            raise RuntimeError('OCR failed for every target. Install requirements.txt or explicitly use --ocr off. '
                               +by_id[failures[0]]['ocr'].get('error', by_id[failures[0]]['ocr']['status']))
        # Separate confidently detected number changes that the low-res pass missed.
        suspicious_ids = {d['image'] for d in suspicious}
        separations = defaultdict(set)
        clear_alternates = defaultdict(set)
        for d in decisions:
            a,b=by_id[d['image']],by_id[d['matched']]
            evidence = dict(compare_text(a, b, reread=True)) if a['id'] in suspicious_ids else {'numeric_conflict': False}
            if d['reason'] == 'duplicate':
                inspect_local_change(a, b, d, evidence)
            if d['reason']=='duplicate' and not evidence['numeric_conflict']:
                obscured = evidence.get('numeric_occlusions')
                if obscured == ['a']:
                    clear_alternates[a['id']].add(b['id'])
                elif obscured == ['b']:
                    clear_alternates[b['id']].add(a['id'])
            if evidence['numeric_conflict']:
                remember_numeric(a, b, evidence)
                separations[a['id']].add(b['id'])
                separations[b['id']].add(a['id'])
                group=next(g for g in groups if a['id'] in g['members'])
                if b['id'] not in group['members'] or len(group['members'])==1:
                    continue
                group['members'].remove(a['id'])
                if group['representative']==a['id']:
                    group['representative']=max(group['members'],key=lambda i:by_id[i]['slide_pixels']*by_id[i]['sharpness'])
                groups.append({'id':max(g['id'] for g in groups)+1,'representative':a['id'],
                               'members':[a['id']],'builds':0})
                if not evidence.get('numeric_reviews'):
                    log['numeric_splits'].append({'a':a['id'],'b':b['id'],**evidence})
        for group in groups:
            representative = group['representative']
            alternatives = clear_alternates.get(representative, set()).intersection(group['members'])
            for index in sorted(alternatives.difference(clear_alternates)):
                if representative == group['representative'] or better_frame(
                        by_id[index], by_id[group['representative']], {'duplicate': True}):
                    group['representative'] = index
        log['ocr_new_representatives'] = read_new_representatives(groups, 'updated grouping')
        # OCR-backed comparison of candidate groups. Pixel similarity alone cannot
        # merge charts; numeric conflicts and changed titles block the merge.
        removed=set()
        for ai,ga in enumerate(groups):
            if ga['id'] in removed or by_id[ga['representative']].get('transition_review'):
                continue
            blocked = set().union(*(separations.get(i, ()) for i in ga['members']))
            for gb in groups[ai+1:]:
                if (gb['id'] in removed or by_id[gb['representative']].get('transition_review')
                        or not blocked.isdisjoint(gb['members'])):
                    continue
                a,b=by_id[ga['representative']],by_id[gb['representative']]
                if a.get('ocr',{}).get('status')!='ok' or b.get('ocr',{}).get('status')!='ok':
                    continue
                evidence=compare_text(a,b)
                ca,cb=evidence['text_cover_a'],evidence['text_cover_b']
                if (min(evidence['text_words_a'],evidence['text_words_b'])<10 or
                    evidence['heading_similarity']<.90 or max(ca,cb)<.94):
                    continue
                metric=compare(a,b,lambda item: native_for_comparison(item['id']))
                if metric['content_replacement']:
                    continue
                dup=(not metric['localized_reveal'] and min(ca,cb)>.96 and metric['changed']<.06
                     and min(metric['coverage_a'],metric['coverage_b'])>.87)
                build=(getattr(args,'builds','complete')=='complete' and max(ca,cb)>.965
                       and min(ca,cb)>.42 and max(metric['coverage_a'],metric['coverage_b'])>.89
                       and metric['changed']<.40 and metric['mae']<.19)
                if not (dup or build):
                    continue
                evidence = dict(compare_text(a, b, reread=True) if evidence['numeric_conflict'] else evidence)
                if dup:
                    inspect_local_change(a, b, metric, evidence)
                if evidence['numeric_conflict']:
                    remember_numeric(a, b, evidence)
                    separations[a['id']].add(b['id'])
                    separations[b['id']].add(a['id'])
                    blocked.add(b['id'])
                    continue
                if build and not dup:
                    choose_b=cb<ca-.025 # B has additional text: almost all of A occurs in B.
                    if abs(ca-cb)<=.025:
                        choose_b=better_frame(b,a,compare(b,a,lambda item: native_for_comparison(item['id'])))
                else:
                    choose_b=better_frame(b,a,{'duplicate':True})
                    if evidence.get('numeric_occlusions') in (['a'], ['b']):
                        choose_b = evidence['numeric_occlusions'] == ['a']
                if bool(a.get('crossfade')) != bool(b.get('crossfade')):
                    choose_b = not b.get('crossfade')
                if choose_b:
                    ga['representative']=b['id']
                for member in gb['members']:
                    blocked.update(separations.get(member, ()))
                ga['members'].extend(gb['members'])
                ga['builds']+=gb['builds']+int(build and not dup)
                removed.add(gb['id'])
                log['ocr_merges'].append({'a':a['id'],'b':b['id'],'reason':'ocr-build' if build and not dup else 'ocr-duplicate',**evidence})
        groups=[g for g in groups if g['id'] not in removed]
        owners={i:g for g in groups for i in g['members']}
        numeric_groups = {frozenset((owners[a]['id'], owners[b]['id'])) for a, b in numeric_reviews}
        filtered=[]
        for pair in candidates:
            first, second = (owners[pair[k]] for k in ('a', 'b'))
            if frozenset((first['id'], second['id'])) in numeric_groups:
                continue
            a, b = by_id[first['representative']], by_id[second['representative']]
            if a.get('ocr',{}).get('status')=='ok' and b.get('ocr',{}).get('status')=='ok':
                ev=compare_text(a,b)
                clearly_different=(ev['heading_similarity']<.65 and max(ev['text_cover_a'],ev['text_cover_b'])<.80)
                clearly_different=clearly_different or (ev['heading_similarity']<.65
                    and min(len(signature(a)[1]),len(signature(b)[1]))>8
                    and min(a['ocr']['mean_confidence'],b['ocr']['mean_confidence'])>.90)
                clearly_different=clearly_different or (max(ev['text_cover_a'],ev['text_cover_b'])<.86
                    and min(ev['text_words_a'],ev['text_words_b'])>=12 and pair['changed']>.06
                    and min(a['ocr']['mean_confidence'],b['ocr']['mean_confidence'])>.90)
                if ev['numeric_conflict'] or clearly_different:
                    log['suppressed_distinct_pairs']+=1
                    continue
            filtered.append(pair)
        candidates=filtered
    log['speaker_detection'] = audit_speaker_frames(source, groups, by_id, args, decode, log['ocr_enabled'])
    if log['ocr_enabled']:
        log['ocr_new_representatives'] += read_new_representatives(groups, 'final selection')
        results = [item['ocr'] for item in items if 'ocr' in item]
        log['ocr_images'] = len(results)
        log['ocr_errors'] = sum(result['status'] != 'ok' for result in results)
    owners = {member: group for group in groups for member in group['members']}
    numeric_issue_groups = set()
    reviewed_group_pairs = set()
    for evidence in numeric_reviews.values():
        a, b = owners[evidence['a']], owners[evidence['b']]
        group_pair = tuple(sorted((a['id'], b['id'])))
        if a.get('excluded') or b.get('excluded') or group_pair in reviewed_group_pairs:
            continue
        reviewed_group_pairs.add(group_pair)
        report_group = group_pair[0]
        log['numeric_reviews'].append({**evidence, 'group': report_group})
        numeric_issue_groups.add(report_group)
    # These flags describe actual source quality. They never imply upscaling can
    # recreate missing letters. Missing OCR is explicit, not a silent pass.
    for group in groups:
        item=by_id[group['representative']]
        reasons=[]
        if item.get('transition_review'):
            reasons.append(ONE_SIDED_REVIEW_REASON)
        if item.get('crossfade'):
            reasons.append('possible slide transition')
        if item['crop_confidence']<.78 or item.get('region_ambiguous'):
            reasons.append('check slide region')
        if item['crop_mode']=='projected-screen':
            reasons.append('check photographed screen')
        if min(item['slide_width'],item['slide_height'])<getattr(args,'min_short_edge',720) or max(item['slide_width'],item['slide_height'])<getattr(args,'min_long_edge',1280):
            reasons.append('low native resolution')
        if item['sharpness']<45:
            reasons.append('blur / little visible detail')
        ocr=item.get('ocr',{})
        if ocr.get('status')!='ok':
            reasons.append('OCR unavailable' if not log['ocr_enabled'] else 'OCR check failed')
        elif ocr['characters']<35:
            reasons.append('little readable text')
        elif ocr.get('median_text_height',0)<10 or ocr.get('mean_confidence',0)<.82:
            reasons.append('text readability needs checking')
        if group['builds'] and not log['ocr_enabled']:
            reasons.append('unverified build match')
        if group['id'] in numeric_issue_groups:
            reasons.append(NUMERIC_REVIEW)
        if group.get('auto_exclusion')=='speaker-only':
            reasons.append(SPEAKER_ONLY)
        elif group.pop('speaker_review',False):
            reasons.append(CHECK_SPEAKER)
        group['review_reasons']=reasons
        group['needs_review']=bool(reasons)
    for pair in candidates:
        owners=[next(g for g in groups if pair[k] in g['members']) for k in ('a','b')]
        if owners[0]['id']!=owners[1]['id'] and not any(g.get('excluded') for g in owners):
            for g in owners:
                if 'possible duplicate / build' not in g['review_reasons']:
                    g['review_reasons'].append('possible duplicate / build')
                g['needs_review']=True
    log['seconds']=round(time.perf_counter()-start,2)
    return groups,candidates,log
