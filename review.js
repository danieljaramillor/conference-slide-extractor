const images=new Map(data.images.map(i=>[i.id,i]));
const server=typeof REVIEW_TOKEN!=='undefined';
const storageKey='slide-review-'+(data.manifest_id||data.source+'-'+data.images.length);
let groups=structuredClone(data.groups),selected=new Set(),crops={},cropId=null,dragStart=null;
let resolvedPairs=new Set(), pairIndex=0, activeView='compare', lastPairDecision=null;
let reviewPlan, compareSelection=null, activeComparison=null, comparisonRequest=null, showHighlights=true;
let completedBuild=null,completedCandidate=null,completedComparison=null,completedRequest=null;
const comparisonCache=new Map();
let issueIndex=0, activeIssue=null, lastIssueDecision=null;
let deckReviewed=false, exporting=false, zoomTrigger=null, cropTrigger=null;
let revision=Date.now();
const $=id=>document.getElementById(id),owner=id=>groups.find(g=>g.members.includes(id));
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const needs=g=>!g.excluded&&g.needs_review&&!g.acknowledged;
const speakerExclusion='probable speaker-only frame (auto-excluded)';
const transitionExclusion='blended slide transition (auto-excluded)';
const possibleBlend='possible blend with unique content';
const automaticExclusions={
 'speaker-only':{reason:speakerExclusion,label:'Likely speaker-only'},
 transition:{reason:transitionExclusion,label:'Blended transition'}
};
const isTransition=id=>Boolean(images.get(id)?.transition);
function exclusionType(group){
 if(isTransition(group.representative))return 'transition';
 if(Object.hasOwn(automaticExclusions,group.auto_exclusion))return group.auto_exclusion;
 if((group.review_reasons||[]).includes(speakerExclusion))return 'speaker-only';
 return null;
}
function setExclusionType(group,type){
 if(type)group.auto_exclusion=type;
 else delete group.auto_exclusion;
}
const payload=()=>({version:2,source:data.source,manifest_id:data.manifest_id,groups,crops,revision,resolved_pairs:[...resolvedPairs]});
function notify(s){$('status').textContent=s}
function persist(){
 deckReviewed=false;
 $('download').hidden=true;
 revision=Math.max(Date.now(),revision+1);
 const body=JSON.stringify(payload());
 try{localStorage.setItem(storageKey,body)}catch{}
 if(server)fetch('decisions',{method:'POST',headers:{'Content-Type':'application/json','X-Review-Token':REVIEW_TOKEN},body})
  .then(response=>{if(!response.ok)throw Error('Save failed')})
  .catch(()=>notify('Could not save to disk. Keep this page open and save a copy from Review file when you reach Review deck.'));
}
function validate(d){
 if(!Array.isArray(d.groups)||d.groups.some(g=>!Number.isInteger(g.id)||g.id<1||!Array.isArray(g.members)||g.members.some(i=>!Number.isInteger(i))||!Number.isInteger(g.representative)||'excluded' in g&&typeof g.excluded!=='boolean'))throw Error('Invalid group data');
 if(d.groups.some(g=>g.review_reasons!=null&&(!Array.isArray(g.review_reasons)||g.review_reasons.some(reason=>typeof reason!=='string'))))throw Error('Invalid review notes');
 const ids=d.groups.flatMap(g=>g.members);
 if(d.source!==data.source||data.manifest_id&&d.manifest_id!==data.manifest_id||ids.length!==images.size||new Set(ids).size!==images.size||ids.some(id=>!images.has(id))||d.groups.some(g=>!g.members.includes(g.representative))||new Set(d.groups.map(g=>g.id)).size!==d.groups.length)throw Error('Decisions do not match this scan');
 for(const [id,b] of Object.entries(d.crops||{}))if(!images.has(+id)||!Array.isArray(b)||b.length!==4||!b.every(Number.isFinite)||!(0<=b[0]&&b[0]<b[2]&&b[2]<=1&&0<=b[1]&&b[1]<b[3]&&b[3]<=1))throw Error('Invalid crop');
 for(const [id,b] of Object.entries(d.crops||{})){const im=images.get(+id);if(Math.round(b[2]*im.width)-Math.round(b[0]*im.width)<2||Math.round(b[3]*im.height)-Math.round(b[1]*im.height)<2)throw Error('Crop must contain at least two pixels in each dimension')}
}
function nativeURL(id){return server?`native/${id}.png`+(crops[id]?'?crop='+crops[id].join(','):''):images.get(id).native_preview||`thumbs/${id}.jpg`}
function pairKey(pair){return [pair.a,pair.b].sort((a,b)=>a-b).join(':')}
function pairs(){
 const owners=new Map(groups.flatMap(g=>g.members.map(id=>[id,g])));
 const queue=new Map();
 for(const pair of data.review_pairs||[]){
  const left=owners.get(pair.a),right=owners.get(pair.b);
  if(!left||!right||left===right||left.excluded||right.excluded||resolvedPairs.has(pairKey(pair)))continue;
  const key=[left.id,right.id].sort((a,b)=>a-b).join(':');
  if(!queue.has(key))queue.set(key,{left,right,examples:[]});
  queue.get(key).examples.push(pair);
 }
 return [...queue.values()];
}
function buildReviewPlan(queue){
 const links=new Map();
 const order=new Map(queue.map((pair,index)=>[pair,index]));
 for(const pair of queue){
  for(const group of [pair.left,pair.right]){
   if(!links.has(group.id))links.set(group.id,[]);
   links.get(group.id).push(pair);
  }
 }
 const visited=new Set(),families=[],comparisons=[];
 for(const seed of queue){
  if(visited.has(seed.left.id))continue;
  const members=[],edges=new Set(),pending=[seed.left];
  while(pending.length){
   const group=pending.pop();
   if(visited.has(group.id))continue;
   visited.add(group.id);
   members.push(group);
   for(const pair of links.get(group.id)){
    edges.add(pair);
    const other=pair.left===group?pair.right:pair.left;
    if(!visited.has(other.id))pending.push(other);
   }
  }
  const familyIndex=families.length;
  const entries=[...edges].sort((a,b)=>order.get(a)-order.get(b)).map(pair=>{
   const entry={...pair,familyIndex,index:comparisons.length};
   comparisons.push(entry);
   return entry;
  });
  members.sort((a,b)=>a.id-b.id);
  families.push({members,entries});
 }
 return {families,comparisons};
}
function manuallySeparated(left,right){
 const a=new Set(left.members),b=new Set(right.members);
 for(const key of resolvedPairs){
  if(typeof key!=='string')continue;
  const [x,y]=key.split(':').map(Number);
  if(a.has(x)&&b.has(y)||a.has(y)&&b.has(x))return true;
 }
 return false;
}
function slideIssues(group){
 return (group.review_reasons||[]).filter(reason=>reason!=='possible duplicate / build');
}
function hasIssue(group){return needs(group)&&slideIssues(group).length>0}
function flagForReview(group,reason){
 const reasons=group.review_reasons||[];
 if(!reasons.includes(reason))group.review_reasons=[...reasons,reason];
 group.needs_review=true;
 group.acknowledged=false;
}
function selectRepresentative(group,id){
 if(exclusionType(group)==='transition'){
  setExclusionType(group,null);
  group.review_reasons=(group.review_reasons||[]).filter(reason=>reason!==transitionExclusion);
 }
 group.review_reasons=(group.review_reasons||[]).filter(reason=>reason!==possibleBlend);
 group.representative=id;
 if(isTransition(id)){
  setExclusionType(group,'transition');
  flagForReview(group,transitionExclusion);
 }else flagForReview(group,images.get(id)?.transition_review?possibleBlend:'check replacement capture');
}
function changeView(view){
 activeView=view;
 if(view==='slides')$('filter').value='included';
 $('search').value='';
 render();
}
function navigate(view){
 notify('');
 changeView(view);
 document.querySelector('[aria-current="step"]').focus();
 window.scrollTo(0,0);
}
function compareSide(group,side,allowCrop=false){
 const image=images.get(group.representative);
 const src=server||image.native_preview?nativeURL(image.id):`thumbs/${image.id}.jpg`;
 const crop=crops[image.id];
 const width=crop?Math.round(crop[2]*image.width)-Math.round(crop[0]*image.width):image.slide_width;
 const height=crop?Math.round(crop[3]*image.height)-Math.round(crop[1]*image.height):image.slide_height;
 const ratio=(width||image.width||16)/(height||image.height||9);
 return `<article class="compare-side" data-side="${side.toLowerCase()}">
  <div class="info"><h3><span class="side-marker" aria-hidden="true"></span>${side} <span class="side-info">Group ${group.id}</span></h3>
  <span class="badge ready build-recommendation" hidden>Suggested complete capture</span>
  <div class="side-info">${group.members.length} captures · ${width||'?'} × ${height||'?'} px</div>
  ${allowCrop&&server?`<button class="compare-crop" data-crop="${image.id}" aria-label="Adjust ${side.toLowerCase()} crop">Adjust crop</button>`:''}
  <div class="file">${esc(image.name.split(/[\\/]/).pop())}</div></div>
  <button class="preview-button compare-preview" data-zoom="${image.id}" aria-label="Enlarge ${side.toLowerCase()} slide">
   <span class="compare-image-frame" style="--slide-fit-width:${43*ratio}vh">
    <img src="${src}" alt="${side} slide preview"><span class="change-layer" aria-hidden="true"></span>
   </span>
  </button></article>`;
}
function variantCard(group,base,candidate){
 const image=images.get(group.representative);
 const label=group===base?'Base capture':group===candidate?'Compared capture':'Compare capture';
 return `<button data-variant="${group.id}" aria-label="${label}, group ${group.id}" aria-pressed="${group===candidate}" ${group===base?'disabled':''}>
  <img src="thumbs/${image.id}.jpg" loading="lazy" alt="">
  <span>Group ${group.id}${group===base?' · Base':''}</span>
 </button>`;
}
function validRegions(regions){
 if(!Array.isArray(regions))return [];
 return regions.filter(box=>Array.isArray(box)&&box.length===4&&box.every(Number.isFinite)
  &&0<=box[0]&&box[0]<box[2]&&box[2]<=1&&0<=box[1]&&box[1]<box[3]&&box[3]<=1);
}
function drawChanges(cards,side,regions){
 const layer=$(cards).querySelector(`[data-side="${side}"] .change-layer`);
 layer.replaceChildren();
 if(!showHighlights)return;
 for(const [x1,y1,x2,y2] of regions){
  const region=document.createElement('span');
  region.className='change-region';
  Object.assign(region.style,{left:`${100*x1}%`,top:`${100*y1}%`,width:`${100*(x2-x1)}%`,height:`${100*(y2-y1)}%`});
  layer.append(region);
 }
}
const reviewHighlights={cards:'compare-cards',toggle:'highlight-toggle',status:'highlight-status',sides:['base','candidate']};
const completedHighlights={cards:'completed-cards',toggle:'completed-highlight-toggle',status:'completed-highlight-status',sides:['selected','alternate']};
function paintHighlights(current,controls){
 const evidence=current.evidence;
 const [leftSide,rightSide]=controls.sides;
 const left=validRegions(evidence?.regions?.left),right=validRegions(evidence?.regions?.right);
 const available=evidence?.status==='ok'&&Boolean(left.length+right.length);
 const toggle=$(controls.toggle);
 const count=(regions,side)=>`${regions.length} changed area${regions.length===1?'':'s'} on the ${side}`;
 toggle.disabled=!available;
 toggle.setAttribute('aria-pressed',String(showHighlights&&available));
 $(controls.status).textContent=current.pending?'Checking image differences…':
  !server?'Open the batch in the app to highlight changes.':
  available&&showHighlights?`${count(left,leftSide)} · ${count(right,rightSide)}`:
  available?'Highlights hidden.':
  evidence?.status==='ok'?'No visible changes detected.':'Reliable highlights unavailable for these captures.';
 drawChanges(controls.cards,leftSide,available?left:[]);
 drawChanges(controls.cards,rightSide,available?right:[]);
}
function paintComparison(){
 const current=activeComparison;
 if(!current)return;
 const evidence=current.evidence;
 const a=current.left.representative,b=current.right.representative;
 const recorded=(current.entry?.examples||[]).filter(pair=>pair.a===a&&pair.b===b||pair.a===b&&pair.b===a);
 const numeric=Boolean(evidence?.numeric_conflict)||recorded.some(pair=>pair.numeric_conflict);
 const conflict=numeric||evidence?.relationship==='content-replaced'||
  (!crops[a]&&!crops[b]&&evidence?.status!=='ok'&&recorded.some(pair=>pair.content_replacement));
 const separate=manuallySeparated(current.left,current.right);
 const preferred=evidence?.status==='ok'&&!conflict?{
  'left-complete':'base','right-complete':'candidate'
 }[evidence.relationship]:null;
 let reason='Check how these captures differ before merging them.';
 if(!current.entry)reason='Choose a suggested comparison to save a decision about these captures.';
 else if(separate)reason='A previous decision keeps these captures separate.';
 else if(numeric)reason='A changed number or value was detected. Keep both slides to preserve it.';
 else if(conflict)reason='Both captures contain distinct content. Keep both slides to preserve it.';
 else if(preferred)reason=`The ${preferred} capture appears to include the content from both images.`;
 else if(evidence?.relationship==='same-content')reason='Both captures appear to have the same content.';
 $('pair-reason').textContent=reason;
 paintHighlights(current,reviewHighlights);
 for(const side of ['base','candidate']){
  $('compare-cards').querySelector(`[data-side="${side}"] .build-recommendation`).hidden=preferred!==side;
 }
 current.mergeBlocked=conflict||separate;
 $('keep-both').disabled=!current.entry;
 $('keep-both').classList.toggle('primary',!preferred||current.mergeBlocked);
 for(const [side,button] of [['base',$('keep-left')],['candidate',$('keep-right')]]){
  button.disabled=!current.entry||current.mergeBlocked||current.pending;
  button.textContent=`Merge, keep ${side}`;
  button.classList.toggle('primary',preferred===side&&!current.mergeBlocked);
 }
}
function loadComparison(current,paint,isCurrent){
 if(!server){paint();return null}
 const params=new URLSearchParams();
 for(const [key,group] of [['a',current.left],['b',current.right]]){
  const id=group.representative;
  params.set(key,id);
  if(crops[id])params.set(`crop_${key}`,crops[id].join(','));
 }
 const query=params.toString();
 if(comparisonCache.has(query)){
  current.evidence=comparisonCache.get(query);
  paint();
  return null;
 }
 const controller=new AbortController();
 const timeout=setTimeout(()=>controller.abort(),5000);
 current.pending=true;
 paint();
 fetch(`comparison?${query}`,{signal:controller.signal})
  .then(response=>response.ok?response.json():null).then(evidence=>{
   if(!evidence)return;
   comparisonCache.set(query,evidence);
   current.evidence=evidence;
  }).catch(()=>{}).finally(()=>{
   clearTimeout(timeout);
   current.pending=false;
   if(isCurrent())paint();
  });
 return controller;
}
function fitComparisonImages(cards){
 for(const image of $(cards).querySelectorAll('img')){
  const fit=()=>{
   if(image.naturalHeight)image.parentElement.style.setProperty('--slide-fit-width',`${43*image.naturalWidth/image.naturalHeight}vh`);
  };
  if(image.complete)fit();
  else image.addEventListener('load',fit,{once:true});
 }
}
function renderComparison(){
 comparisonRequest?.abort();
 const {families,comparisons}=reviewPlan;
 pairIndex=Math.max(0,Math.min(pairIndex,comparisons.length-1));
 const suggestion=comparisons[pairIndex];
 for(const id of ['pair-header','build-choices','build-evidence','pair-actions','pair-help','compare-cards'])$(id).hidden=!suggestion;
 $('comparison-empty').hidden=Boolean(suggestion);
 $('undo-pair').hidden=!lastPairDecision;
 if(!suggestion){activeComparison=null;return}
 const family=families[suggestion.familyIndex];
 const base=family.members.find(group=>group.id===compareSelection?.left)||suggestion.left;
 const candidate=family.members.find(group=>group.id===compareSelection?.right&&group!==base)||
  (suggestion.right===base?suggestion.left:suggestion.right);
 const entry=family.entries.find(pair=>
  pair.left===base&&pair.right===candidate||pair.left===candidate&&pair.right===base);
 if(entry)pairIndex=entry.index;
 const strip=$('family-strip'),scroll=activeComparison?.familyIndex===suggestion.familyIndex?strip.scrollLeft:0;
 activeComparison={left:base,right:candidate,entry,familyIndex:suggestion.familyIndex};
 $('pair-title').textContent=`Build group ${suggestion.familyIndex+1} of ${families.length}`;
 $('family-summary').textContent=`${family.members.length} versions · ${family.entries.length} comparison${family.entries.length===1?'':'s'} left`;
 $('family-picker').innerHTML=families.map((_,index)=>`<option value="${index}">Build group ${index+1}</option>`).join('');
 $('family-picker').value=suggestion.familyIndex;
 $('previous-family').disabled=suggestion.familyIndex===0;
 $('next-family').disabled=suggestion.familyIndex===families.length-1;
 $('base-picker').innerHTML=family.members.map(group=>`<option value="${group.id}">Group ${group.id}</option>`).join('');
 $('base-picker').value=base.id;
 $('pair-picker').innerHTML='<option value="" disabled>Choose a suggestion</option>'+
  family.entries.map(pair=>`<option value="${pair.index}">Groups ${pair.left.id} and ${pair.right.id}</option>`).join('');
 $('pair-picker').value=entry?entry.index:'';
 $('previous-pair').disabled=pairIndex===0;
 $('next-pair').disabled=pairIndex>=comparisons.length-1;
 strip.innerHTML=family.members.map(group=>variantCard(group,base,candidate)).join('');
 strip.scrollLeft=scroll;
 $('compare-cards').innerHTML=compareSide(base,'Base',true)+compareSide(candidate,'Candidate',true);
 fitComparisonImages('compare-cards');
 const current=activeComparison;
 comparisonRequest=loadComparison(current,paintComparison,()=>activeComparison===current);
}
function decidePair(decision){
 const {entry,left,right,mergeBlocked,pending,evidence}=activeComparison||{};
 if(!entry||decision!=='both'&&(mergeBlocked||pending))return;
 lastPairDecision={groups:structuredClone(groups),resolved:new Set(resolvedPairs),index:pairIndex,selection:compareSelection};
 if(decision==='both'){
  for(const example of entry.examples)resolvedPairs.add(pairKey(example));
 }else{
  const keep=decision==='left'?left:right;
  const other=decision==='left'?right:left;
  // Keep the earliest position in the export while honoring the chosen image.
  const first=groups.indexOf(keep)<groups.indexOf(other)?keep:other;
  const second=first===keep?other:keep;
  first.members=[...first.members,...second.members];
  first.representative=keep.representative;
  const isBuild=evidence?.relationship==='left-complete'||evidence?.relationship==='right-complete';
  first.builds=(first.builds||0)+(second.builds||0)+Number(isBuild);
  setExclusionType(first,exclusionType(keep));
  first.review_reasons=slideIssues(keep);
  first.needs_review=first.review_reasons.length>0;
  first.acknowledged=Boolean(keep.acknowledged);
  groups=groups.filter(group=>group!==second);
  selected.clear();
 }
 compareSelection=null;
 lastIssueDecision=null;
 persist();render();
 if($('pair-actions').hidden)$('continue-step').focus();
 notify(decision==='both'?'Both slides kept.':'Slides merged. Chosen capture kept.');
}
function restore(d){
 validate(d);
 groups=d.groups;crops=d.crops||{};
 resolvedPairs=new Set(d.resolved_pairs||[]);
 revision=Math.max(revision,d.revision||0);
 pairIndex=0;compareSelection=null;issueIndex=0;lastPairDecision=null;lastIssueDecision=null;selected.clear();
}

const issueDescriptions={
 [speakerExclusion]:'The scan flagged this as likely speaker footage. Confirm slide content is visible before marking it checked.',
 [transitionExclusion]:'The scan detected an overlap between two slides. Restore this capture only if it shows the slide you want to keep.',
 [possibleBlend]:'This frame appears to contain part of the previous slide. Keep it if that content matters. View the following slide to compare.',
 'check whether slide content is visible':'Confirm this frame contains a slide. Exclude it if it shows only people or the venue.',
 'check replacement capture':'Verify that the selected capture shows the whole slide with readable text and charts.',
 'possible slide transition':'Check for an overlap between slides. Choose a capture from before or after the transition.',
 'check slide region':'Check the crop. Include the whole slide and remove surrounding interface or stage footage.',
 'check photographed screen':'Check that the full projected slide is visible and its edges are readable.',
 'low native resolution':'The capture has few pixels. Enlarge it to check small text, or choose a higher-resolution capture.',
 'blur / little visible detail':'Check text and chart labels for blur. Choose a sharper capture if available.',
 'OCR unavailable':'Text recognition was unavailable. Inspect the slide for readable text and charts.',
 'OCR check failed':'Text recognition failed for this capture. Check small text, or choose another capture.',
 'little readable text':'The app found little readable text. Check that the crop includes any labels or text that should be visible.',
 'text readability needs checking':'Check small text, chart labels, and numbers against the image.',
 'unverified build match':'Check that the chosen capture includes everything revealed in this slide build.',
 'check adjusted crop':'Open the updated crop and make sure all the slide content is visible.',
 'check split capture':'Verify that this separated capture belongs in the deck as its own slide.',
 'confirm most complete capture':'Verify that the chosen capture contains all the information from this group.'
};
function memberMarkup(group,id,showSplit){
 const image=images.get(id);
 const transition=isTransition(id);
 const splitLabel=group.excluded||transition?'Split as excluded capture':'Split into a separate slide';
 return `<div class="member">
  <label><input type="radio" name="rep-${group.id}" data-rep="${id}" data-group="${group.id}" ${id===group.representative?'checked':''}> Use this capture</label>
  ${transition?'<span class="badge">Blended transition</span>':''}
  <button class="preview-button" data-zoom="${id}" aria-label="Enlarge alternate capture"><img loading="lazy" src="thumbs/${id}.jpg" alt=""></button>
  <div class="file">${esc(image.name)}</div>
  ${showSplit&&group.members.length>1?`<button data-split="${id}">${splitLabel}</button>`:''}
 </div>`;
}
function renderIssues(issues){
 issueIndex=Math.max(0,Math.min(issueIndex,issues.length-1));
 const group=issues[issueIndex];
 activeIssue=group||null;
 $('issue-content').hidden=!group;
 $('issue-empty').hidden=Boolean(group);
 $('undo-issue').hidden=!lastIssueDecision;
 if(!group)return;
 const image=images.get(group.representative);
 const preview=$('issue-preview');
 preview.dataset.zoom=image.id;
 const src=nativeURL(image.id);
 const current=preview.querySelector('img');
 if(current.getAttribute('src')!==src)current.src=src;
 $('issue-title').textContent=`Issue ${issueIndex+1} of ${issues.length} · Group ${group.id}`;
 $('issue-source').textContent=image.name.split(/[\\/]/).pop();
 const crop=crops[image.id];
 const width=crop?Math.round(crop[2]*image.width)-Math.round(crop[0]*image.width):image.slide_width;
 const height=crop?Math.round(crop[3]*image.height)-Math.round(crop[1]*image.height):image.slide_height;
 $('issue-size').textContent=`${width||'?'} × ${height||'?'} px · Select the image to enlarge`;
 $('issue-reasons').innerHTML=slideIssues(group).map(reason=>`<li>${esc(issueDescriptions[reason]||reason)}</li>`).join('');
 const following=$('issue-following');
 const after=image.transition_review?.after;
 following.hidden=!images.has(after);
 if(following.hidden)delete following.dataset.zoom;
 else following.dataset.zoom=after;
 $('issue-picker').innerHTML=issues.map((item,index)=>`<option value="${index}">${index+1}. Group ${item.id}</option>`).join('');
 $('issue-picker').value=issueIndex;
 $('previous-issue').disabled=issueIndex===0;
 $('next-issue').disabled=issueIndex>=issues.length-1;
 $('issue-crop').hidden=!server;
 $('issue-crop').dataset.crop=image.id;
 const alternatives=$('issue-alternates');
 if(alternatives.dataset.group!==String(group.id))alternatives.open=false;
 alternatives.dataset.group=group.id;
 alternatives.hidden=group.members.length<2;
 $('alternate-title').textContent=`Choose another capture (${group.members.length})`;
 $('issue-members').innerHTML=group.members.length>1?group.members.map(id=>memberMarkup(group,id,false)).join(''):'';
 const text=$('issue-ocr');
 if(text.dataset.image!==String(image.id))text.open=false;
 text.dataset.image=image.id;
 text.hidden=Boolean(crop)||image.ocr?.status!=='ok';
 text.querySelector('pre').textContent=text.hidden?'':image.ocr.text;
}
function decideIssue(exclude){
 if(!activeIssue)return;
 lastPairDecision=null;
 lastIssueDecision={group:activeIssue,excluded:activeIssue.excluded,acknowledged:activeIssue.acknowledged};
 if(exclude)activeIssue.excluded=true;
 else activeIssue.acknowledged=true;
 persist();render();
 notify(exclude?'Slide excluded from export.':'Slide marked checked.');
 if(activeIssue){
  $('issue-title').focus({preventScroll:true});
  $('issue-title').scrollIntoView({block:'nearest'});
 }else $('continue-step').focus();
}

const stepOrder=['compare','issues','slides','export'];
const sections=['comparison-view','issue-view','slide-view','export-view'];
const stepNames={compare:'Review builds',issues:'Slide issues',slides:'Review deck',export:'Export'};
function renderWorkflow(familyCount,issueCount,included,excluded){
 const step=stepOrder.indexOf(activeView);
 const firstPending=familyCount?'compare':issueCount?'issues':null;
 $('pair-count').textContent=familyCount?`${familyCount} group${familyCount===1?'':'s'} left`:'Complete';
 $('issue-count').textContent=issueCount?`${issueCount} left`:'Complete';
 $('deck-count').textContent=deckReviewed?'Reviewed':`${included} included`;
 $('export-step-status').textContent=deckReviewed&&included?'Ready':'After review';
 const locked=[false,Boolean(familyCount),Boolean(firstPending),!deckReviewed||!included||Boolean(firstPending)];
 stepOrder.forEach((view,index)=>{
  const button=$(view+'-tab');
  const current=index===step;
  button.disabled=exporting||locked[index]&&!current;
  if(current)button.setAttribute('aria-current','step');
  else button.removeAttribute('aria-current');
  $(sections[index]).hidden=!current;
 });
 $('back-step').hidden=step===0;
 $('back-step').disabled=exporting;
 $('back-step').textContent=step>0?`Back to ${stepNames[stepOrder[step-1]].toLowerCase()}`:'Back';
 const next=$('continue-step');
 const earlierPending=firstPending&&stepOrder.indexOf(firstPending)<step;
 const nextView=earlierPending?firstPending:stepOrder[step+1];
 next.hidden=step===3;
 next.disabled=activeView===firstPending||(step===2&&!included&&!firstPending);
 next.dataset.next=nextView||'';
 next.textContent=earlierPending?`Return to ${stepNames[nextView].toLowerCase()}`:
  ({issues:'Continue to slide issues',slides:'Continue to deck review',export:'Continue to export'})[nextView]||'Continue';
 $('step-hint').textContent=activeView===firstPending?
  `Resolve the remaining ${activeView==='compare'?'build comparisons':'slide issues'} to continue.`:
  step===2&&!included&&!firstPending?(excluded?'Restore an excluded capture to continue.':'There are no slides to export.'):'';
 $('step-hint').hidden=!$('step-hint').textContent;
 $('export').hidden=!server;
 $('export').disabled=exporting||locked[3];
 $('export-offline').hidden=server;
 $('export-summary').textContent=`${included} slides included · ${excluded} excluded`;
}
function deckCard(group,position){
 const image=images.get(group.representative);
 const title=position?`Slide ${position}`:`Excluded · Group ${group.id}`;
 const exclusion=automaticExclusions[exclusionType(group)];
 const badges=group.excluded?
  `<span class="badge">${exclusion?.label||'Excluded'}</span>`:
  needs(group)?(group.review_reasons||[]).map(reason=>`<span class="badge">${esc(reason)}</span>`).join(''):
  '<span class="badge ready">Ready</span>';
 const exclusionControl=group.excluded?
  `<button data-restore="${group.id}">Restore to deck</button>`:
  `<label><input type="checkbox" data-exclude="${group.id}"> Exclude</label>
   <label><input type="checkbox" data-checked="${group.id}" ${group.acknowledged?'checked':''}> Checked</label>`;
 return `<article class="card ${selected.has(group.id)?'selected':''}" id="group-${group.id}">
  <button class="preview-button" data-zoom="${image.id}" aria-label="Enlarge ${title.toLowerCase()}"><img loading="lazy" src="${crops[image.id]&&server?nativeURL(image.id):`thumbs/${image.id}.jpg`}" alt=""></button>
  <div class="info">
   <h2><label><input type="checkbox" data-select="${group.id}" ${selected.has(group.id)?'checked':''}> ${title}</label></h2>
   <div class="side-info">${group.members.length} captures · ${image.slide_width||'?'} × ${image.slide_height||'?'} px</div>
   ${badges}
   <div class="controls">
    ${exclusionControl}
    <button data-crop="${image.id}" ${server?'':'disabled title="Use the local review server to edit crops"'}>Adjust crop</button>
   </div>
   <div class="file">${esc(image.name)}</div>
   ${image.ocr?.status==='ok'?`<details><summary>Extracted text</summary><pre>${esc(image.ocr.text)}</pre></details>`:''}
   <details class="inspect"><summary>Inspect captures / change selected image</summary>
    <div class="members">${group.members.map(id=>memberMarkup(group,id,true)).join('')}</div>
   </details>
  </div>
 </article>`;
}
function renderCompletedBuild(included){
 completedRequest?.abort();
 completedComparison=null;
 const builds=included.filter(group=>group.builds>0&&group.members.length>1);
 const build=builds.find(group=>group.id===completedBuild);
 $('completed-picker-label').hidden=!builds.length;
 $('completed-build-picker').innerHTML='<option value="">Choose a build</option>'+
  builds.map(group=>`<option value="${group.id}">Group ${group.id} · ${group.members.length} captures</option>`).join('');
 $('completed-build-picker').value=build?.id||'';
 $('completed-build').hidden=!build;
 if(!build){
  $('completed-cards').replaceChildren();
  return;
 }
 const imagesInBuild=build.members.filter(id=>id!==build.representative).sort((a,b)=>a-b);
 if(!imagesInBuild.includes(completedCandidate))completedCandidate=imagesInBuild[0];
 $('completed-build-title').textContent=`Grouped build · Group ${build.id}`;
 $('completed-capture-picker').innerHTML=imagesInBuild.map(id=>
  `<option value="${id}">${esc(images.get(id).name.split(/[\\/]/).pop())}</option>`).join('');
 $('completed-capture-picker').value=completedCandidate;
 const alternate={...build,representative:completedCandidate};
 const current={left:build,right:alternate};
 completedComparison=current;
 $('completed-cards').innerHTML=compareSide(build,'Selected')+compareSide(alternate,'Alternate');
 fitComparisonImages('completed-cards');
 completedRequest=loadComparison(current,()=>paintHighlights(current,completedHighlights),()=>completedComparison===current);
}
function renderDeck(included){
 const expanded=[...document.querySelectorAll('.inspect[open]')].map(el=>el.closest('.card').id);
 const term=$('search').value.trim().toLowerCase(),filter=$('filter').value;
 const excludedCount=groups.length-included.length;
 const reviewingExclusions=filter==='excluded';
 renderCompletedBuild(reviewingExclusions?[]:included);
 const showExcluded=$('show-excluded');
 showExcluded.hidden=!excludedCount&&!reviewingExclusions;
 showExcluded.textContent=reviewingExclusions?
  `Back to included slides (${included.length})`:`Review excluded captures (${excludedCount})`;
 $('deck-title').textContent=reviewingExclusions?'Review excluded captures':'Review the deck';
 $('deck-description').textContent=reviewingExclusions?
  'Restore a capture if it contains a slide you want in the export.':
  'Check the included slides in the order they’ll appear in the ZIP.';
 const matches={
  included:group=>!group.excluded,
  all:()=>true,
  review:hasIssue,
  ready:group=>!group.excluded&&!needs(group),
  excluded:group=>group.excluded,
  multi:group=>group.members.length>1,
  full:group=>images.get(group.representative).crop_mode==='full'
 };
 const positions=new Map(included.map((group,index)=>[group.id,index+1]));
 const visible=groups.filter(group=>matches[filter](group)&&(!term||group.members.some(id=>images.get(id).name.toLowerCase().includes(term))));
 $('grid').innerHTML=visible.map(group=>deckCard(group,positions.get(group.id))).join('')||'<p>No slides match this view.</p>';
 for(const id of expanded){const target=$(id)?.querySelector('.inspect');if(target)target.open=true}
}
function render(){
 const issues=groups.filter(hasIssue);
 const included=groups.filter(group=>!group.excluded);
 const excluded=groups.length-included.length;
 reviewPlan=buildReviewPlan(pairs());
 $('stats').textContent=`${included.length} slides included · ${excluded} excluded · ${data.images.length}/${data.stats.input_images} captures processed`;
 $('errors').hidden=data.errors.length===0;
 $('errors').querySelector('summary').textContent=`${data.errors.length} images could not be read`;
 $('errors').querySelector('pre').textContent=data.errors.map(error=>error.name+': '+error.error).join('\n');
 renderWorkflow(reviewPlan.families.length,issues.length,included.length,excluded);
 if(activeView==='compare')renderComparison();
 else{
  activeComparison=null;
  comparisonRequest?.abort();
 }
 if(activeView==='issues')renderIssues(issues);
 if(activeView==='slides')renderDeck(included);
 else{
  completedRequest?.abort();
  completedComparison=null;
  $('grid').replaceChildren();
 }
}
for(const step of stepOrder)$(step+'-tab').onclick=()=>navigate(step);
$('back-step').onclick=()=>{
 const previous=stepOrder[stepOrder.indexOf(activeView)-1];
 navigate($(previous+'-tab').disabled?'compare':previous);
};
$('continue-step').onclick=()=>{
 const next=$('continue-step').dataset.next;
 if(next==='export')deckReviewed=true;
 navigate(next);
};
function goToSuggestion(index){
 if(!reviewPlan.comparisons[index])return;
 pairIndex=index;
 compareSelection=null;
 renderComparison();
}
function goToFamily(index){
 const first=reviewPlan.families[index]?.entries[0];
 if(first)goToSuggestion(first.index);
}
$('family-picker').onchange=e=>goToFamily(Number(e.target.value));
$('previous-family').onclick=()=>goToFamily(activeComparison.familyIndex-1);
$('next-family').onclick=()=>goToFamily(activeComparison.familyIndex+1);
$('pair-picker').onchange=e=>goToSuggestion(Number(e.target.value));
$('previous-pair').onclick=()=>goToSuggestion(pairIndex-1);
$('next-pair').onclick=()=>goToSuggestion(pairIndex+1);
$('base-picker').onchange=e=>{
 const left=Number(e.target.value);
 compareSelection={left,right:activeComparison.right.id===left?activeComparison.left.id:activeComparison.right.id};
 renderComparison();
};
$('family-strip').onclick=e=>{
 const button=e.target.closest('[data-variant]');
 if(!button||button.disabled)return;
 compareSelection={left:activeComparison.left.id,right:Number(button.dataset.variant)};
 renderComparison();
 $('family-strip').querySelector(`[data-variant="${compareSelection.right}"]`)?.focus({preventScroll:true});
};
$('highlight-toggle').onclick=()=>{showHighlights=!showHighlights;paintComparison()};
$('completed-build-picker').onchange=e=>{
 completedBuild=Number(e.target.value)||null;
 completedCandidate=null;
 renderCompletedBuild(groups.filter(group=>!group.excluded));
};
$('completed-capture-picker').onchange=e=>{
 completedCandidate=Number(e.target.value);
 renderCompletedBuild(groups.filter(group=>!group.excluded));
};
$('close-completed-build').onclick=()=>{
 completedBuild=null;
 renderCompletedBuild(groups.filter(group=>!group.excluded));
 $('completed-build-picker').focus();
};
$('completed-highlight-toggle').onclick=()=>{
 showHighlights=!showHighlights;
 if(completedComparison)paintHighlights(completedComparison,completedHighlights);
};
$('keep-left').onclick=()=>decidePair('left');
$('keep-both').onclick=()=>decidePair('both');
$('keep-right').onclick=()=>decidePair('right');
$('undo-pair').onclick=()=>{
 if(!lastPairDecision)return;
 ({groups,resolved:resolvedPairs,index:pairIndex,selection:compareSelection}=lastPairDecision);
 lastPairDecision=null;selected.clear();persist();render();notify('Last pair decision undone.');
};
$('issue-picker').onchange=event=>{issueIndex=Number(event.target.value);renderIssues(groups.filter(hasIssue))};
$('previous-issue').onclick=()=>{issueIndex--;renderIssues(groups.filter(hasIssue))};
$('next-issue').onclick=()=>{issueIndex++;renderIssues(groups.filter(hasIssue))};
$('issue-check').onclick=()=>decideIssue(false);
$('issue-exclude').onclick=()=>decideIssue(true);
$('undo-issue').onclick=()=>{
 if(!lastIssueDecision)return;
 const {group,excluded,acknowledged}=lastIssueDecision;
 group.excluded=Boolean(excluded);
 group.acknowledged=Boolean(acknowledged);
 issueIndex=groups.filter(hasIssue).indexOf(group);
 lastIssueDecision=null;
 persist();render();notify('Last issue decision undone.');
 $('issue-title').focus();
};
$('search').oninput=render;$('filter').onchange=render;
$('show-excluded').onclick=()=>{
 $('filter').value=$('filter').value==='excluded'?'included':'excluded';
 $('search').value='';
 render();
 $('deck-title').focus();
};
function changeCapture(e){
 const d=e.target.dataset;
 if(d.select){const id=+d.select;e.target.checked?selected.add(id):selected.delete(id);e.target.closest('.card').classList.toggle('selected',e.target.checked);return}
 if(d.rep){
  const group=groups.find(group=>group.id===+d.group);
  selectRepresentative(group,+d.rep);
 }
 if(d.exclude)groups.find(g=>g.id===+d.exclude).excluded=e.target.checked;
 if(d.checked)groups.find(g=>g.id===+d.checked).acknowledged=e.target.checked;
 if(!d.rep&&!d.exclude&&!d.checked)return;
 lastPairDecision=null;
 lastIssueDecision=null;
 persist();render();
 if(activeView==='issues')$('issue-preview').focus();
 notify(d.rep?'Capture updated. Check the selected image.':'Changes saved in this browser.');
}
$('grid').onchange=changeCapture;
$('issue-members').onchange=changeCapture;
document.addEventListener('click',e=>{
 const d=e.target.dataset;
 const zoom=e.target.closest('[data-zoom]');
 if(zoom){
  const id=+zoom.dataset.zoom;
  zoomTrigger=zoom;
  $('lightbox').style.display='flex';
  $('lightbox').querySelector('img').src=nativeURL(id);
  $('caption').textContent=images.get(id).name+(server||images.get(id).native_preview?' · native-resolution preview':' · thumbnail; start the local server for the native image');
  $('lightbox').querySelector('button').focus();
 }
 if(d.restore){
  const group=groups.find(group=>group.id===+d.restore);
  group.excluded=false;
  selected.delete(group.id);
  const exclusion=automaticExclusions[exclusionType(group)];
  const reason=exclusion?.reason||(images.get(group.representative).transition_review?possibleBlend:null);
  if(reason)flagForReview(group,reason);
  lastPairDecision=null;
  lastIssueDecision=null;
  persist();render();
  ($('grid').querySelector('[data-restore]')||$('show-excluded')).focus();
  notify(reason?'Capture restored. Check it in Slide issues before exporting.':'Capture restored to the deck.');
 }
 if(d.split){
  const id=+d.split,group=owner(id);
  const transition=isTransition(id);
  const child={id:Math.max(...groups.map(group=>group.id))+1,representative:id,members:[id],builds:0,
   excluded:Boolean(group.excluded||transition),needs_review:true,review_reasons:['check split capture']};
  if(transition){
   child.auto_exclusion='transition';
   child.review_reasons.push(transitionExclusion);
  }else if(images.get(id).transition_review)child.review_reasons.push(possibleBlend);
  group.members=group.members.filter(member=>member!==id);
  if(group.representative===id)selectRepresentative(group,group.members[0]);
  groups.push(child);
  lastPairDecision=null;
  lastIssueDecision=null;
  persist();render();
  notify(child.excluded?'Capture split and kept excluded.':'Capture split.');
 }
 if(d.crop&&server){
  cropId=+d.crop;
  cropTrigger=e.target;
  $('crop-error').hidden=true;
  $('cropdialog').style.display='flex';
  $('cropsource').src=`source/${cropId}.png`;
  cropFields(crops[cropId]||images.get(cropId).crop);
  $('cx1').focus();
 }
});
$('merge').onclick=()=>{
 const chosen=groups.filter(g=>selected.has(g.id));if(chosen.length<2)return notify('Select at least two groups.');const target=chosen[0];
 const included=chosen.filter(group=>!group.excluded);
 const candidates=included.length?included:chosen;
 const clean=candidates.filter(group=>exclusionType(group)!=='transition'&&!images.get(group.representative).transition_review);
 const rank=image=>[image.ocr?.characters||image.detail||0,image.slide_pixels||0];
 const best=(clean.length?clean:candidates).map(group=>({group,score:rank(images.get(group.representative))}))
  .sort((a,b)=>b.score[0]-a.score[0]||b.score[1]-a.score[1])[0].group;
 const type=exclusionType(best);
 target.members=chosen.flatMap(g=>g.members);target.builds=chosen.reduce((sum,group)=>sum+(group.builds||0),0);
 target.representative=best.representative;
 setExclusionType(target,type);
 target.excluded=!included.length;
 target.review_reasons=[];
 if(type==='transition')flagForReview(target,transitionExclusion);
 if(images.get(target.representative).transition_review)flagForReview(target,possibleBlend);
 flagForReview(target,'confirm most complete capture');
 groups=groups.filter(g=>!selected.has(g.id)||g.id===target.id);
 selected.clear();
 lastPairDecision=null;
 lastIssueDecision=null;
 persist();render();
 notify(target.excluded?'Groups merged. The merged group is still excluded.':'Groups merged. Confirm the most complete capture in Slide issues.');
};
$('ack').onclick=()=>{if(!selected.size)return notify('Select at least one slide.');for(const g of groups)if(selected.has(g.id))g.acknowledged=true;selected.clear();lastPairDecision=null;lastIssueDecision=null;persist();render();notify('Marked checked.')};
$('save').onclick=()=>{const url=URL.createObjectURL(new Blob([JSON.stringify(payload(),null,2)],{type:'application/json'})),a=document.createElement('a');a.href=url;a.download='review-decisions.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);notify('Decisions downloaded.')};
$('load').onclick=()=>$('loadfile').click();$('loadfile').onchange=async e=>{try{restore(JSON.parse(await e.target.files[0].text()));persist();render();notify('Decisions loaded')}catch(err){notify(err.message)}};
function exportFilename(){
 const basename=data.source.split(/[\\/]/).pop().replace(/\.zip$/i,'');
 const safe=basename.replace(/[<>:"/\\|?*\u0000-\u001f]/g,'-');
 return `${safe==='input'||!safe?'conference':safe}-slides.zip`;
}

async function pickExportFile(){
 if(typeof window.showSaveFilePicker!=='function')return null;
 try{
  return await window.showSaveFilePicker({suggestedName:exportFilename(),
   types:[{description:'ZIP archive',accept:{'application/zip':['.zip']}}]});
 }catch(error){
  if(error.name==='AbortError')throw error;
  // Embedded browsers and restricted contexts may only support downloads.
  return null;
 }
}

if(typeof window.showSaveFilePicker==='function')$('export').textContent='Export ZIP…';
$('export').onclick=async()=>{
 if(activeView!=='export'||$('export').disabled)return;
 exporting=true;
 render();
 try{
  // Open Save As while the click still grants browser permission.
  const destination=await pickExportFile();
  $('download').hidden=true;
  notify('Preparing the ZIP with slides, originals, and text…');
  const response=await fetch('export',{method:'POST',headers:{'Content-Type':'application/json','X-Review-Token':REVIEW_TOKEN},body:JSON.stringify(payload())});
  const result=await response.json();
  if(!response.ok)throw Error(result.error);
  $('download').href=result.download;
  $('download').download=exportFilename();
  $('download').hidden=false;
  if(!destination){
   notify(`ZIP ready with ${result.count} slides. Select Download export ZIP to save it.`);
   return;
  }
  try{
   const zip=await fetch(result.download);
   if(!zip.ok||!zip.body)throw Error('The export ZIP could not be opened.');
   await zip.body.pipeTo(await destination.createWritable());
   notify(`Saved ${result.count} slides in ${destination.name}.`);
  }catch(error){
   notify(`Could not save to the chosen location. Use Download export ZIP to save the completed file. ${error.message}`);
  }
 }catch(error){
  notify(error.name==='AbortError'?'Export canceled.':error.message);
 }finally{exporting=false;render()}
};
function closeZoom(){
 $('lightbox').style.display='none';
 zoomTrigger?.focus();
 zoomTrigger=null;
}
$('lightbox').querySelector('button').onclick=closeZoom;
function drawCrop(b){Object.assign($('cropselection').style,{left:b[0]*100+'%',top:b[1]*100+'%',width:(b[2]-b[0])*100+'%',height:(b[3]-b[1])*100+'%'})}
function cropValues(){return ['cx1','cy1','cx2','cy2'].map(id=>+$(id).value/100)}
function cropFields(b){['cx1','cy1','cx2','cy2'].forEach((id,j)=>$(id).value=(b[j]*100).toFixed(2));drawCrop(b)}
for(const id of ['cx1','cy1','cx2','cy2'])$(id).oninput=()=>drawCrop(cropValues());
function pointer(e){const r=$('cropsource').getBoundingClientRect();return [Math.max(0,Math.min(1,(e.clientX-r.left)/r.width)),Math.max(0,Math.min(1,(e.clientY-r.top)/r.height))]}
$('cropbox').onpointerdown=e=>{e.preventDefault();dragStart=pointer(e);$('cropbox').setPointerCapture(e.pointerId)};
$('cropbox').onpointermove=e=>{if(dragStart){const p=pointer(e);cropFields([Math.min(p[0],dragStart[0]),Math.min(p[1],dragStart[1]),Math.max(p[0],dragStart[0]),Math.max(p[1],dragStart[1])])}};
$('cropbox').onpointerup=()=>dragStart=null;$('cropbox').onpointercancel=()=>dragStart=null;
function closeCrop(){
 $('cropdialog').style.display='none';
 cropTrigger?.focus();
 cropTrigger=null;
}
$('cancelcrop').onclick=closeCrop;
$('applycrop').onclick=()=>{
 const box=cropValues();
 try{
  validate({...payload(),crops:{...crops,[cropId]:box}});
 }catch(error){
  $('crop-error').textContent=error.message;
  $('crop-error').hidden=false;
  return;
 }
 crops[cropId]=box;
 const group=owner(cropId);
 const type=exclusionType(group);
 setExclusionType(group,type);
 group.review_reasons=[];
 flagForReview(group,'check adjusted crop');
 if(type==='transition')flagForReview(group,transitionExclusion);
 if(images.get(group.representative).transition_review)flagForReview(group,possibleBlend);
 issueIndex=groups.filter(hasIssue).indexOf(group);
 lastIssueDecision=null;
 lastPairDecision=null;
 closeCrop();
 persist();render();
 const updatedCrop=activeView==='compare'?$('compare-cards').querySelector(`[data-crop="${cropId}"]`):null;
 (updatedCrop||(activeView==='issues'?$('issue-preview'):$('continue-step'))).focus();
 notify('Crop adjusted. Text will be refreshed on export.');
};
document.addEventListener('keydown',e=>{if(e.key==='Escape'){closeZoom();closeCrop()}});
let localDecision=null;
try{localDecision=JSON.parse(localStorage.getItem(storageKey))}catch{}
const savedDecision=typeof SAVED_REVIEW!=='undefined'?SAVED_REVIEW:null;
for(const candidate of [localDecision,savedDecision].filter(Boolean).sort((a,b)=>(b.revision||0)-(a.revision||0))){
 try{restore(candidate);break}catch{}
}
if(pairs().length)changeView('compare');
else if(groups.some(hasIssue))changeView('issues');
else changeView('slides');
