const images=new Map(data.images.map(i=>[i.id,i]));
const server=typeof REVIEW_TOKEN!=='undefined';
const storageKey='slide-review-'+(data.manifest_id||data.source+'-'+data.images.length);
let groups=structuredClone(data.groups),selected=new Set(),crops={},cropId=null,dragStart=null;
let resolvedPairs=new Set(), pairIndex=0, activeView='compare', lastPairDecision=null;
let issueIndex=0, activeIssue=null, lastIssueDecision=null;
let deckReviewed=false, exporting=false, zoomTrigger=null, cropTrigger=null;
let revision=Date.now();
const $=id=>document.getElementById(id),owner=id=>groups.find(g=>g.members.includes(id));
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const needs=g=>!g.excluded&&g.needs_review&&!g.acknowledged;
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
 for(const pair of data.review_pairs){
  const left=owners.get(pair.a),right=owners.get(pair.b);
  if(!left||!right||left===right||left.excluded||right.excluded||resolvedPairs.has(pairKey(pair)))continue;
  const key=[left.id,right.id].sort((a,b)=>a-b).join(':');
  if(!queue.has(key))queue.set(key,{left,right,examples:[]});
  queue.get(key).examples.push(pair);
 }
 return [...queue.values()];
}
function slideIssues(group){
 return (group.review_reasons||[]).filter(reason=>reason!=='possible duplicate / build');
}
function hasIssue(group){return needs(group)&&slideIssues(group).length>0}
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
function compareSide(group,side){
 const image=images.get(group.representative);
 const src=server||image.native_preview?nativeURL(image.id):`thumbs/${image.id}.jpg`;
 return `<article class="compare-side">
  <div class="info"><h3><span class="side-marker" aria-hidden="true"></span>${side} · Group ${group.id}</h3>
  <div class="side-info">${group.members.length} captures · ${image.slide_width||'?'} × ${image.slide_height||'?'} px</div>
  <div class="file">${esc(image.name.split(/[\\/]/).pop())}</div></div>
  <button class="preview-button" data-zoom="${image.id}" aria-label="Enlarge ${side.toLowerCase()} slide"><img src="${src}" alt="${side} slide preview"></button></article>`;
}
function renderComparison(queue){
 pairIndex=Math.max(0,Math.min(pairIndex,queue.length-1));
 const current=queue[pairIndex];
 for(const id of ['pair-header','pair-actions','pair-help','compare-cards'])$(id).hidden=!current;
 $('comparison-empty').hidden=Boolean(current);
 $('undo-pair').hidden=!lastPairDecision;
 if(!current)return;
 $('pair-title').textContent=`Pair ${pairIndex+1} of ${queue.length} remaining`;
 const example=current.examples[0];
 const coverageGap=Math.abs((example.coverage_a??0)-(example.coverage_b??0));
 let reason='These captures look similar. Open either image to inspect details.';
 if(example.content_replacement)reason='Content changed in both captures. Check for different chart or table panels.';
 else if(coverageGap>.10&&example.title_changed<.025)reason='Possible build. Check whether one capture contains the complete slide.';
 $('pair-reason').textContent=reason;
 $('pair-picker').innerHTML=queue.map((pair,i)=>`<option value="${i}">${i+1}. Groups ${pair.left.id} and ${pair.right.id}</option>`).join('');
 $('pair-picker').value=pairIndex;
 $('previous-pair').disabled=pairIndex===0;
 $('next-pair').disabled=pairIndex>=queue.length-1;
 $('compare-cards').innerHTML=compareSide(current.left,'Left')+compareSide(current.right,'Right');
}
function decidePair(decision){
 const entry=pairs()[pairIndex];
 if(!entry)return;
 lastPairDecision={groups:structuredClone(groups),resolved:new Set(resolvedPairs),index:pairIndex};
 if(decision==='both'){
  for(const example of entry.examples)resolvedPairs.add(pairKey(example));
 }else{
  const keep=decision==='left'?entry.left:entry.right;
  const other=decision==='left'?entry.right:entry.left;
  // Keep the earliest position in the export while honoring the chosen image.
  const first=groups.indexOf(keep)<groups.indexOf(other)?keep:other;
  const second=first===keep?other:keep;
  first.members=[...first.members,...second.members];
  first.representative=keep.representative;
  first.builds=(first.builds||0)+(second.builds||0);
  first.review_reasons=slideIssues(keep);
  first.needs_review=first.review_reasons.length>0;
  first.acknowledged=Boolean(keep.acknowledged);
  groups=groups.filter(group=>group!==second);
  selected.clear();
 }
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
 pairIndex=0;issueIndex=0;lastPairDecision=null;lastIssueDecision=null;selected.clear();
}

const issueDescriptions={
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
 return `<div class="member">
  <label><input type="radio" name="rep-${group.id}" data-rep="${id}" data-group="${group.id}" ${id===group.representative?'checked':''}> Use this capture</label>
  <button class="preview-button" data-zoom="${id}" aria-label="Enlarge alternate capture"><img loading="lazy" src="thumbs/${id}.jpg" alt=""></button>
  <div class="file">${esc(image.name)}</div>
  ${showSplit&&group.members.length>1?`<button data-split="${id}">Split into a separate slide</button>`:''}
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
const stepNames={compare:'Compare pairs',issues:'Slide issues',slides:'Review deck',export:'Export'};
function renderWorkflow(pairCount,issueCount,included,excluded){
 const step=stepOrder.indexOf(activeView);
 const firstPending=pairCount?'compare':issueCount?'issues':null;
 $('pair-count').textContent=pairCount?`${pairCount} left`:'Complete';
 $('issue-count').textContent=issueCount?`${issueCount} left`:'Complete';
 $('deck-count').textContent=deckReviewed?'Reviewed':`${included} included`;
 $('export-step-status').textContent=deckReviewed&&included?'Ready':'After review';
 const locked=[false,Boolean(pairCount),Boolean(firstPending),!deckReviewed||!included||Boolean(firstPending)];
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
  `Resolve the remaining ${activeView==='compare'?'pairs':'slide issues'} to continue.`:
  step===2&&!included&&!firstPending?'Use the Excluded filter to restore a slide before exporting.':'';
 $('step-hint').hidden=!$('step-hint').textContent;
 $('export').hidden=!server;
 $('export').disabled=exporting||locked[3];
 $('export-offline').hidden=server;
 $('export-summary').textContent=`${included} slides included · ${excluded} excluded`;
}
function deckCard(group,position){
 const image=images.get(group.representative);
 const title=position?`Slide ${position}`:`Excluded · Group ${group.id}`;
 const badges=group.excluded?'<span class="badge">Excluded</span>':
  needs(group)?(group.review_reasons||[]).map(reason=>`<span class="badge">${esc(reason)}</span>`).join(''):
  '<span class="badge ready">Ready</span>';
 return `<article class="card ${selected.has(group.id)?'selected':''}" id="group-${group.id}">
  <button class="preview-button" data-zoom="${image.id}" aria-label="Enlarge ${title.toLowerCase()}"><img loading="lazy" src="${crops[image.id]&&server?nativeURL(image.id):`thumbs/${image.id}.jpg`}" alt=""></button>
  <div class="info">
   <h2><label><input type="checkbox" data-select="${group.id}" ${selected.has(group.id)?'checked':''}> ${title}</label></h2>
   <div class="side-info">${group.members.length} captures · ${image.slide_width||'?'} × ${image.slide_height||'?'} px</div>
   ${badges}
   <div class="controls">
    <label><input type="checkbox" data-exclude="${group.id}" ${group.excluded?'checked':''}> Exclude</label>
    <label><input type="checkbox" data-checked="${group.id}" ${group.acknowledged?'checked':''}> Checked</label>
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
function renderDeck(included){
 const expanded=[...document.querySelectorAll('.inspect[open]')].map(el=>el.closest('.card').id);
 const term=$('search').value.trim().toLowerCase(),filter=$('filter').value;
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
 const queue=pairs();
 $('stats').textContent=`${included.length} slides included · ${excluded} excluded · ${data.images.length}/${data.stats.input_images} captures processed`;
 $('errors').hidden=data.errors.length===0;
 $('errors').querySelector('summary').textContent=`${data.errors.length} images could not be read`;
 $('errors').querySelector('pre').textContent=data.errors.map(error=>error.name+': '+error.error).join('\n');
 renderWorkflow(queue.length,issues.length,included.length,excluded);
 if(activeView==='compare')renderComparison(queue);
 if(activeView==='issues')renderIssues(issues);
 if(activeView==='slides')renderDeck(included);
 else $('grid').replaceChildren();
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
$('pair-picker').onchange=e=>{pairIndex=Number(e.target.value);renderComparison(pairs())};
$('previous-pair').onclick=()=>{pairIndex--;renderComparison(pairs())};
$('next-pair').onclick=()=>{pairIndex++;renderComparison(pairs())};
$('keep-left').onclick=()=>decidePair('left');
$('keep-both').onclick=()=>decidePair('both');
$('keep-right').onclick=()=>decidePair('right');
$('undo-pair').onclick=()=>{
 if(!lastPairDecision)return;
 ({groups,resolved:resolvedPairs,index:pairIndex}=lastPairDecision);
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
function changeCapture(e){
 const d=e.target.dataset;
 if(d.select){const id=+d.select;e.target.checked?selected.add(id):selected.delete(id);e.target.closest('.card').classList.toggle('selected',e.target.checked);return}
 if(d.rep){const g=groups.find(g=>g.id===+d.group);g.representative=+d.rep;g.acknowledged=false}
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
 if(d.split){const id=+d.split,g=owner(id);g.members=g.members.filter(i=>i!==id);if(g.representative===id)g.representative=g.members[0];g.acknowledged=false;groups.push({id:Math.max(...groups.map(g=>g.id))+1,representative:id,members:[id],builds:0,needs_review:true,review_reasons:['check split capture']});lastPairDecision=null;lastIssueDecision=null;persist();render();notify('Capture split.')}
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
 target.members=chosen.flatMap(g=>g.members);target.builds=chosen.reduce((s,g)=>s+g.builds,0);
 target.representative=chosen.map(g=>images.get(g.representative)).sort((a,b)=>((b.ocr?.characters||b.detail||0)-(a.ocr?.characters||a.detail||0))||((b.slide_pixels||0)-(a.slide_pixels||0)))[0].id;
 target.excluded=chosen.every(g=>g.excluded);target.acknowledged=false;target.needs_review=true;target.review_reasons=['confirm most complete capture'];groups=groups.filter(g=>!selected.has(g.id)||g.id===target.id);selected.clear();lastPairDecision=null;lastIssueDecision=null;persist();render();notify('Groups merged. Confirm the most complete capture in Slide issues.');
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
 group.acknowledged=false;
 group.needs_review=true;
 group.review_reasons=['check adjusted crop'];
 issueIndex=groups.filter(hasIssue).indexOf(group);
 lastIssueDecision=null;
 lastPairDecision=null;
 closeCrop();
 persist();render();
 (activeView==='issues'?$('issue-preview'):$('continue-step')).focus();
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
