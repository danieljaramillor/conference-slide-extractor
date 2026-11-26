const images=new Map(data.images.map(i=>[i.id,i]));
const server=typeof REVIEW_TOKEN!=='undefined';
const storageKey='slide-review-'+(data.manifest_id||data.source+'-'+data.images.length);
let groups=structuredClone(data.groups),selected=new Set(),crops={},cropId=null,dragStart=null;
let resolvedPairs=new Set(), pairIndex=0, activeView='compare', lastPairDecision=null;
let revision=Date.now();
const $=id=>document.getElementById(id),owner=id=>groups.find(g=>g.members.includes(id));
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const needs=g=>!g.excluded&&g.needs_review&&!g.acknowledged;
const payload=()=>({version:2,source:data.source,manifest_id:data.manifest_id,groups,crops,revision,resolved_pairs:[...resolvedPairs]});
function notify(s){$('status').textContent=s}
function persist(){
 revision=Math.max(Date.now(),revision+1);
 const body=JSON.stringify(payload());
 try{localStorage.setItem(storageKey,body)}catch{}
 if(server)fetch('decisions',{method:'POST',headers:{'Content-Type':'application/json','X-Review-Token':REVIEW_TOKEN},body})
  .then(response=>{if(!response.ok)throw Error('Save failed')})
  .catch(()=>notify('Could not save to disk. Use Save decisions in All slides to keep a copy.'));
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
 if(view==='slides')$('filter').value='all';
 if(view==='issues')$('filter').value='review';
 $('search').value='';
 render();
}
function compareSide(group,side){
 const image=images.get(group.representative);
 const src=server||image.native_preview?nativeURL(image.id):`thumbs/${image.id}.jpg`;
 return `<article class="compare-side">
  <button class="preview-button" data-zoom="${image.id}" aria-label="Enlarge ${side.toLowerCase()} slide"><img src="${src}" alt="${side} slide preview"></button>
  <div class="info"><h3>${side} · Group ${group.id}</h3>
  <div class="side-info">${group.members.length} captures · ${image.slide_width||'?'} × ${image.slide_height||'?'} px</div>
  <div class="file">${esc(image.name.split(/[\\/]/).pop())}</div></div></article>`;
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
 persist();render();
 notify(decision==='both'?'Both slides kept.':'Slides merged. Chosen capture kept.');
}
function restore(d){
 validate(d);
 groups=d.groups;crops=d.crops||{};
 resolvedPairs=new Set(d.resolved_pairs||[]);
 revision=Math.max(revision,d.revision||0);
 pairIndex=0;lastPairDecision=null;selected.clear();
}
function render(){
 const expanded=[...document.querySelectorAll('.inspect[open]')].map(el=>el.closest('.card').id);
 const term=$('search').value.toLowerCase(),filter=$('filter').value,review=groups.filter(hasIssue).length,excluded=groups.filter(g=>g.excluded).length;
 const queue=pairs();
 $('stats').textContent=`${groups.length-excluded} slides selected · ${excluded} excluded · ${data.images.length}/${data.stats.input_images} captures processed`+(data.errors.length?` · ${data.errors.length} unreadable`: '');
 $('errors').hidden=data.errors.length===0;
 $('errors').querySelector('summary').textContent=`${data.errors.length} images could not be read`;
 $('errors').querySelector('pre').textContent=data.errors.map(e=>e.name+': '+e.error).join('\n');
 $('compare-tab').textContent=`Compare pairs (${queue.length})`;
 $('issues-tab').textContent=`Slide issues (${review})`;
 for(const view of ['compare','issues','slides'])$(view+'-tab').setAttribute('aria-pressed',view===activeView);
 $('comparison-view').hidden=activeView!=='compare';
 $('slide-view').hidden=activeView==='compare';
 if(activeView==='compare'){
  $('grid').innerHTML='';
  renderComparison(queue);
  return;
 }
 const visible=groups.filter(g=>(!term||g.members.some(i=>images.get(i).name.toLowerCase().includes(term)))&&(filter==='all'||filter==='review'&&hasIssue(g)||filter==='ready'&&!g.excluded&&!needs(g)||filter==='excluded'&&g.excluded||filter==='multi'&&g.members.length>1||filter==='full'&&images.get(g.representative).crop_mode==='full'));
 $('grid').innerHTML=visible.map(g=>{const im=images.get(g.representative);return `<article class="card ${selected.has(g.id)?'selected':''}" id="group-${g.id}"><img loading="lazy" src="${crops[im.id]&&server?nativeURL(im.id):`thumbs/${im.id}.jpg`}" alt="Group ${g.id}" data-zoom="${im.id}"><div class="info"><h2><label><input type="checkbox" data-select="${g.id}" ${selected.has(g.id)?'checked':''}> Slide group ${g.id}</label></h2><div>${g.members.length} captures · ${im.slide_width||'?'} × ${im.slide_height||'?'} px</div>${g.excluded?'<span class="badge">Excluded</span>':needs(g)?(g.review_reasons||[]).map(s=>`<span class="badge">${esc(s)}</span>`).join(''):'<span class="badge ready">Ready</span>'}<div class="controls"><label><input type="checkbox" data-exclude="${g.id}" ${g.excluded?'checked':''}> Exclude</label><label><input type="checkbox" data-checked="${g.id}" ${g.acknowledged?'checked':''}> Checked</label><button data-crop="${im.id}" ${server?'':'disabled title="Use the local review server to edit crops"'}>Adjust crop</button></div><div class="file">${esc(im.name)}</div>${im.ocr?.status==='ok'?`<details><summary>Extracted text</summary><pre>${esc(im.ocr.text)}</pre></details>`:''}<details class="inspect"><summary>Inspect captures / change selected image</summary><div class="members">${g.members.map(id=>`<div class="member"><label><input type="radio" name="rep-${g.id}" data-rep="${id}" data-group="${g.id}" ${id===im.id?'checked':''}> Use this capture</label><img loading="lazy" src="thumbs/${id}.jpg" data-zoom="${id}" alt="Member preview"><div class="file">${esc(images.get(id).name)}</div>${g.members.length>1?`<button data-split="${id}">Split into a separate slide</button>`:''}</div>`).join('')}</div></details></div></article>`}).join('')||'<p>No slides in this view. Choose All groups to inspect every selection.</p>';
 for(const id of expanded){const el=document.getElementById(id)?.querySelector('.inspect');if(el)el.open=true}
}
$('compare-tab').onclick=()=>changeView('compare');
$('issues-tab').onclick=$('open-issues').onclick=()=>changeView('issues');
$('slides-tab').onclick=()=>changeView('slides');
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
$('search').oninput=render;$('filter').onchange=render;
$('grid').onchange=e=>{
 const d=e.target.dataset;
 if(d.select){const id=+d.select;e.target.checked?selected.add(id):selected.delete(id);e.target.closest('.card').classList.toggle('selected',e.target.checked);return}
 if(d.rep){const g=groups.find(g=>g.id===+d.group);g.representative=+d.rep;g.acknowledged=false}
 if(d.exclude)groups.find(g=>g.id===+d.exclude).excluded=e.target.checked;
 if(d.checked)groups.find(g=>g.id===+d.checked).acknowledged=e.target.checked;
 lastPairDecision=null;
 persist();render();notify('Changes saved in this browser.');
};
document.addEventListener('click',e=>{
 const d=e.target.dataset;
 if(d.zoom){const id=+d.zoom;$('lightbox').style.display='flex';$('lightbox').querySelector('img').src=nativeURL(id);$('caption').textContent=images.get(id).name+(server||images.get(id).native_preview?' · native-resolution preview':' · thumbnail; start the local server for the native image')}
 if(d.split){const id=+d.split,g=owner(id);g.members=g.members.filter(i=>i!==id);if(g.representative===id)g.representative=g.members[0];g.acknowledged=false;groups.push({id:Math.max(...groups.map(g=>g.id))+1,representative:id,members:[id],builds:0,needs_review:true,review_reasons:['check split capture']});lastPairDecision=null;persist();render();notify('Capture split.')}
 if(d.crop&&server){cropId=+d.crop;$('cropdialog').style.display='flex';$('cropsource').src=`source/${cropId}.png`;cropFields(crops[cropId]||images.get(cropId).crop)}
});
$('merge').onclick=()=>{
 const chosen=groups.filter(g=>selected.has(g.id));if(chosen.length<2)return notify('Select at least two groups.');const target=chosen[0];
 target.members=chosen.flatMap(g=>g.members);target.builds=chosen.reduce((s,g)=>s+g.builds,0);
 target.representative=chosen.map(g=>images.get(g.representative)).sort((a,b)=>((b.ocr?.characters||b.detail||0)-(a.ocr?.characters||a.detail||0))||((b.slide_pixels||0)-(a.slide_pixels||0)))[0].id;
 target.excluded=chosen.every(g=>g.excluded);target.acknowledged=false;target.needs_review=true;target.review_reasons=['confirm most complete capture'];groups=groups.filter(g=>!selected.has(g.id)||g.id===target.id);selected.clear();lastPairDecision=null;persist();render();notify('Groups merged. Confirm the most complete capture and mark checked.');
};
$('ack').onclick=()=>{for(const g of groups)if(selected.has(g.id))g.acknowledged=true;selected.clear();lastPairDecision=null;persist();render();notify('Marked checked.')};
$('save').onclick=()=>{const url=URL.createObjectURL(new Blob([JSON.stringify(payload(),null,2)],{type:'application/json'})),a=document.createElement('a');a.href=url;a.download='review-decisions.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);notify('Decisions downloaded.')};
$('load').onclick=()=>$('loadfile').click();$('loadfile').onchange=async e=>{try{restore(JSON.parse(await e.target.files[0].text()));persist();render();notify('Decisions loaded')}catch(err){notify(err.message)}};
$('export').hidden=!server;$('export').onclick=async()=>{ $('export').disabled=true;notify('Exporting native-resolution slides and text…');try{const r=await fetch('export',{method:'POST',headers:{'Content-Type':'application/json','X-Review-Token':REVIEW_TOKEN},body:JSON.stringify(payload())}),d=await r.json();if(!r.ok)throw Error(d.error);$('download').href=d.download;$('download').download=d.download;$('download').hidden=false;notify(`Exported ${d.count} slides to ${d.directory}. Download the ZIP above.`)}catch(err){notify(err.message)}finally{$('export').disabled=false}};
function closeZoom(){$('lightbox').style.display='none'}$('lightbox').querySelector('button').onclick=closeZoom;
function drawCrop(b){Object.assign($('cropselection').style,{left:b[0]*100+'%',top:b[1]*100+'%',width:(b[2]-b[0])*100+'%',height:(b[3]-b[1])*100+'%'})}
function cropValues(){return ['cx1','cy1','cx2','cy2'].map(id=>+$(id).value/100)}
function cropFields(b){['cx1','cy1','cx2','cy2'].forEach((id,j)=>$(id).value=(b[j]*100).toFixed(2));drawCrop(b)}
for(const id of ['cx1','cy1','cx2','cy2'])$(id).oninput=()=>drawCrop(cropValues());
function pointer(e){const r=$('cropsource').getBoundingClientRect();return [Math.max(0,Math.min(1,(e.clientX-r.left)/r.width)),Math.max(0,Math.min(1,(e.clientY-r.top)/r.height))]}
$('cropbox').onpointerdown=e=>{e.preventDefault();dragStart=pointer(e);$('cropbox').setPointerCapture(e.pointerId)};
$('cropbox').onpointermove=e=>{if(dragStart){const p=pointer(e);cropFields([Math.min(p[0],dragStart[0]),Math.min(p[1],dragStart[1]),Math.max(p[0],dragStart[0]),Math.max(p[1],dragStart[1])])}};
$('cropbox').onpointerup=()=>dragStart=null;$('cropbox').onpointercancel=()=>dragStart=null;
$('cancelcrop').onclick=()=>$('cropdialog').style.display='none';
$('applycrop').onclick=()=>{const b=cropValues();try{validate({...payload(),crops:{...crops,[cropId]:b}});crops[cropId]=b;const g=owner(cropId);g.acknowledged=false;g.needs_review=true;g.review_reasons=['check adjusted crop'];$('cropdialog').style.display='none';persist();render();notify('Crop adjusted. Text will be refreshed on export.')}catch(err){notify(err.message)}};
document.addEventListener('keydown',e=>{if(e.key==='Escape'){closeZoom();$('cropdialog').style.display='none'}});
let localDecision=null;
try{localDecision=JSON.parse(localStorage.getItem(storageKey))}catch{}
const savedDecision=typeof SAVED_REVIEW!=='undefined'?SAVED_REVIEW:null;
for(const candidate of [localDecision,savedDecision].filter(Boolean).sort((a,b)=>(b.revision||0)-(a.revision||0))){
 try{restore(candidate);break}catch{}
}
if(pairs().length)changeView('compare');
else if(groups.some(hasIssue))changeView('issues');
else changeView('slides');
