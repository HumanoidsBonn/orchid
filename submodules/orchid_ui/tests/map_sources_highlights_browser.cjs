// Synthetic map/status input only: no real ROS connection or goal dispatch.
const pw=require('/tmp/orchid-ui-tools/node_modules/playwright');
const fs=require('fs'),path=require('path'),assert=require('assert/strict');
const root=path.resolve(__dirname,'..'),out=path.join(root,'artifacts/map-feedback');
const base=process.env.ORCHID_UI_TEST_URL||'http://127.0.0.1:8000';
const pngHeight=fs.readFileSync(path.join(root,'static/map.png')).readUInt32BE(20);
(async()=>{
 fs.mkdirSync(out,{recursive:true});
 for(const engine of ['chromium','webkit']){
  const browser=await pw[engine].launch({headless:true});const page=await browser.newPage({viewport:{width:1194,height:834},hasTouch:true});
  const errors=[];page.on('pageerror',error=>errors.push(error.message));let pngLoads=0;
  await page.addInitScript(()=>{
   localStorage.setItem('scene-graph-file','/highlight-scene.json');localStorage.setItem('map-yaml-path','/test-map.yaml');
   window.mapLabels={};window.haloCount=0;window.hatchCount=0;
   const text=CanvasRenderingContext2D.prototype.fillText;
   CanvasRenderingContext2D.prototype.fillText=function(label,x,y,...rest){if(['mug','desk'].includes(label))mapLabels[label]={x,y};return text.call(this,label,x,y,...rest);};
   const radial=CanvasRenderingContext2D.prototype.createRadialGradient;
   CanvasRenderingContext2D.prototype.createRadialGradient=function(...args){if(args[5]===28)haloCount++;return radial.apply(this,args);};
   const clip=CanvasRenderingContext2D.prototype.clip;
   CanvasRenderingContext2D.prototype.clip=function(...args){if(this.canvas.id==='map-canvas')hatchCount++;return clip.apply(this,args);};
   const clear=CanvasRenderingContext2D.prototype.clearRect;
   CanvasRenderingContext2D.prototype.clearRect=function(...args){if(this.canvas.id==='map-canvas'){haloCount=0;hatchCount=0;}return clear.apply(this,args);};
  });
  await page.route('**/vendor/roslib.js',r=>r.fulfill({contentType:'text/javascript',body:fs.readFileSync(path.join(__dirname,'roslib_mock.js'),'utf8')}));
  await page.route('**/api/followup',r=>r.fulfill({contentType:'application/json',body:'{"prompt":null,"ok":true}'}));
  await page.route('**/map.png',r=>{pngLoads++;return r.fulfill({contentType:'image/png',body:fs.readFileSync(path.join(root,'static/map.png'))});});
  await page.route('**/test-map.yaml',r=>r.fulfill({body:'resolution: 0.05\norigin: [10, 20, 1.5707963267948966]\nimage: map.png\n'}));
  await page.route('**/missing-map.yaml',r=>r.fulfill({status:404,body:'missing'}));
  await page.route('**/slow-map.yaml',async r=>{await new Promise(resolve=>setTimeout(resolve,500));try{await r.fulfill({body:'resolution: 1\norigin: [999,999,0]\n'});}catch{}});
  await page.route('**/highlight-scene.json',r=>r.fulfill({contentType:'application/json',body:JSON.stringify({metadata:{relationship_config:'highlight-rules.yaml'},objects:[{id:1,label:'mug',center:[-4.5,41.7,.8],size:[.4,.4,.8]},{id:2,label:'desk',center:[5,30,.8],size:[2,1,.8]}]})}));
  await page.route('**/highlight-rules.yaml',r=>r.fulfill({body:'static_furniture: [desk]\nnull_classes: []\n'}));
  await page.goto(base);await page.waitForFunction(()=>window.uiMock&&mapLabels.mug);
  assert.equal(await page.locator('#map-source').inputValue(),'static');
  assert.equal(await page.evaluate(()=>uiMock.sockets.length),0); // Map is usable before ROS connection.
  const alignment=await page.evaluate(pngHeight=>{
   const scale=Number(document.querySelector('#map-scale').value),x=Number(document.querySelector('#map-origin-x').value),y=Number(document.querySelector('#map-origin-y').value);
   return {actual:mapLabels.mug,expected:{x:x+430*.05*scale+3,y:y+(pngHeight-1-294)*.05*scale-3}};
  },pngHeight);
  assert(Math.abs(alignment.actual.x-alignment.expected.x)<.01,JSON.stringify(alignment));assert(Math.abs(alignment.actual.y-alignment.expected.y)<.01,JSON.stringify(alignment));
  const moduleChecks=await page.evaluate(async()=>{const m=await import('/map_source.js');let failed=false;try{m.parseMapMetadata('resolution: -1\norigin: [0,0,0]');}catch{failed=true;}return{failed,yaw:m.parseMapMetadata('resolution: 1\norigin: [1,2,1.5707963267948966]').origin.orientation.z};});
  assert(moduleChecks.failed);assert(Math.abs(moduleChecks.yaw-Math.SQRT1_2)<1e-9);
  await page.click('#btn-connect');await page.waitForTimeout(60);
  await page.evaluate(()=>{const beat=()=>uiMock.heartbeat({session_id:window.testSession||'maps',ready:true,busy:false});beat();setInterval(beat,1000);});
  await page.waitForFunction(()=>[...uiMock.topics].some(t=>t.name==='/hsrb_bot/gui_state'));
  assert.equal(await page.evaluate(()=>[...uiMock.topics].filter(t=>t.name==='/map').length),0);
  const highlights={navigation_target:{id:'desk_2',boxer_id:2},known_objects:[{id:'mug_1',boxer_id:1}]};
  const status=async(state)=>page.evaluate(state=>uiMock.emitTopic('/hsrb_bot/gui_state',JSON.stringify(state)),state);
  await page.evaluate(()=>{haloCount=0;hatchCount=0;});
  await status({session_id:'maps',scene_highlights:highlights});await page.waitForFunction(()=>haloCount>=2&&hatchCount>=2);
  assert((await page.evaluate(()=>haloCount))>=2);assert((await page.evaluate(()=>hatchCount))>=2);
  await page.screenshot({path:path.join(out,`${engine}-static-highlights.png`)});
  // Disable furniture nodes and movable boxes: one halo and one hatched box remain.
  await page.evaluate(()=>{for(const id of ['scene-show-furniture-nodes','scene-show-movable-boxes']){const el=document.getElementById(id);el.checked=false;el.dispatchEvent(new Event('change'));}haloCount=0;hatchCount=0;});
  await status({session_id:'maps',scene_highlights:highlights});await page.waitForFunction(()=>haloCount===1&&hatchCount===1);
  assert.equal(await page.evaluate(()=>haloCount),1);assert.equal(await page.evaluate(()=>hatchCount),1);
  await page.evaluate(()=>{window.testSession='new';uiMock.heartbeat({session_id:'new',ready:true,busy:false});});await page.waitForTimeout(250);
  await page.evaluate(()=>{haloCount=0;hatchCount=0;});await status({session_id:'maps',scene_highlights:highlights});await page.waitForTimeout(250);
  assert.equal(await page.evaluate(()=>haloCount),0);assert.equal(await page.evaluate(()=>hatchCount),0);
  const apply=async(values)=>page.evaluate(values=>{for(const [id,value]of Object.entries(values))document.getElementById(id).value=value;document.getElementById('btn-save-map-settings').click();},values);
  await apply({'map-source':'topic'});await page.waitForFunction(()=>[...uiMock.topics].some(t=>t.name==='/map'));
  await page.evaluate(()=>[...uiMock.topics].find(t=>t.name==='/map').callback({info:{width:3,height:3,resolution:.5,origin:{position:{x:0,y:0},orientation:{w:1}}},data:Array(9).fill(0)}));
  await page.waitForTimeout(250);assert.equal(await page.evaluate(()=>[...uiMock.topics].filter(t=>t.name==='/map').length),0);
  await apply({'map-source':'static','map-yaml-path':'/missing-map.yaml'});await page.waitForFunction(()=>document.querySelector('#map-load-state').textContent.includes('404'));
  assert.equal(await page.evaluate(()=>[...uiMock.topics].filter(t=>t.name==='/map').length),0);
  await apply({'map-yaml-path':'/test-map.yaml'});await page.waitForFunction(()=>document.querySelector('#map-load-state').classList.contains('hidden'));
  const before=pngLoads;await page.click('#btn-refresh-map');await page.waitForFunction(()=>document.querySelector('#map-load-state').classList.contains('hidden'));assert(pngLoads>before);
  await apply({'map-yaml-path':'/slow-map.yaml'});
  await apply({'map-source':'topic'});
  await page.waitForFunction(()=>[...uiMock.topics].some(t=>t.name==='/map'));
  await page.waitForTimeout(650);
  assert(await page.locator('#map-load-state').isHidden());
  assert.equal(await page.locator('#map-source').inputValue(),'topic');
  await apply({'map-source':'static','map-yaml-path':'/test-map.yaml'});
  await page.waitForFunction(()=>document.querySelector('#map-load-state').classList.contains('hidden'));
  await page.reload();await page.waitForFunction(()=>window.uiMock&&mapLabels.mug);assert.equal(await page.locator('#map-source').inputValue(),'static');
  assert.equal(await page.evaluate(()=>uiMock.goals.length),0);assert.deepEqual(errors,[]);
  await browser.close();console.log(engine+': static/topic source, world alignment, refresh/error and role-specific highlights passed');
 }
})().catch(error=>{console.error(error);process.exit(1);});
