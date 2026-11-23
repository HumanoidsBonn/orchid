// Static-image / published-map registration parity with the current demo map.
const pw=require('/tmp/orchid-ui-tools/node_modules/playwright');
const fs=require('fs'),path=require('path'),assert=require('assert/strict');
const root=path.resolve(__dirname,'..'),base=process.env.ORCHID_UI_TEST_URL||'http://127.0.0.1:8000';
const png=fs.readFileSync(path.join(root,'static/map.png')),width=png.readUInt32BE(16),height=png.readUInt32BE(20);
(async()=>{
 for(const engine of ['chromium','webkit']){
  const browser=await pw[engine].launch({headless:true});const page=await browser.newPage({viewport:{width:1194,height:834}});
  await page.addInitScript(()=>{
   window.sceneLabels={};const original=CanvasRenderingContext2D.prototype.fillText;
   CanvasRenderingContext2D.prototype.fillText=function(label,x,y,...rest){if(label==='desk'||label==='mug')sceneLabels[label]={x,y};return original.call(this,label,x,y,...rest);};
   localStorage.setItem('scene-graph-file','/parity-scene.json');
  });
  await page.route('**/vendor/roslib.js',r=>r.fulfill({contentType:'text/javascript',body:fs.readFileSync(path.join(__dirname,'roslib_mock.js'),'utf8')}));
  await page.route('**/api/followup',r=>r.fulfill({contentType:'application/json',body:'{"prompt":null,"ok":true}'}));
  await page.route('**/parity-scene.json',r=>r.fulfill({contentType:'application/json',body:JSON.stringify({metadata:{relationship_config:'parity-rules.yaml'},objects:[{id:41,label:'desk',center:[-8.,-3.4,.7],size:[2,1,.8]},{id:42,label:'mug',center:[-7.8,-3.3,.8],size:[.08,.08,.1]}]})}));
  await page.route('**/parity-rules.yaml',r=>r.fulfill({body:'static_furniture: [desk]\nnull_classes: []\n'}));
  await page.goto(base);await page.waitForFunction(()=>sceneLabels.desk&&sceneLabels.mug);
  const staticPositions=await page.evaluate(()=>({...sceneLabels}));
  await page.click('#btn-connect');await page.waitForTimeout(60);
  await page.evaluate(()=>{const beat=()=>uiMock.heartbeat({session_id:'parity',ready:true,busy:false});beat();setInterval(beat,1000);});
  await page.evaluate(()=>{document.querySelector('#map-source').value='topic';document.querySelector('#btn-save-map-settings').click();sceneLabels={};});
  await page.waitForFunction(()=>[...uiMock.topics].some(t=>t.name==='/map'));
  await page.evaluate(({width,height})=>[...uiMock.topics].find(t=>t.name==='/map').callback({info:{width,height,resolution:0.05000000074505806,origin:{position:{x:-17.2,y:-22.6},orientation:{w:1}}},data:Array(width*height).fill(0)}),{width,height});
  await page.waitForFunction(()=>sceneLabels.desk&&sceneLabels.mug);
  const topicPositions=await page.evaluate(()=>({...sceneLabels}));
  for(const label of ['desk','mug'])for(const axis of ['x','y'])assert(Math.abs(staticPositions[label][axis]-topicPositions[label][axis])<.1,`${label}.${axis}: ${JSON.stringify({staticPositions,topicPositions})}`);
  assert.equal(await page.evaluate(()=>uiMock.goals.length),0);
  console.log(engine+': current PNG and topic scene positions match within 0.1 CSS px');await browser.close();
 }
})().catch(error=>{console.error(error);process.exit(1);});
