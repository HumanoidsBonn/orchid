// Short, mocked ROS checks. No socket or robot command is created.
const pw=require('/tmp/orchid-ui-tools/node_modules/playwright');
const fs=require('fs'),path=require('path'),assert=require('assert/strict');
const out=path.resolve(__dirname,'../artifacts/perception-previews');
const base=process.env.ORCHID_UI_TEST_URL||'http://127.0.0.1:8000';
(async()=>{
 fs.mkdirSync(out,{recursive:true});
 for(const engine of ['chromium','webkit']){
  const browser=await pw[engine].launch({headless:true});
  const page=await browser.newPage({viewport:{width:1194,height:834},hasTouch:true});
  const errors=[];page.on('pageerror',error=>errors.push(error.message));
  await page.addInitScript(()=>{
   window.labelDraws=[];const original=CanvasRenderingContext2D.prototype.fillText;
   CanvasRenderingContext2D.prototype.fillText=function(text,x,y,...rest){
    if(['mug','desk','wall'].includes(text))window.labelDraws.push({text,x,y,baseline:this.textBaseline});
    return original.call(this,text,x,y,...rest);
   };
  });
  await page.route('**/vendor/roslib.js',r=>r.fulfill({contentType:'text/javascript',body:fs.readFileSync(path.join(__dirname,'roslib_mock.js'),'utf8')}));
  await page.route('**/api/followup',r=>r.fulfill({contentType:'application/json',body:'{"prompt":null,"ok":true}'}));
  await page.route('**/preview-scene.json',r=>r.fulfill({contentType:'application/json',body:JSON.stringify({metadata:{relationship_config:'preview-rules.yaml'},objects:[{id:1,label:'mug',center:[0,0,.8],size:[.005,.005,.1]},{id:2,label:'desk',center:[1,0,.8],size:[.08,.08,.1]},{id:3,label:'wall',center:[-1,0,.8],size:[.005,.005,.1]}]})}));
  await page.route('**/preview-rules.yaml',r=>r.fulfill({body:'static_furniture: [desk]\nnull_classes: [wall]\n'}));
  await page.addInitScript(()=>{localStorage.setItem('scene-graph-file','/preview-scene.json');localStorage.setItem('camera-message-type','sensor_msgs/msg/CompressedImage');});
  await page.addInitScript(()=>localStorage.setItem('map-source','topic'));
  await page.goto(base);await page.waitForFunction(()=>window.uiMock);
  await page.click('#btn-connect');await page.waitForTimeout(50);
  await page.evaluate(()=>{const beat=()=>uiMock.heartbeat({session_id:'preview',ready:true,busy:false});beat();setInterval(beat,1000);});
  await page.waitForFunction(()=>[...uiMock.topics].filter(t=>t.name.includes('/visualization/')).length===2);
  const setting=async(id,value)=>page.evaluate(({id,value})=>{const input=document.getElementById(id);input.value=String(value);input.dispatchEvent(new Event('input',{bubbles:true}));},{id,value});
  await setting('preview-duration',1);
  const sendImage=async(topic,color)=>page.evaluate(({topic,color})=>{const canvas=document.createElement('canvas');canvas.width=640;canvas.height=360;const ctx=canvas.getContext('2d');ctx.fillStyle=color;ctx.fillRect(0,0,640,360);uiMock.emitTopic(topic,canvas.toDataURL('image/jpeg',.7).split(',')[1]);},{topic,color});
  await sendImage('/orchid_ui/camera/compressed','#114477');
  await page.waitForSelector('#widget-cam.has-image');
  assert((await page.locator('#widget-cam').boundingBox()).width>280);
  await sendImage('/hsrb_bot/visualization/segmentation/compressed','#11aa33');
  await page.waitForSelector('#widget-cam[data-preview="segmentation"]');
  await sendImage('/orchid_ui/camera/compressed','#773311');
  const oldSegmentation=await page.locator('#cam-image').getAttribute('src');
  await sendImage('/hsrb_bot/visualization/segmentation/compressed','#22bb44');
  await page.waitForFunction(old=>document.querySelector('#cam-image').src!==old,oldSegmentation);
  assert.equal(await page.locator('#widget-cam').getAttribute('data-preview'),'segmentation');
  const segmentationShownAt=Date.now();
  await sendImage('/hsrb_bot/visualization/grasp/compressed','#9933cc');
  assert.equal(await page.locator('#widget-cam').getAttribute('data-preview'),'segmentation');
  await page.waitForSelector('#widget-cam[data-preview="grasp"]');
  assert(Date.now()-segmentationShownAt>=850);
  await page.waitForFunction(()=>!document.querySelector('#widget-cam').dataset.preview);
  await page.waitForTimeout(600);
  const pixel=await page.evaluate(()=>{const c=document.createElement('canvas');c.width=c.height=1;c.getContext('2d').drawImage(document.querySelector('#cam-image'),0,0,1,1);return [...c.getContext('2d').getImageData(0,0,1,1).data];});
  assert(pixel[0]>pixel[1]&&pixel[1]>pixel[2]); // Newest live frame, not the preview.
  await page.evaluate(()=>{const el=document.querySelector('#chk-cam');el.checked=false;el.dispatchEvent(new Event('change'));});
  assert.equal(await page.evaluate(()=>[...uiMock.topics].filter(t=>t.name.includes('/visualization/')||t.name==='/orchid_ui/camera/compressed').length),0);
  await page.evaluate(()=>{const el=document.querySelector('#chk-cam');el.checked=true;el.dispatchEvent(new Event('change'));});
  await page.click('#page-tree');
  assert.equal(await page.evaluate(()=>[...uiMock.topics].filter(t=>t.name.includes('/visualization/')).length),0);
  await page.click('#page-map');
  assert.equal(await page.locator('#widget-cam').getAttribute('data-preview'),null);
  await sendImage('/orchid_ui/camera/compressed','#114477');await page.waitForSelector('#widget-cam.has-image');
  await page.waitForFunction(()=>document.querySelector('#map-status').textContent.includes('3 objects'));
  await page.evaluate(()=>[...uiMock.topics].find(t=>t.name==='/map').callback({info:{width:5,height:5,resolution:1,origin:{position:{x:-2,y:-2},orientation:{w:1}}},data:Array(25).fill(0)}));
  await page.waitForTimeout(250);
  await page.evaluate(()=>{document.querySelector('#scene-show-movable-boxes').checked=false;document.querySelector('#scene-show-movable-boxes').dispatchEvent(new Event('change'));});
  const point=await page.evaluate(()=>{const c=document.querySelector('#map-canvas'),r=c.getBoundingClientRect(),s=Number(document.querySelector('#map-scale').value);return{x:r.left+Number(document.querySelector('#map-origin-x').value)+2*s+8,y:r.top+Number(document.querySelector('#map-origin-y').value)+2*s};});
  await page.mouse.click(point.x,point.y);await page.waitForSelector('#scene-info:not(.hidden)');
  assert((await page.locator('#scene-info-content').textContent()).includes('mug'));
  const toggle=async(id,value)=>page.evaluate(({id,value})=>{const el=document.getElementById(id);el.checked=value;el.dispatchEvent(new Event('change'));},{id,value});
  const clickObject=async(x,offset=8)=>{
   const p=await page.evaluate(({x,offset})=>{const c=document.querySelector('#map-canvas'),r=c.getBoundingClientRect(),s=Number(document.querySelector('#map-scale').value);return{x:r.left+Number(document.querySelector('#map-origin-x').value)+(x+2)*s+offset,y:r.top+Number(document.querySelector('#map-origin-y').value)+2*s};},{x,offset});
   await page.mouse.click(p.x,p.y);
  };
  await toggle('scene-show-furniture-nodes',false);
  await page.waitForTimeout(250);
  const labels=await page.evaluate(()=>Object.fromEntries(labelDraws.map(row=>[row.text,row.baseline])));
  assert.equal(labels.desk,'bottom');assert.equal(labels.mug,'middle');
  await clickObject(1);assert(await page.locator('#scene-info').isHidden());
  await clickObject(1,0);assert((await page.locator('#scene-info-content').textContent()).includes('desk'));
  await clickObject(0);assert((await page.locator('#scene-info-content').textContent()).includes('mug'));
  await toggle('scene-show-movable-nodes',false);await clickObject(0);assert(await page.locator('#scene-info').isHidden());
  await toggle('scene-show-structural-nodes',true);await clickObject(-1);assert((await page.locator('#scene-info-content').textContent()).includes('wall'));
  await toggle('scene-show-movable-nodes',true);await clickObject(0);
  assert.equal(await page.evaluate(()=>localStorage.getItem('scene-show-furnitureNodes')),'false');
  await setting('scene-outline-width',4);await setting('scene-node-radius',8);await setting('camera-window-scale',1.25);
  assert.equal(await page.evaluate(()=>localStorage.getItem('scene-outline-width')),'4');
  assert.equal(await page.evaluate(()=>localStorage.getItem('scene-node-radius')),'8');
  for(const [width,height]of [[1194,834],[390,844]]){
   await page.setViewportSize({width,height});await page.waitForTimeout(250);
   const pip=await page.locator('#widget-cam').boundingBox();assert(pip.x>=0&&pip.x+pip.width<=width+.5);
   assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
   await page.screenshot({path:path.join(out,`${engine}-${width}.png`)});
  }
  assert.deepEqual(errors,[]);assert.equal(await page.evaluate(()=>uiMock.goals.length),0);
  console.log(engine+': preview timing/live restoration, subscription gating, node selection and tablet/phone layout passed');
  await browser.close();
 }
})().catch(e=>{console.error(e);process.exit(1);});
