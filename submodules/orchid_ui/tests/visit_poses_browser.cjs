// Small offline smoke check. No ROS connection or robot commands.
let pw;try{pw=require('playwright');}catch{pw=require('/tmp/orchid-ui-tools/node_modules/playwright');}
const assert=require('assert/strict'),fs=require('fs'),path=require('path');
const base=process.env.TEST_BASE_URL||'http://127.0.0.1:8000';
const pose=(valid,x=0,y=0,yaw=0)=>({valid,frame_id:'map',position:{x,y,z:0},orientation:{x:0,y:0,z:Math.sin(yaw/2),w:Math.cos(yaw/2)}});
const graph={metadata:{world_frame:'map',relationship_config:'visit-rules.yaml'},objects:[
 {id:41,label:'desk',center:[0,0,.4],size:[1,1,.8],base_access_poses:[pose(true,1,2,Math.PI/2),pose(false),pose(undefined)]},
 {id:42,label:'desk',center:[2,0,.4],size:[1,1,.8],base_access_poses:[pose(true,-1,-2,-Math.PI/2),pose(true,NaN)]},
]};
(async()=>{
 for(const engine of ['chromium','webkit']){
  const browser=await pw[engine].launch({headless:true});
  const page=await browser.newPage({viewport:{width:1194,height:834}}),errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  await page.route('**/vendor/roslib.js',r=>r.fulfill({contentType:'text/javascript',body:fs.readFileSync(path.join(__dirname,'roslib_mock.js'),'utf8')}));
  await page.route('**/scene_graph_complete.json',r=>r.fulfill({contentType:'application/json',body:JSON.stringify(graph)}));
  await page.route('**/visit-rules.yaml',r=>r.fulfill({body:'static_furniture: [desk]\nnull_classes: []\n'}));
  await page.addInitScript(()=>{
   window.visitArrows=0;const arc=CanvasRenderingContext2D.prototype.arc;
   CanvasRenderingContext2D.prototype.arc=function(x,y,r,...args){if(r===4)window.visitArrows++;return arc.call(this,x,y,r,...args);};
  });
  await page.goto(base);await page.waitForFunction(()=>document.querySelector('#scene-visit-pose-count')?.textContent==='2');
  const resolved=await page.evaluate(async graph=>{
   const {validVisitPoses}=await import('/scene_data.js?v=20261103a');
   return graph.objects.map(o=>validVisitPoses(o));
  },graph);
  assert.equal(resolved[0].length,1);assert.equal(resolved[1].length,1);
  assert(Math.abs(resolved[0][0].yaw-Math.PI/2)<1e-9);
  assert(Math.abs(resolved[1][0].yaw+Math.PI/2)<1e-9);
  assert.equal(await page.locator('#scene-show-visit-poses').isChecked(),false);
  await page.evaluate(()=>{document.querySelector('#scene-show-visit-poses').checked=true;document.querySelector('#scene-show-visit-poses').dispatchEvent(new Event('change'));});
  await page.waitForFunction(()=>window.visitArrows>=2);
  assert.equal(await page.evaluate(()=>localStorage.getItem('scene-show-visitPoses')),'true');
  await page.reload();await page.waitForFunction(()=>document.querySelector('#scene-show-visit-poses').checked);
  assert.deepEqual(errors,[]);
  await browser.close();console.log(`${engine}: valid-only counts, quaternion headings, pose arrows and persisted toggle passed`);
 }
})().catch(e=>{console.error(e);process.exit(1);});
