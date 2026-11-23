// Short transport regression with mocked ROS: never sends robot commands.
let pw; try {pw=require('playwright');} catch {pw=require('/tmp/orchid-ui-tools/node_modules/playwright');}
const fs=require('fs'),path=require('path'),assert=require('assert/strict');
(async()=>{
 for(const name of ['chromium','webkit']) {
  const browser=await pw[name].launch({headless:true});
  const page=await browser.newPage({viewport:{width:390,height:844},hasTouch:true}); const errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  await page.route('**/vendor/roslib.js',r=>r.fulfill({contentType:'text/javascript',body:fs.readFileSync(path.join(__dirname,'roslib_mock.js'),'utf8')}));
  await page.route('**/api/followup',r=>r.fulfill({contentType:'application/json',body:'{"prompt":null}'}));
  await page.goto('http://127.0.0.1:8000/#map'); await page.waitForFunction(()=>window.uiMock);
  await page.click('#btn-connect');await page.waitForTimeout(40);
  await page.evaluate(()=>{const beat=()=>uiMock.heartbeat({session_id:'transport',ready:true,busy:false});beat();setInterval(beat,1000);});
  const count=topic=>page.evaluate(topic=>[...uiMock.topics].filter(t=>t.name===topic).length,topic);
  assert.equal(await count('/map'),1);
  await page.evaluate(()=>{window.grid={header:{frame_id:'map'},info:{width:2,height:2,resolution:.05,origin:{position:{x:0,y:0},orientation:{w:1,z:0}}},data:[0,100,-1,0]}; [...uiMock.topics].find(t=>t.name==='/map').callback(grid);});
  await page.waitForTimeout(300);assert.equal(await count('/map'),0);
  await page.click('#page-tree'); await page.click('#page-map');assert.equal(await count('/map'),0);
  await page.click('#btn-refresh-map');assert.equal(await count('/map'),1);
  await page.evaluate(()=>[...uiMock.topics].find(t=>t.name==='/map').callback(grid));await page.waitForTimeout(300);assert.equal(await count('/map'),0);
  // Browser subscription and already queued callbacks stop when camera is disabled.
  assert.equal(await count('/orchid_ui/camera/compressed'),1);
  await page.evaluate(()=>{window.oldCamera=[...uiMock.topics].find(t=>t.name==='/orchid_ui/camera/compressed');const box=document.querySelector('#chk-cam');box.checked=false;box.dispatchEvent(new Event('change'));});
  assert.equal(await count('/orchid_ui/camera/compressed'),0);
  await page.evaluate(()=>oldCamera.callback({data:'invalid stale frame'}));
  await page.waitForTimeout(250);assert(await page.locator('#cam-image').isHidden());
  await page.evaluate(()=>{const p=[...uiMock.topics].find(t=>t.name==='/global_pose');if(p.throttle_rate!==100)throw Error('pose not 10Hz');p.callback({header:{frame_id:'map'},pose:{position:{x:1.25,y:.4},orientation:{x:0,y:0,z:0,w:1}}});});
  await page.waitForTimeout(80);assert.equal(await page.locator('#pose-x').textContent(),'1.25');
  // Reconnect keeps same-map cache; changing identity invalidates it.
  await page.click('#btn-connect');await page.click('#btn-connect');await page.waitForTimeout(40);
  await page.evaluate(()=>uiMock.heartbeat({session_id:'transport',ready:true,busy:false}));assert.equal(await count('/map'),0);
  assert.equal(await count('/orchid_ui/camera/compressed'),0);
  assert.equal(await page.evaluate(()=>uiMock.goals.length),0);assert.deepEqual(errors,[]);
  console.log(`${name}: map load/refresh/cache, live pose, camera-off unsubscribe passed`);await browser.close();
 }
})().catch(e=>{console.error(e);process.exit(1);});
