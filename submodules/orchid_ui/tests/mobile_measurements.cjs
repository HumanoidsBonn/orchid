// Controlled before/after indicators and Chromium trusted touch input. No ROS connection.
let pw;try{pw=require('playwright');}catch{pw=require('/tmp/orchid-ui-tools/node_modules/playwright');}
const fs=require('fs'),path=require('path'),assert=require('assert/strict');
const root=path.resolve(__dirname,'..'),base=process.env.ORCHID_UI_TEST_URL||'http://127.0.0.1:18000';
const mock=fs.readFileSync(path.join(__dirname,'roslib_mock.js'),'utf8');
(async()=>{
const browser=await pw.chromium.launch({headless:true}),rows=[];
for(const before of [true,false]){
 const page=await browser.newPage({viewport:{width:1194,height:834},hasTouch:true});
 await page.route('**/vendor/roslib.js',r=>r.fulfill({contentType:'text/javascript',body:mock}));
 await page.route('**/api/followup',r=>r.fulfill({contentType:'application/json',body:'{"prompt":null}'}));
 if(before)await page.route('**/*',r=>{const file=new URL(r.request().url()).pathname.split('/').at(-1)||'index.html';const type=file.endsWith('.js')?'text/javascript':file.endsWith('.css')?'text/css':'text/html';if(['app.js','style.css','index.html'].includes(file))return r.fulfill({contentType:type,body:fs.readFileSync(path.join(__dirname,'.baseline','pages-'+file))});return r.fallback();});
 const started=Date.now();await page.goto(base+'/index.html');await page.waitForFunction(()=>window.uiMock);const loaded_ms=Date.now()-started;
 await page.click('#btn-connect');await page.waitForTimeout(100);
 await page.evaluate(()=>uiMock.heartbeat({session_id:'measure',ready:true,busy:false,task_description:''}));
 rows.push({version:before?'before':'after',loaded_ms,...await page.evaluate(()=>({dom:document.querySelectorAll('*').length,subscriptions:uiMock.topics.size,heap:performance.memory?.usedJSHeapSize||null,navigation_ms:performance.getEntriesByType('navigation')[0].domContentLoadedEventEnd}))});
 if(!before){
  const session=await page.context().newCDPSession(page),scale=Number(await page.inputValue('#map-scale'));
  const touch=(x,y,id)=>({x,y,id,radiusX:1,radiusY:1});
  await session.send('Input.dispatchTouchEvent',{type:'touchStart',touchPoints:[touch(400,400,1)]});
  await session.send('Input.dispatchTouchEvent',{type:'touchStart',touchPoints:[touch(400,400,1),touch(600,400,2)]});
  await session.send('Input.dispatchTouchEvent',{type:'touchMove',touchPoints:[touch(350,400,1),touch(650,400,2)]});
  await session.send('Input.dispatchTouchEvent',{type:'touchEnd',touchPoints:[touch(350,400,1)]});
  await session.send('Input.dispatchTouchEvent',{type:'touchMove',touchPoints:[touch(380,430,1)]});
  await session.send('Input.dispatchTouchEvent',{type:'touchCancel',touchPoints:[]});
  await page.waitForTimeout(100);assert(Number(await page.inputValue('#map-scale'))>scale);assert.equal(await page.evaluate(()=>visualViewport.scale),1);
  rows.at(-1).trusted_touch='pinch, transition, pan, cancel passed';
 }
 await page.close();
}
// Actual vendored assets, no mock and no connection: external startup requests are forbidden.
const page=await browser.newPage(),requests=[],errors=[];page.on('request',r=>requests.push(r.url()));page.on('pageerror',e=>errors.push(e.message));await page.goto(base);await page.waitForTimeout(500);assert.deepEqual(errors,[]);assert(requests.every(u=>u.startsWith(base)));await browser.close();
fs.writeFileSync(path.join(root,'artifacts/mobile-pages/measurements.json'),JSON.stringify({rows,external_requests:0,notes:'Single controlled run; timing is observational, coarse heap reporting is not a precise memory measurement.'},null,2));console.log(JSON.stringify(rows));
})().catch(e=>{console.error(e);process.exit(1);});
