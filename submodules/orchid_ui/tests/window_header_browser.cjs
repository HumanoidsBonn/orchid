// Small layout/control smoke check only: no availability suite, timers, soak or ROS traffic.
let pw;try{pw=require('playwright');}catch{pw=require('/tmp/orchid-ui-tools/node_modules/playwright');}
const fs=require('fs'),path=require('path'),assert=require('assert/strict');
const root=path.resolve(__dirname,'..'),out=path.join(root,'artifacts/window-header');fs.mkdirSync(out,{recursive:true});
(async()=>{
 const browser=await pw.chromium.launch({headless:true}),page=await browser.newPage({viewport:{width:1194,height:834},hasTouch:true});const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.route('**/vendor/roslib.js',r=>r.fulfill({contentType:'text/javascript',body:fs.readFileSync(path.join(__dirname,'roslib_mock.js'),'utf8')}));
 await page.route('**/api/followup',r=>r.fulfill({contentType:'application/json',body:'{"prompt":null}'}));
 await page.goto(process.env.ORCHID_UI_TEST_URL||'http://127.0.0.1:8000');await page.waitForFunction(()=>window.uiMock);await page.click('#btn-connect');await page.waitForTimeout(40);await page.evaluate(()=>{const heartbeat=()=>uiMock.heartbeat({session_id:'layout',ready:true,busy:false,task_description:''});heartbeat();setInterval(heartbeat,1000);});
 const muted=await page.locator('#btn-toggle-speech').getAttribute('aria-pressed');await page.click('#btn-toggle-speech');assert.notEqual(await page.locator('#btn-toggle-speech').getAttribute('aria-pressed'),muted);assert(await page.locator('.ribbon-connection #btn-toggle-speech').isVisible());
 const rows=[];
 for(const theme of ['dark','light']){
  if(theme==='light')await page.click('#btn-theme-toggle');
  for(const [width,height]of [[1194,834],[390,844]]){
   await page.setViewportSize({width,height});
   for(const route of ['map','tree']){
    await page.click('#page-'+route);await page.waitForFunction(route=>document.body.dataset.page===route,route);await page.waitForTimeout(250);
    const frame=await page.locator('#widget-'+route).boundingBox(),workspace=await page.locator('#workspace').boundingBox();assert(Math.abs(frame.x-workspace.x-(width<650?8:12))<1);assert(Math.abs(frame.y-workspace.y-(width<650?8:12))<1);
    const style=await page.locator('#widget-'+route).evaluate(el=>{const s=getComputedStyle(el);return{top:s.borderTopWidth,left:s.borderLeftWidth};});assert.equal(style.top,style.left);
    assert.equal(await page.locator('#page-'+route).evaluate(el=>getComputedStyle(el).boxShadow),'none');
    assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
    if(width<650){const grid=await page.locator('.top-ribbon').evaluate(el=>getComputedStyle(el).gridTemplateRows.split(' ').length);assert.equal(grid,2);rows.push(grid);const controls=await page.locator('.ribbon-connection').boundingBox(),status=await page.locator('.robot-summary').boundingBox();assert(controls.y+controls.height<=status.y+1);}
    else assert.equal(await page.locator('.top-ribbon').evaluate(el=>getComputedStyle(el).display),'flex');
    if(route==='tree'){
     await page.evaluate(()=>uiMock.emitTopic('/hsrb_bot/behavior_tree',JSON.stringify({name:'Root',node_type:'Sequence',status:'RUNNING',children:[{name:'Leaf',node_type:'BtActionLeaf',status:'RUNNING'}]})));await page.waitForSelector('.tree-node');await page.locator('[data-node-id="0.0"]').click();assert(await page.locator('#tree-details').isVisible());const close=await page.locator('#close-tree-details').boundingBox();assert(close.width<=28&&close.height<=28);assert.equal(await page.locator('#close-tree-details').textContent(),'×');await page.click('#close-tree-details');assert(!(await page.locator('#tree-details').isVisible()));
    }
    await page.screenshot({path:path.join(out,`${theme}-${route}-${width}.png`)});
   }
  }
 }
 await page.setViewportSize({width:320,height:700});assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));assert.equal(await page.locator('.top-ribbon').evaluate(el=>getComputedStyle(el).gridTemplateRows.split(' ').length),2);for(const id of ['btn-connect','btn-theme-toggle','btn-toggle-speech','btn-toggle-settings']){const r=await page.locator('#'+id).boundingBox();assert(r.x>=0&&r.x+r.width<=320);}
 assert.deepEqual(errors,[]);await browser.close();fs.writeFileSync(path.join(out,'results.json'),JSON.stringify({browser:'Chromium',layouts:['1194x834','390x844','320x700'],themes:['dark','light'],controls:'passed',phoneRows:2},null,2));console.log('Window/header smoke checks passed');
})().catch(e=>{console.error(e);process.exit(1);});
