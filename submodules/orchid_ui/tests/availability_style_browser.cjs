// Focused availability/style regression only. ROS and follow-up transport are mocked.
let pw;try{pw=require('playwright');}catch{pw=require('/tmp/orchid-ui-tools/node_modules/playwright');}
const fs=require('fs'),path=require('path'),assert=require('assert/strict');
const root=path.resolve(__dirname,'..'),base=process.env.ORCHID_UI_TEST_URL||'http://127.0.0.1:8000';
const out=path.join(root,'artifacts','availability-style');fs.mkdirSync(out,{recursive:true});
const mock=fs.readFileSync(path.join(__dirname,'roslib_mock.js'),'utf8');
async function run(engine,name){
 const browser=await engine.launch({headless:true}),page=await browser.newPage({viewport:{width:1194,height:834},hasTouch:true});const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.clock.install();await page.route('**/vendor/roslib.js',r=>r.fulfill({contentType:'text/javascript',body:mock}));
 await page.route('**/api/followup',r=>r.fulfill({contentType:'application/json',body:JSON.stringify({prompt:'Which object?',prompt_id:7})}));
 await page.goto(base);await page.waitForFunction(()=>window.uiMock);await page.click('#btn-connect');await page.clock.runFor(30);
 const phase=()=>page.locator('#lbl-robot-status').getAttribute('data-availability');
 const hb=async(id='a',busy=false)=>page.evaluate(({id,busy})=>uiMock.heartbeat({session_id:id,ready:true,busy,task_description:busy?'Mock task':''}),{id,busy});
 const heavy=()=>page.evaluate(()=>[...uiMock.topics].filter(t=>t.name==='/map'||t.name.endsWith('/compressed')||t.name.endsWith('/behavior_tree')).length);
 assert.equal(await phase(),'waiting');assert.equal(await heavy(),0);await hb();assert.equal(await phase(),'ready');assert(await heavy()>0);
 await page.clock.fastForward(11000);assert.equal(await phase(),'delayed');assert(await page.isEnabled('#btn-send-task'));await hb();await page.clock.runFor(1000);await hb();await page.clock.runFor(1000);await hb();
 await page.fill('#txt-command','Mock task');await page.click('#btn-send-task');await page.evaluate(()=>{window.savedGoal=uiMock.lastAction.goal;savedGoal.feedback({current_action:'Waiting for input'});});
 await page.clock.runFor(1100);await page.waitForSelector('#answer-buttons:not(.hidden)');await page.fill('#txt-followup-answer','Keep this answer');
 await page.clock.fastForward(11000);assert.equal(await phase(),'delayed');assert.equal(await heavy(),0);assert.equal(await page.inputValue('#txt-followup-answer'),'Keep this answer');assert(await page.isVisible('#btn-cancel-followup'));
 // Keep other ROS traffic alive while the session heartbeat remains stale.
 await page.evaluate(()=>uiMock.emitTopic('/hsrb_bot/gui_state','{"robot_status":"Running"}'));
 await page.clock.fastForward(19000);assert.equal(await phase(),'unknown');assert(await page.isDisabled('#btn-send-task'));assert.equal(await page.inputValue('#txt-followup-answer'),'Keep this answer');
 // The original action callback is still valid after heartbeat expiry.
 await page.evaluate(()=>savedGoal.result({success:true}));assert.equal(await page.locator('#action-status-badge').textContent(),'Completed');
 await hb();assert.equal(await heavy(),0);await hb();await hb();assert.equal(await heavy(),0);await page.clock.runFor(1000);await hb();assert.equal(await heavy(),0);await page.clock.runFor(1000);await hb();assert(await heavy()>0);assert(await page.isEnabled('#btn-send-task'));
 // Foregrounding preserves a new task and its follow-up answer.
 await page.fill('#txt-command','Second task');await page.click('#btn-send-task');await page.evaluate(()=>{window.oldGoal=uiMock.lastAction.goal;oldGoal.feedback({current_action:'Waiting for input'});});await page.clock.runFor(1100);await page.fill('#txt-followup-answer','Still here');
 await page.evaluate(()=>{Object.defineProperty(document,'hidden',{configurable:true,get:()=>true});document.dispatchEvent(new Event('visibilitychange'));Object.defineProperty(document,'hidden',{configurable:true,get:()=>false});document.dispatchEvent(new Event('visibilitychange'));});
 assert.equal(await phase(),'resynchronizing');assert.equal(await page.inputValue('#txt-followup-answer'),'Still here');await hb('a',true);assert.equal(await page.inputValue('#txt-followup-answer'),'Still here');
 // Explicit restart invalidates old request callbacks and reconciles fresh idle state.
 await hb('b');assert(await page.isEnabled('#btn-send-task'));await page.fill('#txt-command','New instance task');await page.click('#btn-send-task');await page.evaluate(()=>oldGoal.result({success:true}));assert(await page.isVisible('#btn-cancel-task'));
 // Genuine socket loss keeps task state; reconnect never replays the goal.
 const goals=await page.evaluate(()=>uiMock.goals.length);await page.evaluate(()=>uiMock.sockets.at(-1).close());assert.equal(await phase(),'connecting');assert(await page.isVisible('#btn-cancel-task'));await page.clock.runFor(550);await hb('b',true);assert.equal(await page.evaluate(()=>uiMock.goals.length),goals);
 await hb('b',false);assert(await page.isEnabled('#btn-send-task'));
 // No inbound ROS at all causes exactly one replacement socket, with bounded topics.
 const sockets=await page.evaluate(()=>uiMock.sockets.length);await page.clock.fastForward(31000);await page.clock.runFor(550);assert.equal(await page.evaluate(()=>uiMock.sockets.length),sockets+1);assert.equal(await heavy(),0);await hb('b');assert.equal(await phase(),'ready');
 // Missing acknowledgement must not discard a possibly accepted goal.
 await page.route('**/api/followup',r=>r.fulfill({contentType:'application/json',body:'{"prompt":null}'}));
 await page.fill('#txt-command','Await acknowledgement');await page.click('#btn-send-task');await page.clock.fastForward(11000);assert(await page.isVisible('#btn-cancel-task'));await page.evaluate(()=>uiMock.lastAction.goal.result({success:true}));await hb('b');
 // Style fixtures keep the session healthy while taking screenshots.
 await page.evaluate(()=>setInterval(()=>uiMock.heartbeat({session_id:'b',ready:true,busy:false,task_description:''}),1000));await page.clock.runFor(3100);
 // Waiting-tree message is a single ellipsized line in both themes and sizes.
 await page.click('#page-tree');await page.clock.runFor(1100);await hb('b');await page.clock.runFor(1100);await hb('b');
 for(const theme of ['dark','light']){
  if(theme==='light')await page.click('#btn-theme-toggle');
  for(const [width,height]of [[1194,834],[390,844]]){
   await page.setViewportSize({width,height});await page.clock.runFor(250);
   const metrics=await page.locator('.tree-placeholder-line').evaluate(el=>({line:getComputedStyle(el).whiteSpace,overflow:getComputedStyle(el).textOverflow,height:el.getBoundingClientRect().height,width:el.getBoundingClientRect().width,screen:innerWidth}));
   assert.equal(metrics.line,'nowrap');assert.equal(metrics.overflow,'ellipsis');assert(metrics.height<30);assert(metrics.width<=width);assert(await page.locator('.orchid-brand').isVisible());
   assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
   await page.screenshot({path:path.join(out,`${name}-${theme}-${width}-waiting.png`)});
  }
 }
 for(const theme of ['dark','light']){
  if((await page.locator('body').getAttribute('class')||'').includes('light-theme') !== (theme==='light'))await page.click('#btn-theme-toggle');
  await page.setViewportSize({width:1194,height:834});
  await page.evaluate(()=>uiMock.emitTopic('/hsrb_bot/behavior_tree',JSON.stringify({name:'Pickup',node_type:'Sequence',status:'RUNNING',children:[{name:'Complete',node_type:'BtActionLeaf',status:'SUCCESS'},{name:'Approach',node_type:'BtActionLeaf',status:'RUNNING'},{name:'Rejected candidate',node_type:'BtActionLeaf',status:'FAILURE'}]})));
  await page.clock.runFor(500);
  assert.equal(await page.locator('.tree-node-rect.SUCCESS').count(),1);assert.equal(await page.locator('.tree-node-rect.FAILURE').count(),1);
  const ink=await page.locator('.tree-node-text').first().evaluate(el=>getComputedStyle(el).fill);assert.equal(ink,theme==='light'?'rgb(21, 34, 56)':'rgb(241, 245, 252)');
  await page.screenshot({path:path.join(out,`${name}-${theme}-nodes.png`)});
  await page.click('#page-map');await page.waitForFunction(()=>document.body.dataset.page==='map');await page.clock.runFor(500);const pixel=await page.locator('#map-canvas').evaluate(el=>Array.from(el.getContext('2d').getImageData(100,100,1,1).data).slice(0,3));assert.deepEqual(pixel,theme==='light'?[237,241,249]:[11,15,25]);await page.screenshot({path:path.join(out,`${name}-${theme}-map.png`)});await page.click('#page-tree');
 }
 assert.deepEqual(errors,[]);await browser.close();return{name,availability:'passed',styles:'passed'};
}
(async()=>{const results=[];for(const [name,engine]of [['chromium',pw.chromium],['webkit',pw.webkit]])results.push(await run(engine,name));fs.writeFileSync(path.join(out,'results.json'),JSON.stringify(results,null,2));console.log(JSON.stringify(results));})().catch(e=>{console.error(e);process.exit(1);});
