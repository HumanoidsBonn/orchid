let pw;try{pw=require('playwright');}catch{pw=require('/tmp/orchid-ui-tools/node_modules/playwright');}
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const ROOT=path.resolve(__dirname,'..'),OUT=path.join(ROOT,'artifacts','mobile-pages');fs.mkdirSync(OUT,{recursive:true});
const BASE=process.env.ORCHID_UI_TEST_URL||'http://127.0.0.1:18000';
const mock=fs.readFileSync(path.join(__dirname,'roslib_mock.js'),'utf8');
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
function tree(depth=4,name='root'){return {id:name,name,node_type:depth?'Sequence':'BtActionLeaf',status:'INVALID',children:depth?[0,1,2].map(i=>tree(depth-1,`${name}-${i}`)):[]};}
async function open(browser,viewport={width:1194,height:834}){
 const page=await browser.newPage({viewport,hasTouch:true});const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.route('**/vendor/roslib.js',r=>r.fulfill({contentType:'text/javascript',body:mock}));
 await page.route('**/api/followup',r=>r.fulfill({contentType:'application/json',body:JSON.stringify({prompt:null,prompt_id:null,ok:true})}));
 await page.goto(BASE);await page.waitForFunction(()=>window.uiMock);await page.click('#btn-connect');await sleep(80);
 await page.evaluate(()=>uiMock.heartbeat({session_id:'a',ready:true,busy:false,task_description:''}));
 return {page,errors};
}
async function heartbeat(page,id='a',busy=false){await page.evaluate(({id,busy})=>uiMock.heartbeat({session_id:id,ready:true,busy,task_description:busy?'Test task':''}),{id,busy});}
async function send(page,name,msg){await page.evaluate(({name,msg})=>{for(const t of uiMock.topics)if(t.name===name)t.callback(msg);},{name,msg});}
async function frame(page){await page.evaluate(()=>{const c=document.createElement('canvas');c.width=640;c.height=360;const x=c.getContext('2d');x.fillStyle='#376c77';x.fillRect(0,0,640,360);x.fillStyle='white';x.font='22px sans-serif';x.fillText('Synthetic preview',35,180);uiMock.emitTopic('/orchid_ui/camera/compressed',c.toDataURL('image/jpeg',.7).split(',')[1]);});await sleep(650);}
async function gesture(page,selector,events){await page.locator(selector).evaluate((el,events)=>{const r=el.getBoundingClientRect();for(const [type,id,x,y] of events)el.dispatchEvent(new PointerEvent(type,{bubbles:true,cancelable:true,pointerId:id,pointerType:'touch',clientX:r.left+x,clientY:r.top+y,button:0,buttons:type==='pointerup'?0:1}));},events);await sleep(80);}
async function exercise(engine,name){
 const browser=await engine.launch({headless:true});const {page,errors}=await open(browser);const health=await page.request.get(BASE+'/healthz');assert(health.ok());assert.equal(health.headers()['cache-control'],'no-store');
 assert.equal(await page.locator('body').getAttribute('data-page'),'map');assert.equal(await page.locator('#lbl-battery').textContent(),'/');
 assert.equal(await page.evaluate(()=>[...uiMock.topics].filter(t=>t.name.endsWith('gui_logs')).length),0);
 await page.fill('#txt-command','Keep this draft');const connections=await page.evaluate(()=>uiMock.sockets.length);
 await page.click('#page-tree');assert.equal(await page.inputValue('#txt-command'),'Keep this draft');
 assert.equal(await page.evaluate(()=>uiMock.sockets.length),connections);
 assert.equal(await page.evaluate(()=>[...uiMock.topics].filter(t=>t.name.endsWith('/compressed')||t.name==='/map').length),0);
 const data=tree();await send(page,'/hsrb_bot/behavior_tree',{data:JSON.stringify(data)});await sleep(300);assert.equal(await page.locator('.tree-node').count(),13);
 const order=await page.locator('[data-node-id="0.0"]').boundingBox();const next=await page.locator('[data-node-id="0.1"]').boundingBox();assert(order.y<next.y);const root=await page.locator('[data-node-id="0"]').boundingBox(),workspace=await page.locator('#workspace').boundingBox();assert(Math.abs(root.x-workspace.x-20)<2);assert(Math.abs(root.y+root.height/2-workspace.y-workspace.height/2)<2);
 let n=data;while(n.children.length){n.status='RUNNING';n=n.children[0];}n.status='RUNNING';await send(page,'/hsrb_bot/behavior_tree',{data:JSON.stringify(data)});await sleep(300);
 const leafBox=await page.locator('[data-node-id="0.0.0.0.0"]').boundingBox();const dy=workspace.y+workspace.height/2-leafBox.y-leafBox.height/2;await gesture(page,'#tree-svg',[['pointerdown',1,100,100],['pointermove',1,100,100+dy],['pointerup',1,100,100+dy]]);await sleep(400);await page.locator('[data-node-id="0.0.0.0.0"]').tap();assert(await page.locator('#tree-details').isVisible());
 await page.click('#close-tree-details');assert(!(await page.locator('#tree-details').isVisible()));
 const original=await page.locator('#tree-viewport').getAttribute('transform');
 await gesture(page,'#tree-svg',[['pointerdown',1,300,200],['pointermove',1,380,240],['pointerup',1,380,240]]);
 assert.notEqual(await page.locator('#tree-viewport').getAttribute('transform'),original);
 await gesture(page,'#tree-svg',[['pointerdown',1,200,150],['pointerdown',2,400,150],['pointermove',1,150,150],['pointermove',2,450,150],['pointerup',2,450,150],['pointermove',1,170,180],['pointercancel',1,170,180]]);
 assert.equal(await page.evaluate(()=>visualViewport.scale),1);
 await page.click('#page-map');await frame(page);const pip=await page.locator('#widget-cam').boundingBox();assert(Math.abs(pip.width/pip.height-640/360)<.03);
 const mapScale=Number(await page.inputValue('#map-scale'));
 await gesture(page,'#map-canvas',[['pointerdown',1,200,150],['pointerdown',2,400,150],['pointermove',2,500,150],['pointerup',2,500,150],['pointermove',1,230,170],['pointerup',1,230,170]]);
 assert(Number(await page.inputValue('#map-scale'))>mapScale);assert.equal(await page.evaluate(()=>visualViewport.scale),1);
 await gesture(page,'#widget-cam',[['pointerdown',1,30,30],['pointermove',1,-80,-40],['pointerup',1,-80,-40]]);
 assert((await page.locator('#widget-cam').boundingBox()).x<pip.x);
 // Execution, long follow-up, error acknowledgement and route persistence.
 await heartbeat(page);await page.click('#btn-send-task');await page.evaluate(()=>uiMock.lastAction.goal.feedback({current_action:'Waiting for input'}));
 await page.route('**/api/followup',r=>r.fulfill({contentType:'application/json',body:JSON.stringify({prompt:'Explain the object location. '.repeat(120),prompt_id:41,ok:true})}));
 await page.waitForSelector('#answer-buttons:not(.hidden)');await page.fill('#txt-followup-answer','The left object');
 await page.setViewportSize({width:390,height:420});await sleep(150);
 for(const id of ['btn-send-followup','btn-cancel-followup']){const r=await page.locator('#'+id).boundingBox();assert(r.y>=0&&r.y+r.height<=420,`${name}: ${id} clipped`);}
 await page.click('#page-tree');assert.equal(await page.inputValue('#txt-followup-answer'),'The left object');
 await page.route('**/api/followup',r=>r.fulfill({status:503,contentType:'application/json',body:'{"ok":false}'}));await page.click('#btn-send-followup');await sleep(80);assert(await page.locator('#answer-buttons').isVisible());assert.equal(await page.inputValue('#txt-followup-answer'),'The left object');
 await page.route('**/api/followup',r=>r.fulfill({contentType:'application/json',body:'{"ok":true,"prompt":null}'}));await page.click('#btn-send-followup');await sleep(80);assert(!(await page.locator('#answer-buttons').isVisible()));
 await heartbeat(page,'b');assert(await page.isEnabled('#btn-send-task'));
 // Bridge reconnect must not replay a goal.
 const goals=await page.evaluate(()=>uiMock.goals.length);await page.evaluate(()=>uiMock.sockets.at(-1).close());await sleep(700);await heartbeat(page,'b');assert.equal(await page.evaluate(()=>uiMock.goals.length),goals);
 await page.click('#btn-toggle-settings');await page.locator('.settings-section').filter({has:page.locator('#chk-logs')}).evaluate(el=>el.open=true);await page.check('#chk-logs');await page.click('#btn-close-sidebar');assert.equal(await page.evaluate(()=>[...uiMock.topics].filter(t=>t.name.endsWith('gui_logs')).length),1);
 await page.click('#btn-toggle-settings');await page.locator('.settings-section').filter({has:page.locator('#chk-logs')}).evaluate(el=>el.open=true);await page.uncheck('#chk-logs');await page.click('#btn-close-sidebar');assert.equal(await page.evaluate(()=>[...uiMock.topics].filter(t=>t.name.endsWith('gui_logs')).length),0);
 // Terminal results and old callbacks cannot alter a newer request.
 for(const result of [{success:true,result_message:'done'},{success:false,result_message:'failed'}]){
  await heartbeat(page,'b');await page.fill('#txt-command','mock only');await page.click('#btn-send-task');
  await page.evaluate(result=>uiMock.lastAction.goal.result(result),result);assert(await page.isEnabled('#btn-send-task'));
 }
 await page.fill('#txt-command','rejected mock');await page.click('#btn-send-task');await page.evaluate(()=>{window.oldRejectedGoal=uiMock.lastAction.goal;window.oldRejectedGoal.error('rejected');});assert(await page.isEnabled('#btn-send-task'));
 await page.fill('#txt-command','new mock');await page.click('#btn-send-task');await page.evaluate(()=>{window.oldRejectedGoal.result({success:true});});assert(await page.isVisible('#btn-cancel-task'));await page.click('#btn-cancel-task');
 // Bounded log insertion and battery availability.
 await page.click('#btn-toggle-settings');await page.locator('.settings-section').filter({has:page.locator('#chk-logs')}).evaluate(el=>el.open=true);await page.check('#chk-logs');await page.click('#btn-close-sidebar');
 await page.evaluate(()=>{for(let i=0;i<900;i++)uiMock.emitTopic('/hsrb_bot/gui_logs','log '+i);});await sleep(300);assert(await page.locator('.log-line').count()<=500);
 await page.click('#btn-toggle-settings');await page.locator('.settings-section').filter({has:page.locator('#chk-logs')}).evaluate(el=>el.open=true);await page.uncheck('#chk-logs');await page.click('#btn-close-sidebar');
 const layouts=[];
 const displayTree=tree(2);
 const displayMap={info:{width:60,height:40,resolution:.1,origin:{position:{x:0,y:0},orientation:{w:1}}},data:Array.from({length:2400},(_,i)=>i<60||i>=2340||i%60===0||i%60===59?100:0)};
 for(const [width,height]of [[1194,834],[1194,700],[834,1194],[390,844],[390,420]]){
  await page.setViewportSize({width,height});await heartbeat(page,'b');await sleep(100);
  for(const route of ['map','tree']){await page.click('#page-'+route);if(route==='map'){await send(page,'/map',displayMap);await page.click('#btn-reset-map');await frame(page);await page.locator('#widget-cam').evaluate(el=>{el.style.left='';el.style.top='';el.style.right='';el.style.bottom='';});}else{await send(page,'/hsrb_bot/behavior_tree',{data:JSON.stringify(displayTree)});await sleep(250);await page.click('#btn-reset-tree');}await sleep(150);const metrics=await page.evaluate(()=>({w:innerWidth,scroll:document.documentElement.scrollWidth,h:innerHeight,body:document.body.scrollHeight}));assert(metrics.scroll<=width+1);assert(metrics.body<=height+1);layouts.push({route,width,height,...metrics});await page.screenshot({path:path.join(OUT,`${name}-${route}-${width}x${height}.png`)});}
 }
 await page.setViewportSize({width:1194,height:834});await page.click('#btn-theme-toggle');await page.click('#page-tree');await page.click('#btn-reset-tree');const colors=await page.locator('.tree-node-text').first().evaluate(el=>getComputedStyle(el).fill);assert.equal(colors,'rgb(21, 34, 56)');await page.locator('.tree-node').last().tap();assert(await page.locator('#tree-details').isVisible());await page.screenshot({path:path.join(OUT,`${name}-tree-light-details.png`)});assert.deepEqual(errors,[]);await browser.close();return{name,layouts};
}
async function soak(){
 const browser=await pw.chromium.launch({headless:true}),clients=[];
 for(let i=0;i<2;i++){const c=await open(browser);clients.push(c);}
 const seconds=Number(process.env.ORCHID_UI_SOAK_SECONDS||1200),samples=[],data=tree(5);
 const map={info:{width:100,height:100,resolution:.05,origin:{position:{x:0,y:0},orientation:{w:1}}},data:Array(10000).fill(0)};
 for(const {page}of clients)await page.evaluate(({data,map})=>{
  const emit=(name,msg)=>{for(const t of uiMock.topics)if(t.name===name)t.callback(msg);};
  const canvas=document.createElement('canvas');canvas.width=640;canvas.height=360;canvas.getContext('2d').fillRect(0,0,640,360);const jpeg=canvas.toDataURL('image/jpeg',.7).split(',')[1];
  setInterval(()=>uiMock.heartbeat({session_id:'soak',ready:true,busy:false,task_description:''}),1000);
  setInterval(()=>emit('/hsrb_bot/behavior_tree',{data:JSON.stringify(data)}),200);
  setInterval(()=>emit('/orchid_ui/camera/compressed',{data:jpeg}),500);
  setInterval(()=>emit('/map',map),2000);
  setInterval(()=>location.hash=location.hash==='#tree'?'#map':'#tree',15000);
 },{data,map});
 const began=Date.now();
 while(Date.now()-began<seconds*1000){
  await sleep(Math.min(15000,seconds*1000-(Date.now()-began)));
  const health=await clients[0].page.request.get(BASE+'/healthz',{timeout:3000});assert(health.ok());
  for(let i=0;i<clients.length;i++){const {page,errors}=clients[i];assert.deepEqual(errors,[]);const row=await page.evaluate(()=>({dom:document.querySelectorAll('*').length,subscriptions:uiMock.topics.size,treeNodes:document.querySelectorAll('.tree-node').length,heap:performance.memory?.usedJSHeapSize||null,goals:uiMock.goals.length}));assert(row.dom<2000);assert(row.subscriptions<=9);assert.equal(row.goals,0);samples.push({seconds:Math.round((Date.now()-began)/1000),client:i,...row});}
  fs.writeFileSync(path.join(OUT,'soak-progress.json'),JSON.stringify({elapsed_s:Math.round((Date.now()-began)/1000),target_s:seconds,samples},null,2));
 }
 await browser.close();fs.writeFileSync(path.join(OUT,'soak-results.json'),JSON.stringify({duration_s:seconds,clients:2,samples},null,2));console.log('SOAK_PASS',seconds);
}
if(require.main===module)(async()=>{if(process.argv.includes('--soak'))await soak();else{const out=[];for(const [name,engine]of [['chromium',pw.chromium],['webkit',pw.webkit]])out.push(await exercise(engine,name));fs.writeFileSync(path.join(OUT,'browser-results.json'),JSON.stringify(out,null,2));console.log('BROWSER_PASS',JSON.stringify(out));}})().catch(e=>{console.error(e);process.exit(1);});
