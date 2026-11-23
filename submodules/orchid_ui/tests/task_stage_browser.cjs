// Status-only UI checks; no task goals or robot commands.
const pw=require('/tmp/orchid-ui-tools/node_modules/playwright');
const fs=require('fs'),path=require('path'),assert=require('assert/strict');
(async()=>{
 for(const engine of ['chromium','webkit']){
  const browser=await pw[engine].launch({headless:true});
  for(const width of [1194,390,320]){
   const page=await browser.newPage({viewport:{width,height:width===1194?834:700},hasTouch:true});const errors=[];
   page.on('pageerror',e=>errors.push(e.message));
   await page.route('**/vendor/roslib.js',r=>r.fulfill({contentType:'text/javascript',body:fs.readFileSync(path.join(__dirname,'roslib_mock.js'),'utf8')}));
   await page.route('**/api/followup',r=>r.fulfill({contentType:'application/json',body:'{"prompt":null}'}));
   await page.goto('http://127.0.0.1:18088/#map');await page.waitForFunction(()=>window.uiMock);
   assert(await page.locator('#task-stage').isHidden());
   await page.click('#btn-connect');await page.waitForTimeout(80);
   const beat=async(busy,id='stage-session')=>page.evaluate(({busy,id})=>{
    window.stageHeartbeat={session_id:id,ready:true,busy,task_description:busy?'Fetch coffee mug':''};
    uiMock.heartbeat(window.stageHeartbeat);
    window.stageHeartbeatTimer ||= setInterval(()=>uiMock.heartbeat(window.stageHeartbeat),1000);
   },{busy,id});
   await beat(true);await page.waitForFunction(()=>[...uiMock.topics].some(t=>t.name.endsWith('/gui_state')));
   const stage=async text=>page.evaluate(text=>uiMock.emitTopic('/hsrb_bot/gui_state',JSON.stringify({session_id:'stage-session',robot_status:'Running',task_stage:text})),text);
   await stage('Verifying at known location');await page.waitForFunction(()=>document.querySelector('#task-stage').textContent==='Verifying at known location');
   for(const [raw,expected] of [['Navigation Running','Navigation'],['Pickup Running','Pickup'],['Planning Running','Planning'],['Navigation Failed','Navigation Failed']]){
    await page.evaluate(raw=>uiMock.emitTopic('/hsrb_bot/gui_state',JSON.stringify({session_id:'stage-session',robot_status:raw})),raw);
    await page.waitForFunction(expected=>document.querySelector('#lbl-robot-status').textContent===expected,expected);
   }
   for(const theme of ['dark','light']){
    await page.evaluate(theme=>document.body.classList.toggle('light-theme',theme==='light'),theme);
    const b=await page.locator('#task-stage').boundingBox(),input=await page.locator('#txt-command').boundingBox();
    const style=await page.locator('#task-stage').evaluate(el=>({font:parseFloat(getComputedStyle(el).fontSize),edge:parseFloat(getComputedStyle(el).borderLeftWidth)}));
    assert.equal(style.font,width<=600?14:17);assert.equal(style.edge,1);
    assert(b.width<=(width<=600?96:440)+1);assert(b.x+b.width<=input.x+1);
    if(width>600)assert(Math.abs(b.width-440)<=1);
    assert.equal(await page.locator('#task-stage').evaluate(el=>getComputedStyle(el,'::after').animationName),'stage-border-breathe');
    assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
   }
   await page.click('#page-tree');assert.equal(await page.locator('#task-stage').textContent(),'Verifying at known location');
   await stage('Asking for drop');assert(await page.locator('#task-stage').isVisible());
   await stage('An unusually long task-stage description that must wrap within the available card width');
   assert(await page.locator('#task-stage').evaluate(el=>el.scrollWidth<=el.clientWidth+1));
   await page.route('**/api/followup',r=>r.fulfill({contentType:'application/json',body:JSON.stringify({prompt:'Where should I bring it? '.repeat(80),prompt_id:41})}));
   await page.waitForSelector('#answer-buttons:not(.hidden)');
   await page.fill('#txt-followup-answer','The kitchen table');
   assert(await page.locator('#task-stage').isVisible());
   for(const id of ['btn-send-followup','btn-cancel-followup']){
    const rect=await page.locator('#'+id).boundingBox();
    assert(rect && rect.x>=0 && rect.x+rect.width<=width+1 && rect.y+rect.height<=(width===1194?834:700));
   }
   await page.click('#page-map');assert.equal(await page.inputValue('#txt-followup-answer'),'The kitchen table');
   await beat(false);await page.waitForSelector('#task-stage',{state:'hidden'});
   await beat(true);await stage('Performing pickup');await beat(false,'new-session');assert(await page.locator('#task-stage').isHidden());
   assert.equal(await page.evaluate(()=>uiMock.goals.length),0);assert.deepEqual(errors,[]);
   await page.close();
  }
  await browser.close();console.log(`${engine}: stage updates, lifecycle, page switching and tablet/phone wrapping passed`);
 }
})().catch(e=>{console.error(e);process.exit(1);});
