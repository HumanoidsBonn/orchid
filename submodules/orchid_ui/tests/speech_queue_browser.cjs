const pw=require('/tmp/orchid-ui-tools/node_modules/playwright');
const fs=require('fs'),path=require('path'),assert=require('assert/strict');
(async()=>{
 for(const engine of ['chromium','webkit']){
  const browser=await pw[engine].launch({headless:true});const page=await browser.newPage();
  await page.addInitScript(()=>{
   window.voiceTest={spoken:[],cancelled:0};
   Object.defineProperty(window,'speechSynthesis',{value:{getVoices:()=>[],addEventListener:()=>{},speak:u=>voiceTest.spoken.push(u.text),cancel:()=>voiceTest.cancelled++}});
   window.SpeechSynthesisUtterance=class{constructor(text){this.text=text;}};
  });
  await page.route('**/vendor/roslib.js',r=>r.fulfill({contentType:'text/javascript',body:fs.readFileSync(path.join(__dirname,'roslib_mock.js'),'utf8')}));
  await page.route('**/api/followup',r=>r.fulfill({contentType:'application/json',body:'{"prompt":null,"ok":true}'}));
  const connect=async()=>{
   await page.click('#btn-connect');await page.waitForTimeout(60);
   await page.evaluate(()=>uiMock.heartbeat({session_id:'speech-test',ready:true,busy:false}));
   await page.waitForFunction(()=>[...uiMock.topics].some(t=>t.name==='/hsrb_bot/ui_speech'));
  };
  const say=async text=>page.evaluate(text=>uiMock.emitTopic('/hsrb_bot/ui_speech',text),text);
  const toggle=async value=>page.evaluate(value=>{const el=document.querySelector('#speech-interrupt');el.checked=value;el.dispatchEvent(new Event('change'));},value);
  await page.goto('http://127.0.0.1:8000');await page.waitForFunction(()=>window.uiMock);await connect();
  assert.equal(await page.locator('#speech-interrupt').isChecked(),false);
  await say('First');await say('Second');
  assert.deepEqual(await page.evaluate(()=>voiceTest),{spoken:['First','Second'],cancelled:0});
  await toggle(true);await say('Third');
  assert.equal(await page.evaluate(()=>voiceTest.cancelled),1);
  await page.reload();await page.waitForFunction(()=>window.uiMock);assert(await page.locator('#speech-interrupt').isChecked());await connect();
  await say('Fourth');assert.equal(await page.evaluate(()=>voiceTest.cancelled),1);
  await toggle(false);await say('Fifth');assert.equal(await page.evaluate(()=>voiceTest.cancelled),1);
  await page.click('#btn-toggle-speech');await say('Muted');
  assert.equal(await page.evaluate(()=>voiceTest.cancelled),2);
  assert.deepEqual(await page.evaluate(()=>voiceTest.spoken),['Fourth','Fifth']);
  await browser.close();console.log(engine+': queued default, interruption toggle, persistence and mute passed');
 }
})().catch(e=>{console.error(e);process.exit(1);});
