// Workflow UI only; ROSLIB and prompt transport are intercepted, no robot goals.
let pw;try{pw=require('playwright');}catch{pw=require('/tmp/orchid-ui-tools/node_modules/playwright');}
const fs=require('fs'),path=require('path'),assert=require('assert/strict');const root=path.resolve(__dirname,'..'),out=path.join(root,'artifacts/workflows');fs.mkdirSync(out,{recursive:true});
(async()=>{
 const browser=await pw.chromium.launch({headless:true}),page=await browser.newPage({viewport:{width:1194,height:834},hasTouch:true});const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.route('**/vendor/roslib.js',r=>r.fulfill({contentType:'text/javascript',body:fs.readFileSync(path.join(__dirname,'roslib_mock.js'),'utf8')}));let question=false,answer=null;
 await page.route('**/api/followup',async r=>{if(r.request().method()==='POST'){answer=r.request().postDataJSON();question=false;return r.fulfill({contentType:'application/json',body:'{"ok":true}'});}return r.fulfill({contentType:'application/json',body:JSON.stringify(question?{prompt:'Where should I bring it?',prompt_id:42,choices:[{value:'table_1',label:'Table — kitchen'}]}:{prompt:null})});});
 await page.goto('http://127.0.0.1:8000/#tasks');await page.waitForFunction(()=>window.uiMock);await page.click('#btn-connect');await page.waitForTimeout(50);await page.evaluate(()=>{const beat=()=>uiMock.heartbeat({session_id:'workflow-test',ready:true,busy:false,task_description:''});beat();window.workflowHeartbeatTimer=setInterval(beat,1000);});
 const catalog={schema_version:1,robot_id:'hsrb_bot',session_id:'workflow-test',revision:'one',workflows:[{id:'release',title:'Release',available:true,fields:[]},{id:'home',title:'Home',available:true,fields:[]},{id:'fetch',title:'Fetch',available:true,fields:[{name:'item_id',label:'Object',type:'select',choices:'items',required:true}]}],items:[{id:'jar',label:'Blue jar',query:'small blue cylindrical jar'},{id:'banana',label:'Banana'}],locations:[{id:'table_1',label:'Table — kitchen'}]};
 const emit=async data=>page.evaluate(data=>uiMock.emitTopic('/hsrb_bot/workflow_catalog',JSON.stringify(data)),data);
 await emit(catalog);assert.equal(await page.locator('.workflow-card').count(),3);await page.fill('#txt-command','Keep this free-text draft');
 await page.selectOption('.workflow-card select','jar');await page.locator('.workflow-card select').evaluate(el=>window.cachedSelect=el);await emit(catalog);assert(await page.locator('.workflow-card select').evaluate(el=>el===window.cachedSelect));
 for(const id of ['page-map','page-tree','page-tasks'])await page.click('#'+id);
 assert.equal(await page.evaluate(()=>[...uiMock.topics].filter(t=>t.name.endsWith('/workflow_catalog')).length),1);assert.equal(await page.evaluate(()=>uiMock.sockets.length),1);
 for(const name of ['Release','Home','Fetch']){
  await page.getByRole('button',{name,exact:true}).click();const last=await page.evaluate(()=>({name:uiMock.lastAction.name,args:uiMock.lastAction.goal.args}));assert(last.name.endsWith('/execute_workflow'));assert.equal(last.args.workflow_id,name.toLowerCase());assert.deepEqual(JSON.parse(last.args.inputs_json),name==='Fetch'?{item_id:'jar'}:{});
  if(name==='Fetch'){
   assert((await page.locator('#current-request').textContent()).includes('Fetch Blue jar'));
   assert(!(await page.locator('#current-request').textContent()).includes('small blue cylindrical jar'));
   question=true;await page.waitForSelector('#followup-choices:not(.hidden)');await page.selectOption('#followup-choices','table_1');assert.equal(await page.inputValue('#txt-followup-answer'),'table_1');await page.click('#btn-send-followup');assert.equal(answer.prompt_id,42);assert.equal(answer.response,'table_1');
  }
  await page.evaluate(()=>uiMock.lastAction.goal.result({success:true}));assert.equal(await page.inputValue('#txt-command'),'Keep this free-text draft');
 }
 // New catalog revision replaces choices, without opening another connection.
 await emit({...catalog,revision:'two',items:[...catalog.items,{id:'apple',label:'Apple'}]});assert.equal(await page.locator('.workflow-card select option').count(),4);
 await emit({...catalog,session_id:'workflow-new',revision:'two',items:[...catalog.items,{id:'apple',label:'Apple'}]});await page.evaluate(()=>{clearInterval(window.workflowHeartbeatTimer);const beat=()=>uiMock.heartbeat({session_id:'workflow-new',ready:true,busy:false,task_description:''});beat();setInterval(beat,1000);});assert(await page.getByRole('button',{name:'Fetch',exact:true}).isEnabled());
 await page.screenshot({path:path.join(out,'tasks-desktop.png')});await page.setViewportSize({width:390,height:844});assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));await page.screenshot({path:path.join(out,'tasks-phone.png')});
 await page.setViewportSize({width:320,height:700});assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
 await page.reload();await page.waitForSelector('.workflow-card');assert((await page.locator('#workflow-catalog-status').textContent()).includes('Cached'));assert(await page.getByRole('button',{name:'Fetch',exact:true}).isDisabled());assert.equal(await page.evaluate(()=>uiMock.goals.length),0);
 assert.deepEqual(errors,[]);await browser.close();fs.writeFileSync(path.join(out,'results.json'),JSON.stringify({catalogCache:true,oneSocket:true,launches:['release','home','fetch'],choices:true,cacheReload:true,layouts:['1194x834','390x844']},null,2));console.log('Workflow UI checks passed');
})().catch(e=>{console.error(e);process.exit(1);});
