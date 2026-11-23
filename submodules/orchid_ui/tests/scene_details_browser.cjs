// Focused scene/UI smoke checks; all ROS traffic is mocked and no goals are sent.
let pw;try{pw=require('playwright');}catch{pw=require('/tmp/orchid-ui-tools/node_modules/playwright');}
const fs=require('fs'),path=require('path'),assert=require('assert/strict');
const out=path.resolve(__dirname,'../artifacts/scene-details');
(async()=>{
 fs.mkdirSync(out,{recursive:true}); const results=[];
 for(const engine of ['chromium','webkit']) {
  const browser=await pw[engine].launch({headless:true});
  const page=await browser.newPage({viewport:{width:1194,height:834},hasTouch:true});
  const errors=[],external=[];page.on('pageerror',e=>errors.push(e.message));
  page.on('request',r=>{if(!r.url().startsWith('http://127.0.0.1:8000/')&&!r.url().startsWith('data:')&&!r.url().startsWith('blob:'))external.push(r.url());});
  await page.route('**/vendor/roslib.js',r=>r.fulfill({contentType:'text/javascript',body:fs.readFileSync(path.join(__dirname,'roslib_mock.js'),'utf8')}));
  await page.route('**/api/followup',r=>r.fulfill({contentType:'application/json',body:'{"prompt":null}'}));
  const label='illustrative_work_desk_with_a_long_name';
  const graph={metadata:{world_frame:'map',relationship_config:'../rules/custom.yaml'},objects:[{id:41,label,center:[0,0,.4],size:[1,1,.8],rotation:[[1,0,0],[0,1,0],[0,0,1]],room_id:'office',room_type:'office',base_access_poses:[]}]};
  await page.route('**/fixtures/scene.json',r=>r.fulfill({contentType:'application/json',body:JSON.stringify(graph)}));
  await page.route('**/rules/custom.yaml',r=>r.fulfill({contentType:'text/yaml',body:`static_furniture:\n  - ${label}\nnull_classes:\n  - partition\n`}));
  await page.route('**/fixtures/bad.json',r=>r.fulfill({contentType:'application/json',body:JSON.stringify({...graph,metadata:{world_frame:'map',relationship_config:'missing.yaml'}})}));
  await page.route('**/fixtures/missing.yaml',r=>r.fulfill({status:404,body:'missing'}));
  await page.addInitScript(()=>localStorage.setItem('map-source','topic'));
  await page.goto('http://127.0.0.1:8000/#map');await page.waitForFunction(()=>window.uiMock);
  const defaultCount=JSON.parse(fs.readFileSync(path.join(__dirname,'../static/scene_graph_complete.json'),'utf8')).objects.length;
  await page.waitForFunction(count=>document.querySelector('#map-status').textContent.includes(`${count} objects`),defaultCount);
  // Real YAML parser, exact relative URL resolution, configuration validation and parity.
  const parsed=await page.evaluate(async()=>{
   const m=await import('/scene_data.js?v=20261028a');const calls=[];
   const loaded=await m.loadSceneDataset('/nested/graph.json',{baseUrl:location.href,fetcher:async url=>{
    calls.push(url);return new Response(url.endsWith('.json')?JSON.stringify({metadata:{relationship_config:'../rules.yaml'},objects:[]}):'static_furniture: [custom_seat]\nnull_classes: [partition]\n');
   }});
   let invalid='';try{await m.loadSceneDataset('/bad.json',{baseUrl:location.href,fetcher:async url=>new Response(url.endsWith('.json')?JSON.stringify({metadata:{relationship_config:'rules.yaml'},objects:[]}):'static_furniture: invalid\nnull_classes: []\n')});}catch(e){invalid=e.message;}
   return {calls,categories:[m.sceneObjectCategory({label:'custom_seat'},loaded.classes),m.sceneObjectCategory({label:'partition',supported_by:1},loaded.classes),m.sceneObjectCategory({label:'coffee_mug'},loaded.classes)],invalid};
  });
  assert.deepEqual(parsed.categories,['furniture','structural','movable']);assert(parsed.calls[1].endsWith('/rules.yaml'));assert(parsed.invalid.includes('static_furniture must be a list'));
  await page.click('#btn-connect');await page.waitForTimeout(40);
  await page.evaluate(()=>{const beat=()=>uiMock.heartbeat({session_id:'scene-details',ready:true,busy:false});beat();setInterval(beat,1000);});
  const load=async value=>{await page.evaluate(value=>{document.querySelector('#scene-graph-file').value=value;document.querySelector('#btn-load-scene-graph').click();},value);};
  await load('/fixtures/scene.json');await page.waitForFunction(()=>document.querySelector('#map-status').textContent.includes('1 objects'));
  await page.evaluate(()=>[...uiMock.topics].find(t=>t.name==='/map').callback({info:{width:5,height:5,resolution:1,origin:{position:{x:-2,y:-2},orientation:{w:1}}},data:Array(25).fill(0)}));
  await page.waitForTimeout(250);
  const selectMap=async()=>{
   const point=await page.evaluate(()=>{const c=document.querySelector('#map-canvas'),r=c.getBoundingClientRect(),s=Number(document.querySelector('#map-scale').value),x=Number(document.querySelector('#map-origin-x').value)+2*s,y=Number(document.querySelector('#map-origin-y').value)+2*s;return{x:r.left+x*r.width/c.width,y:r.top+y*r.height/c.height};});
   await page.mouse.click(point.x,point.y);await page.waitForSelector('#scene-info:not(.hidden)');
  };
  await selectMap();assert((await page.locator('#scene-info-content').textContent()).includes('furniture'));
  for(const [width,height]of [[1194,834],[390,844]]) {
   await page.setViewportSize({width,height});await page.waitForTimeout(300);
   await page.evaluate(()=>{const c=document.querySelector('#map-canvas');document.querySelector('#map-scale').value=20;document.querySelector('#map-origin-x').value=c.width/2-40;document.querySelector('#map-origin-y').value=c.height/2-40;document.querySelector('#btn-save-map-settings').click();});
   await page.waitForTimeout(250);await selectMap();
   const panel=await page.locator('#scene-info').boundingBox();assert(panel.width<=380.5);assert(panel.x>=0&&panel.x+panel.width<=width);
   const title=await page.locator('.scene-info-title').boundingBox();assert(title.width>200);
   await page.screenshot({path:path.join(out,`${engine}-map-${width}.png`)});
   await page.getByRole('button',{name:'Close scene information',exact:true}).click();assert(await page.locator('#scene-info').isHidden());
  }
  // Failed reload preserves the previous graph/classes as one coherent dataset.
  await load('/fixtures/bad.json');await page.waitForFunction(()=>document.querySelector('#map-status').textContent.includes('showing previous graph'));
  await selectMap();assert((await page.locator('#scene-info-content').textContent()).includes('furniture'));
  await page.click('#btn-close-scene-info');
  await page.setViewportSize({width:1194,height:834});await page.click('#page-tree');
  const tree={id:'root',name:'Fetch',node_type:'Sequence',status:'RUNNING',children:[{id:'search',name:'Search locations',node_type:'Selector',status:'RUNNING',children:[{id:'scan',name:'Scan for mug',node_type:'Action',status:'RUNNING',children:[]},{id:'observe',name:'Observe mug',node_type:'Action',status:'INVALID',children:[]}]}]};
  const emit=async t=>{await page.evaluate(t=>uiMock.emitTopic('/hsrb_bot/behavior_tree',JSON.stringify(t)),t);await page.waitForTimeout(250);};
  await emit(tree);assert(await page.locator('#tree-details').isHidden());
  await page.locator('.tree-node[data-node-id="0.0"]').click();assert(await page.locator('#tree-details').isVisible());assert((await page.locator('#tree-details').textContent()).includes('Search locations'));
  assert.equal(await page.locator('.tree-node').count(),2); // Same tap collapsed the children.
  await page.locator('#tree-details summary').click();
  tree.children[0].status='SUCCESS';tree.status='SUCCESS';await emit(tree);
  assert(await page.locator('#tree-details').isVisible());assert(await page.locator('.tree-raw').evaluate(el=>el.open));
  assert((await page.locator('#tree-details').textContent()).includes('SUCCESS'));
  // Move the intentionally floating panel away from the node before tapping it again.
  const handle=await page.locator('.tree-details-title').boundingBox(),workspace=await page.locator('#workspace').boundingBox();
  await page.mouse.move(handle.x+30,handle.y+8);await page.mouse.down();await page.mouse.move(workspace.x+workspace.width-180,workspace.y+30,{steps:5});await page.mouse.up();
  await page.locator('.tree-node[data-node-id="0.0"]').click();assert.equal(await page.locator('.tree-node').count(),4);
  await page.locator('.tree-node[data-node-id="0.0.0"]').click();assert((await page.locator('#tree-details').textContent()).includes('Scan for mug'));
  await page.getByRole('button',{name:'Close node details',exact:true}).click();assert(await page.locator('#tree-details').isHidden());
  await emit({...tree,status:'RUNNING'});assert(await page.locator('#tree-details').isHidden()); // Auto expansion never opens details.
  await page.locator('.tree-node[data-node-id="0"]').click();assert(await page.locator('#tree-details').isVisible());
  await page.screenshot({path:path.join(out,`${engine}-bt.png`)});
  await emit({id:'replacement',name:'Replacement',node_type:'Action',status:'INVALID',children:[]});assert(await page.locator('#tree-details').isHidden());
  // Saved graph paths remain effective after reload.
  await load('/fixtures/scene.json');await page.reload();await page.waitForFunction(()=>document.querySelector('#map-status').textContent.includes('/fixtures/scene.json'));
  assert.equal(await page.locator('#scene-graph-file').inputValue(),'/fixtures/scene.json');
  assert.equal(await page.evaluate(()=>uiMock.goals.length),0);assert.deepEqual(external,[]);assert.deepEqual(errors,[]);
  results.push({engine,yamlClassification:true,relativePaths:true,atomicReload:true,nodeDetails:true,popupLayouts:[1194,390],motion:false});
  await browser.close();console.log(`${engine}: focused scene and node-details checks passed`);
 }
 fs.writeFileSync(path.join(out,'results.json'),JSON.stringify(results,null,2));
})().catch(e=>{console.error(e);process.exit(1);});
