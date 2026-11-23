import {loadStaticMap} from './map_source.js';
import * as ROSLIB from './vendor/roslib.js';
import {projectTree, topologyKey} from './ui_state.js';
import {loadSceneDataset, validVisitPoses, sceneObjectCategory as classifySceneObject} from './scene_data.js?v=20261103a';
import {bindViewport, bindPicture, clampPicture} from './gestures.js?v=20261026e';

let ros = null;
let currentPage = ['#tree','#tasks'].includes(location.hash) ? location.hash.slice(1) : 'map';
let workflowCatalog = null, catalogBridgeUrl = null, executeWorkflowAction = null, activeActionKind = 'task';
let mapGestures = null, treeGestures = null;
let followupPromptId = null, replyInFlight = false;
let connectionGeneration = 0, requestGeneration = 0;
let reconnectTimer = null, reconnectAttempt = 0, intentionalDisconnect = true;
const HEARTBEAT_DELAY_MS = 10000, HEARTBEAT_UNKNOWN_MS = 30000;
let heartbeatAt = null, serverInstance = null, serverReady = false, serverBusy = false;
let bridgeOpenedAt = 0, availabilityPhase = 'disconnected', resynchronizing = false;
let visualsPaused = true, recoveringTelemetry = false, timelyHeartbeatCount = 0;
let operationStatus = 'Ready';
let taskAwaiting = false, observingRecoveredTask = false;
let goalBinding = null;
let followupInFlight = false;
const lightDefault = matchMedia('(pointer: coarse)').matches;
let lightDisplay = (localStorage.getItem('light-display') ?? String(lightDefault)) === 'true';
let treeDepth = Number(localStorage.getItem('tree-depth') || 2);
const expandedTree = new Set(), collapsedTree = new Set();
let treeTopology = '', treeVisibleKey = '', treeElements = new Map();
const visualJobs = new Map();
function queueVisual(key, callback) {
    visualJobs.set(key, callback);
}
setInterval(() => {
    if (document.hidden) return;
    const jobs = [...visualJobs.values()]; visualJobs.clear();
    requestAnimationFrame(() => jobs.forEach(job => job()));
}, 200);
function finishTask(label, {preserve = false} = {}) {
    taskStage = null;
    clearSceneHighlights();
    if (goalBinding?.id) goalBinding.socket.removeAllListeners?.(goalBinding.id);
    goalBinding = null;
    ++requestGeneration;
    taskRunning = false; taskAwaiting = false; followupActive = false; observingRecoveredTask = false;
    renderTaskStage();
    hideExecutionOverlay(); setActionBadge(label);
    clearCurrentRequest();
    if (!preserve && activeActionKind === 'task') elTxtCommand.value = '';
    activeActionKind = 'task';
    updateAvailability();
}
function canSubmitTask() {
    return Boolean(ros?.isConnected && serverReady && !serverBusy && !taskRunning &&
        ['ready', 'delayed'].includes(availabilityPhase));
}
function updateSubmission() { elBtnSendTask.disabled = !canSubmitTask(); updateWorkflowButtons(); }
function refreshPageSubscriptions() {
    Object.keys(subscriptionSpecs).forEach(refreshWidgetSubscription);
}
function pauseTelemetry() {
    if (visualsPaused) return;
    visualsPaused = true; recoveringTelemetry = true; timelyHeartbeatCount = 0;
    visualJobs.clear(); refreshPageSubscriptions();
}
function updateAvailability(now = performance.now()) {
    if (!ros?.isConnected) {
        availabilityPhase = intentionalDisconnect ? 'disconnected' : 'connecting';
    } else {
        const age = now - (heartbeatAt ?? bridgeOpenedAt);
        if (age >= HEARTBEAT_UNKNOWN_MS) availabilityPhase = 'unknown';
        else if (resynchronizing) availabilityPhase = 'resynchronizing';
        else if (heartbeatAt === null) availabilityPhase = 'waiting';
        else if (age >= HEARTBEAT_DELAY_MS) availabilityPhase = 'delayed';
        else availabilityPhase = serverReady ? 'ready' : 'waiting';
        if (age >= HEARTBEAT_DELAY_MS) pauseTelemetry();
    }
    const ribbonOperation = operationStatus.replace(/\s+running$/i,'');
    const activeStatus = /^(ready|running)$/i.test(ribbonOperation) ? 'Task' : ribbonOperation;
    const labels = {disconnected:'Bridge disconnected', connecting:'Connecting to bridge',
        waiting:'Waiting for Orchid', resynchronizing:'Resynchronizing',
        delayed:'Updates delayed', unknown:'Orchid status unknown', ready:serverBusy || taskRunning ? activeStatus : 'Ready'};
    elLblRobotStatus.textContent = labels[availabilityPhase];
    elLblRobotStatus.dataset.availability = availabilityPhase;
    updateSubmission();
    renderTaskStage();
}
function bridgeUnavailable() {
    clearSceneHighlights();
    // Losing transport does not establish the outcome of an accepted task.
    serverReady = false; heartbeatAt = null; resynchronizing = false;
    visualsPaused = true; recoveringTelemetry = false; timelyHeartbeatCount = 0;
    if (taskRunning) observingRecoveredTask = true;
    updateAvailability();
}
function handleHeartbeat(message) {
    let state; try { state = JSON.parse(message.data); } catch { return; }
    if (!state.session_id || typeof state.ready !== 'boolean' || typeof state.busy !== 'boolean') return;
    const now = performance.now(), changed = serverInstance && serverInstance !== state.session_id;
    if (changed) {
        finishTask('Server restarted', {preserve: true});
        executeTaskAction = null; setupActionAndService();
        timelyHeartbeatCount = 0;
    }
    const timely = heartbeatAt !== null && now - heartbeatAt >= 500 && now - heartbeatAt < HEARTBEAT_DELAY_MS;
    timelyHeartbeatCount = timely && !changed ? timelyHeartbeatCount + 1 : 1;
    reconnectAttempt = 0; serverInstance = state.session_id; heartbeatAt = now; resynchronizing = false;
    serverReady = state.ready; serverBusy = state.busy;
    if (visualsPaused && (!recoveringTelemetry || timelyHeartbeatCount >= 3)) {
        visualsPaused = false; recoveringTelemetry = false; refreshPageSubscriptions();
    }
    if (serverBusy && !taskRunning) {
        activeActionKind = state.task_kind === 'workflow' ? 'workflow' : 'task';
        taskRunning = true; taskAwaiting = false; observingRecoveredTask = true;
        taskStage = taskStage || 'Executing';
        setCurrentRequest('Server task', state.task_description || 'Task running');
        showExecutionOverlay('Running', 'Server reports an active task');
    } else if (!serverBusy && taskRunning && observingRecoveredTask) {
        finishTask('Task ended', {preserve: true});
    }
    updateAvailability(now);
}
setInterval(() => {
    const now = performance.now(); updateAvailability(now);
    if (!document.hidden && !intentionalDisconnect &&
        now - lastRosMessageAt >= HEARTBEAT_UNKNOWN_MS &&
        now - bridgeOpenedAt >= HEARTBEAT_UNKNOWN_MS && !reconnectTimer) {
        scheduleReconnect('no inbound ROS messages for 30 seconds');
    }
}, 1000);
document.addEventListener('visibilitychange', () => {
    if (!document.hidden) {
        resynchronizing = true;
        // Give the resumed transport a fresh opportunity to report state.
        bridgeOpenedAt = performance.now(); heartbeatAt = null;
        pauseTelemetry();
        if (!ros?.isConnected && !intentionalDisconnect && !reconnectTimer) openROSConnection();
        updateAvailability();
    }
    refreshPageSubscriptions();
});
// Command‑history for the task textarea
let commandHistory = [];
let historyIndex = -1;
let currentRobotId = 'hsrb_bot';

// ROS2 Action / Service clients (initialised on connect)
let executeTaskAction = null;
let followupPollTimer = null;
let followupActive = false;
let taskRunning = false;
let taskStage = null;
function renderTaskStage() {
    const card=document.getElementById('task-stage');
    if (!card) return;
    card.textContent=taskStage || '';
    card.classList.toggle('hidden',!taskRunning || !taskStage);
}
let currentRequestKind = 'none';
let currentRequestText = '';

// ROS Topics
let subTaskStatus = null;
let subGuiLogs = null;
let subGuiState = null;
let subCamera = null;
let subPose = null;
let subMap = null;
let subPlan = null;
let subBehaviorTree = null;
let subBattery = null;
let subSpeech = null;
let subscriptionSpecs = {};
let cameraCanvas = null;
let cameraCtx = null;
let lastRosMessageAt = 0;


// Speech Recognition setup (Web Speech API)
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
let recognition = null;
let speechTargetTextarea = null;
let speechFinalTranscript = '';

if (SpeechRecognition) {
    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.lang = 'en-US';
    recognition.interimResults = true;
    recognition.maxAlternatives = 1;
}

// 2D Map Configuration
let robotPose = { x: 0.0, y: 0.0, yaw: 0.0 };
let latestPlanPath = [];
let occupancyGrid = null;
let occupancyCacheKey = null, mapRefreshRequested = true;
let staticMapAbort=null,staticMapGeneration=0,mapLoadMessage='';
function setMapLoadMessage(message){mapLoadMessage=message;const el=document.getElementById('map-load-state');if(el){el.textContent=message;el.title=message;el.classList.toggle('hidden',!message);}}
let sceneHighlights={navigation_target:null,known_objects:[]};
function clearSceneHighlights(){sceneHighlights={navigation_target:null,known_objects:[]};drawMap();}
function mapSource(){return localStorage.getItem('map-source')==='topic'?'topic':'static';}
function staticMapPath(kind){return localStorage.getItem(`map-${kind}-path`) || (kind==='png'?'/map.png':'/map.yaml');}
async function refreshStaticMap(force=false){
    if(mapSource()!=='static')return;
    ensureMapCacheIdentity();
    if(!force && (hasValidOccupancyGrid() || staticMapAbort))return;
    staticMapAbort?.abort();const controller=new AbortController();staticMapAbort=controller;
    const generation=++staticMapGeneration;setMapLoadMessage('Loading static map…');drawMap();
    try {
        const result=await loadStaticMap(staticMapPath('png'),staticMapPath('yaml'),{signal:controller.signal});
        if(generation!==staticMapGeneration || mapSource()!=='static')return;
        occupancyGrid=result.grid;mapCanvasCache=result.canvas;mapCanvasCacheCtx=mapCanvasCache.getContext('2d');
        mapRefreshRequested=false;mapAutoFitPending=true;setMapLoadMessage('');drawMap();
    } catch(error){if(generation===staticMapGeneration && !controller.signal.aborted){setMapLoadMessage(`Static map: ${error.message}`);addSystemLog(mapLoadMessage);drawMap();}}
    finally {if(generation===staticMapGeneration)staticMapAbort=null;}
}
const rosTraffic = new Map();
// Small, bounded diagnostics: aggregate counts only, never task/prompt bodies.
window.orchidTransportStats = () => Object.fromEntries([...rosTraffic].map(([name, stats]) => [name, {...stats}]));
function recordRosTraffic(topic, msg) {
    if (!rosTraffic.has(topic) && rosTraffic.size >= 32) rosTraffic.delete(rosTraffic.keys().next().value);
    const stats = rosTraffic.get(topic) || {messages:0, estimatedJsonBytes:0};
    stats.messages++;
    if (msg?.data?.length !== undefined) {
        stats.estimatedJsonBytes += typeof msg.data === 'string' ? msg.data.length : msg.data.length * 4;
    } else if (msg?.poses) stats.estimatedJsonBytes += msg.poses.length * 200;
    else stats.estimatedJsonBytes += 256;
    rosTraffic.set(topic, stats);
}
function ensureMapCacheIdentity() {
    const key = mapSource()==='static'?JSON.stringify(['static',staticMapPath('png'),staticMapPath('yaml')]):JSON.stringify(['topic',elBridgeUrl.value.trim(), currentRobotId, mapTopic()]);
    if (key === occupancyCacheKey) return;
    staticMapAbort?.abort();staticMapAbort=null;++staticMapGeneration;setMapLoadMessage('');
    occupancyCacheKey = key; occupancyGrid = null; mapCanvasCache = null;
    mapCanvasCacheCtx = null; mapRefreshRequested = true; mapAutoFitPending = true;
    visualJobs.delete('mapGrid');
}
let mapCanvasCache = null;
let mapCanvasCacheCtx = null;
let mapAutoFitPending = true;
let mapZoomLocked = localStorage.getItem('map-zoom-lock') === 'true';
let treeZoomLocked = localStorage.getItem('tree-zoom-lock') === 'true';
let sceneGraph = null;
let sceneGraphFile = localStorage.getItem('scene-graph-file') || '/scene_graph_complete.json';
let sceneGraphLoadGeneration = 0, sceneGraphAbort = null;
let sceneClasses = {furniture:new Set(), structural:new Set()};
let selectedSceneObject = null;
function boundedSetting(key, fallback, min, max) {
    const value = Number(localStorage.getItem(key) ?? fallback);
    return Number.isFinite(value) ? Math.max(min,Math.min(max,value)) : fallback;
}
let sceneOutlineWidth = boundedSetting('scene-outline-width',3,1,10);
let sceneNodeRadius = boundedSetting('scene-node-radius',6,3,16);
let previewDuration = boundedSetting('preview-duration',5,1,30);
let cameraWindowScale = boundedSetting('camera-window-scale',1,.75,2);
function sceneRepresentationDefault(category, kind) {
    const saved=localStorage.getItem(`scene-show-${category}${kind}`);
    if(saved!==null)return saved==='true';
    const categoryVisible=category==='structural'?localStorage.getItem('scene-show-structural')==='true':localStorage.getItem(`scene-show-${category}`)!=='false';
    return categoryVisible && localStorage.getItem(`scene-show-${kind.toLowerCase()}`)!=='false';
}
let sceneOverlay = {
    furnitureBoxes: sceneRepresentationDefault('furniture','Boxes'),
    furnitureNodes: sceneRepresentationDefault('furniture','Nodes'),
    movableBoxes: sceneRepresentationDefault('movable','Boxes'),
    movableNodes: sceneRepresentationDefault('movable','Nodes'),
    structuralBoxes: sceneRepresentationDefault('structural','Boxes'),
    structuralNodes: sceneRepresentationDefault('structural','Nodes'),
    rooms: localStorage.getItem('scene-show-rooms') !== 'false',
    labels: localStorage.getItem('scene-show-labels') !== 'false',
    visitPoses: localStorage.getItem('scene-show-visitPoses') === 'true',
};
let lastBatteryAt = 0;
let batteryStaleTimer = null;
const batteryStaleMs = 5000;

// Interactive Map Drag/Pan State
let mapIsDragging = false;
let mapDragMoved = false;
let mapDragStart = { x: 0, y: 0 };
let mapDragOffset = { x: 0, y: 0 };

// Interactive Behavior Tree Drag/Zoom State
let treePanX = 0;
let treePanY = 0;
let treeZoom = 1.0;
let treeIsDragging = false;
let treeDragStart = { x: 0, y: 0 };
let treeDragOffset = { x: 0, y: 0 };
let selectedTreeNodeId = null;
let latestTreeData = null;
let latestActiveNodeName = '';
let speechMuted = localStorage.getItem('speech-muted') === 'true';
let speechInterrupt = localStorage.getItem('speech-interrupt') === 'true';
let speechRate = Number(localStorage.getItem('speech-rate') || '1');
let speechPitch = Number(localStorage.getItem('speech-pitch') || '1');
let speechVoiceName = localStorage.getItem('speech-voice') || '';

// UI Elements Cache
const elConnectBtn = document.getElementById('btn-connect');
const elConnStatus = document.getElementById('conn-status');
const elRobotId = document.getElementById('robot-id');
const elBridgeUrl = document.getElementById('rosbridge-url');
const elSettingsSidebar = document.getElementById('settings-sidebar');
const elBtnThemeToggle = document.getElementById('btn-theme-toggle');
const elBtnToggleSettings = document.getElementById('btn-toggle-settings');
const elBtnCloseSidebar = document.getElementById('btn-close-sidebar');

const elTxtCommand = document.getElementById('txt-command');
const elBtnMic = document.getElementById('btn-mic');
const elBtnSendTask = document.getElementById('btn-send-task');
const elOverlayExecuting = document.getElementById('cmd-executing-overlay');
const elOverlayTitle = document.getElementById('overlay-title');
const elOverlayFeedback = document.getElementById('overlay-feedback');
const elExecutionFeedbackPanel = document.getElementById('execution-feedback-panel');
const elFollowupInlinePanel = document.getElementById('followup-inline-panel');
const elBtnCancelTask = document.getElementById('btn-cancel-task');
const elSpeechStatus = document.getElementById('speech-status');
const elActionStatusBadge = document.getElementById('action-status-badge');
const elCurrentRequest = document.getElementById('current-request');
const elCurrentRequestText = document.getElementById('current-request-text');

const elMapCanvas = document.getElementById('map-canvas');
const elMapScale = document.getElementById('map-scale');
const elMapOriginX = document.getElementById('map-origin-x');
const elMapOriginY = document.getElementById('map-origin-y');
const elMapTopic = document.getElementById('map-topic');
const elRobotPoseTopic = document.getElementById('robot-pose-topic');
const elPlanTopic = document.getElementById('plan-topic');
const elMapRotation = document.getElementById('map-rotation');
const elBtnMapZoomLock = document.getElementById('btn-map-zoom-lock');
const elBtnSaveMapSettings = document.getElementById('btn-save-map-settings');
const elBtnResetMap = document.getElementById('btn-reset-map');
const elBtnRefreshMap = document.getElementById('btn-refresh-map');

const elCamImage = document.getElementById('cam-image');
const elCamFallback = document.getElementById('cam-fallback');
const elCamStatus = document.getElementById('cam-status');
const elCameraTopic = document.getElementById('camera-topic');
const elCameraMessageType = document.getElementById('camera-message-type');
const elCameraThrottleMs = document.getElementById('camera-throttle-ms');
const elBtnSaveCameraSettings = document.getElementById('btn-save-camera-settings');

const elTreeSvg = document.getElementById('tree-svg');
const elTreeFallback = document.getElementById('tree-fallback');
const elBtnResetTree = document.getElementById('btn-reset-tree');
const elTreeDetails = document.getElementById('tree-details');
const elBtnTreeZoomLock = document.getElementById('btn-tree-zoom-lock');
const elSceneGraphFile = document.getElementById('scene-graph-file');
const elBtnLoadSceneGraph = document.getElementById('btn-load-scene-graph');
const elSceneInfo = document.getElementById('scene-info');
const elSceneInfoContent = document.getElementById('scene-info-content');
const elBtnCloseSceneInfo = document.getElementById('btn-close-scene-info');
const elMapStatus = document.getElementById('map-status');
const sceneToggleElements = {
    furnitureBoxes: document.getElementById('scene-show-furniture-boxes'),
    furnitureNodes: document.getElementById('scene-show-furniture-nodes'),
    movableBoxes: document.getElementById('scene-show-movable-boxes'),
    movableNodes: document.getElementById('scene-show-movable-nodes'),
    structuralBoxes: document.getElementById('scene-show-structural-boxes'),
    structuralNodes: document.getElementById('scene-show-structural-nodes'),
    rooms: document.getElementById('scene-show-rooms'),
    labels: document.getElementById('scene-show-labels'),
    visitPoses: document.getElementById('scene-show-visit-poses'),
};

const elBatteryRing = document.getElementById('battery-ring');
const elLblBattery = document.getElementById('lbl-battery');
const elLblBatteryStatus = document.getElementById('lbl-battery-status');
const elLblRobotStatus = document.getElementById('lbl-robot-status');
const elLblTaskTime = document.getElementById('lbl-task-time');
const elLblCurrentActionTime = document.getElementById('lbl-current-action-time');
const elLblCurrentActionName = document.getElementById('lbl-current-action-name');
const elBtnToggleSpeech = document.getElementById('btn-toggle-speech');
const elSpeechRate = document.getElementById('speech-rate');
const elSpeechRateValue = document.getElementById('speech-rate-value');
const elSpeechPitch = document.getElementById('speech-pitch');
const elSpeechPitchValue = document.getElementById('speech-pitch-value');
const elSpeechVoice = document.getElementById('speech-voice');

const elLogConsole = document.getElementById('log-console');
const elBtnClearLogs = document.getElementById('btn-clear-logs');

const elFollowupPrompt = document.getElementById('followup-prompt');
const elTxtFollowupAnswer = document.getElementById('txt-followup-answer');
const elBtnFollowupMic = document.getElementById('btn-followup-mic');
const elBtnCancelFollowup = document.getElementById('btn-cancel-followup');
const elBtnSendFollowup = document.getElementById('btn-send-followup');
const elFollowupSpeechStatus = document.getElementById('followup-speech-status');

// -------------------------------------------------------------
// 1. Initial Load & Local Storage Layout Cache
// -------------------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
    document.body.classList.toggle('light-display', lightDisplay);
    const lightToggle = document.getElementById('light-display');
    lightToggle.checked = lightDisplay;
    lightToggle.addEventListener('change', () => {
        lightDisplay = lightToggle.checked; localStorage.setItem('light-display', lightDisplay);
        document.body.classList.toggle('light-display', lightDisplay);
    });
    document.getElementById('tree-depth').value = String(treeDepth);
    document.getElementById('tree-depth').addEventListener('change', event => {
        treeDepth = Number(event.target.value); localStorage.setItem('tree-depth', treeDepth);
        treeVisibleKey = ''; if (latestTreeData) renderBehaviorTree(latestTreeData);
    });
    document.getElementById('ribbon-task').addEventListener('click', () => { if (currentRequestText) alert(currentRequestText); });
    initPages();
    elBridgeUrl.value = localStorage.getItem('bridge-url') || `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.hostname}:9090`;
    elRobotId.value = localStorage.getItem('robot-id') || elRobotId.value;
    currentRobotId = elRobotId.value.trim(); loadWorkflowCache();
    updateSubmission(); updateTreeDetails(null);
    elTreeSvg.addEventListener('click', event => {
        if (!event.target.closest('.tree-node')) { selectedTreeNodeId = null; updateTreeDetails(null); }
    });
    initLayoutToggles();
    initTheme();
    initMapSettings();
    initSceneSettings();
    initCameraSettings();
    initSpeechSettings();
    initEventListeners();
    initCanvas();
    initMapInteraction();
    initTreeInteraction();
    addSystemLog("HRI dashboard loaded. Click 'Connect' to initialize ROS2 communication.");
});

// Setup sidebar widgets checkboxes
function initLayoutToggles() {
    const camera = document.getElementById('chk-cam');
    camera.checked = localStorage.getItem('widget-cam-visible') !== 'false';
    camera.addEventListener('change', () => {
        localStorage.setItem('widget-cam-visible', camera.checked); refreshWidgetSubscription('cam');
        document.getElementById('widget-cam').hidden = !camera.checked;
        if (!camera.checked) releaseCamera();
    });
    const logs = document.getElementById('chk-logs'); logs.checked = false;
    logs.addEventListener('change', () => {
        document.body.classList.toggle('logs-open', logs.checked);
        refreshWidgetSubscription('logs');
    });
}

function initCameraSettings() {
    if (!elCameraTopic || !elCameraMessageType || !elCameraThrottleMs) return;
    elCameraTopic.value = cameraTopic();
    elCameraMessageType.value = cameraMessageType();
    elCameraThrottleMs.value = String(cameraThrottleMs());
    elCamFallback.textContent = `Awaiting image topic ${cameraTopic()}`;
}

function initMapSettings() {
    const mapFields = mapSettingFields();
    mapFields.forEach(({ element, key }) => {
        if (!element) return;
        const stored = localStorage.getItem(key);
        if (stored === null) return;
        if (element.type === 'checkbox') element.checked = stored === 'true';
        else element.value = stored;
    });
    updateMapZoomLockButton();
}

function saveMapSettings() {
    mapSettingFields().forEach(({ element, key }) => {
        if (!element) return;
        const value = element.type === 'checkbox' ? element.checked : element.value;
        localStorage.setItem(key, value);
    });
}

function saveMapView() {
    for(const element of [elMapScale,elMapOriginX,elMapOriginY,elMapRotation])localStorage.setItem(element.id,element.value);
}

function mapSettingFields() {
    return [
        { element: document.getElementById('map-source'), key: 'map-source' },
        { element: document.getElementById('map-png-path'), key: 'map-png-path' },
        { element: document.getElementById('map-yaml-path'), key: 'map-yaml-path' },
        { element: elMapTopic, key: 'map-topic' },
        { element: elRobotPoseTopic, key: 'robot-pose-topic' },
        { element: elPlanTopic, key: 'plan-topic' },
        { element: elMapScale, key: 'map-scale' },
        { element: elMapOriginX, key: 'map-origin-x' },
        { element: elMapOriginY, key: 'map-origin-y' },
        { element: elMapRotation, key: 'map-rotation' },
    ];
}

function initSceneSettings() {
    for (const [id,value,min,max,apply] of [
        ['scene-outline-width',sceneOutlineWidth,1,10,v=>{sceneOutlineWidth=v;drawMap();}],
        ['scene-node-radius',sceneNodeRadius,3,16,v=>{sceneNodeRadius=v;drawMap();}],
        ['preview-duration',previewDuration,1,30,v=>{previewDuration=v;}],
        ['camera-window-scale',cameraWindowScale,.75,2,v=>{cameraWindowScale=v;sizeCamera();}],
    ]) {
        const input=document.getElementById(id); input.value=String(value);
        input.addEventListener('input',()=>{const value=Number(input.value);if(!Number.isFinite(value)||value<min||value>max)return;
            localStorage.setItem(id,String(value));apply(value);});
    }
    if (elSceneGraphFile) elSceneGraphFile.value = sceneGraphFile;
    Object.entries(sceneToggleElements).forEach(([key, element]) => {
        if (element) element.checked = sceneOverlay[key];
    });
    updateTreeZoomLockButton();
    loadSceneGraph(sceneGraphFile);
}

function initSpeechSettings() {
    const interrupt = document.getElementById('speech-interrupt');
    interrupt.checked = speechInterrupt;
    interrupt.addEventListener('change', () => {
        speechInterrupt = interrupt.checked;
        localStorage.setItem('speech-interrupt', String(speechInterrupt));
    });
    if (elSpeechRate) elSpeechRate.value = String(speechRate);
    if (elSpeechPitch) elSpeechPitch.value = String(speechPitch);
    updateSpeechSettingLabels();
    populateSpeechVoices();
    if (window.speechSynthesis) {
        window.speechSynthesis.addEventListener('voiceschanged', populateSpeechVoices);
    }
}

function updateSpeechSettingLabels() {
    if (elSpeechRateValue) elSpeechRateValue.textContent = `${speechRate.toFixed(2)}x`;
    if (elSpeechPitchValue) elSpeechPitchValue.textContent = speechPitch.toFixed(2);
}

function populateSpeechVoices() {
    if (!elSpeechVoice || !window.speechSynthesis) return;
    const voices = window.speechSynthesis.getVoices();
    const selected = speechVoiceName;
    elSpeechVoice.innerHTML = '<option value="">Browser default</option>';
    voices
        .slice()
        .sort((a, b) => `${a.lang} ${a.name}`.localeCompare(`${b.lang} ${b.name}`))
        .forEach((voice) => {
            const option = document.createElement('option');
            option.value = voice.name;
            option.textContent = `${voice.name} (${voice.lang})`;
            option.selected = voice.name === selected;
            elSpeechVoice.appendChild(option);
        });
}

// -------------------------------------------------------------
// 2. DOM Event Listeners
// -------------------------------------------------------------
function initEventListeners() {
    // Connect button
    elConnectBtn.addEventListener('click', connectToROS);

    // Theme toggle
    elBtnThemeToggle.addEventListener('click', () => {
        document.body.classList.toggle('light-theme');
        const isLight = document.body.classList.contains('light-theme');
        localStorage.setItem('theme', isLight ? 'light' : 'dark');
        const icon = elBtnThemeToggle.querySelector('i');
        if (isLight) {
            icon.className = 'fa-solid fa-sun';
            addSystemLog("Theme switched to Light Mode.");
        } else {
            icon.className = 'fa-solid fa-moon';
            addSystemLog("Theme switched to Dark Mode.");
        }
        drawMap();
    });

    // Sidebar toggles
    elBtnToggleSettings.addEventListener('click', () => elSettingsSidebar.classList.toggle('open'));
    elBtnCloseSidebar.addEventListener('click', () => elSettingsSidebar.classList.remove('open'));

    // Command submission & actions
    elBtnSendTask.addEventListener('click', () => submitTaskAction());
    document.getElementById('followup-choices').addEventListener('change', event => { if (event.target.value) elTxtFollowupAnswer.value = event.target.value; });
    elBtnCancelTask.addEventListener('click', cancelTaskAction);

    // Speech commanding
    if (recognition) {
        elBtnMic.addEventListener('click', () => startSpeechRecognition(elTxtCommand, elSpeechStatus));
        elBtnFollowupMic.addEventListener('click', () => startSpeechRecognition(elTxtFollowupAnswer, elFollowupSpeechStatus));

        recognition.onstart = () => {
            elBtnMic.disabled = true;
            elBtnFollowupMic.disabled = true;
            elSpeechStatus.textContent = 'Listening... Speak now.';
            elFollowupSpeechStatus.textContent = 'Listening... Speak now.';
        };

        recognition.onresult = (event) => {
            // Build transcript from ALL results so multi-phrase speech is captured,
            // not just event.results[0].
            let finalText = '';
            let interim = '';
            for (let i = 0; i < event.results.length; i++) {
                const r = event.results[i][0].transcript;
                if (event.results[i].isFinal) {
                    finalText += r;
                } else {
                    interim += r;
                }
            }
            if (speechTargetTextarea) {
                speechTargetTextarea.value = finalText + interim;
            }
        };

        recognition.onerror = (event) => {
            let msg = 'Speech recognition error';
            if (event.error === 'no-speech') msg = 'No speech detected.';
            else if (event.error === 'audio-capture') msg = 'No microphone found.';
            else if (event.error === 'not-allowed') msg = 'Microphone blocked.';
            else if (event.error === 'network') msg = 'Network error.';
            addSystemLog(`[Speech] ${msg}`);
        };

        recognition.onend = () => {
            elSpeechStatus.classList.add('hidden');
            elFollowupSpeechStatus.classList.add('hidden');
            elBtnMic.disabled = false;
            elBtnFollowupMic.disabled = false;
        };
    } else {
        elBtnMic.disabled = true;
        elBtnFollowupMic.disabled = true;
        elBtnMic.title = "Speech recognition not supported in this browser";
        elBtnFollowupMic.title = "Speech recognition not supported in this browser";
    }

    // Map configuration apply
    elBtnSaveMapSettings.addEventListener('click', () => {
        const previousMapTopic = mapTopic();
        saveMapSettings();
        mapAutoFitPending = !hasValidOccupancyGrid() || mapTopic() !== previousMapTopic;
        if (subscriptionSpecs.mapGrid) {
            ensureMapCacheIdentity();
            ['mapGrid', 'map', 'plan'].forEach((widget) => {
                unsubscribeWidget(widget);
                if (widget === 'mapGrid') subscriptionSpecs.mapGrid.topic = mapTopic();
                if (widget === 'map') subscriptionSpecs.map.topic = robotPoseTopic();
                if (widget === 'plan') subscriptionSpecs.plan.topic = planTopic();
                refreshWidgetSubscription(widget);
            });
        }
        ensureMapCacheIdentity();
        refreshStaticMap();drawMap();
        addSystemLog("Map settings applied.");
    });

    // Command history: recall previous tasks with the Up/Down arrows
    elTxtCommand.addEventListener('keydown', (e) => {
        if (commandHistory.length === 0) return;
        if (e.key === 'ArrowUp') {
            e.preventDefault();
            historyIndex = historyIndex <= 0 ? commandHistory.length - 1 : historyIndex - 1;
            elTxtCommand.value = commandHistory[historyIndex];
        } else if (e.key === 'ArrowDown') {
            e.preventDefault();
            if (historyIndex >= commandHistory.length - 1) {
                historyIndex = -1;
                elTxtCommand.value = '';
            } else {
                historyIndex = historyIndex + 1;
                elTxtCommand.value = commandHistory[historyIndex];
            }
        }
    });
    elBtnResetMap.addEventListener('click', () => {
        latestPlanPath = [];
        mapAutoFitPending = true;
        drawMap();
        saveMapView();
        addSystemLog("Map view fit reset and plan overlay cleared.");
    });
    if (elBtnRefreshMap) {
        elBtnRefreshMap.addEventListener('click', () => {
            if(mapSource()==='static'){refreshStaticMap(true);return;}
            mapRefreshRequested = true;
            refreshWidgetSubscription('mapGrid');
            addSystemLog('Requested fresh occupancy map.');
        });
    }

    if (elBtnMapZoomLock) {
        elBtnMapZoomLock.addEventListener('click', () => {
            mapZoomLocked = !mapZoomLocked;
            localStorage.setItem('map-zoom-lock', String(mapZoomLocked));
            updateMapZoomLockButton();
        });
    }

    if (elBtnTreeZoomLock) {
        elBtnTreeZoomLock.addEventListener('click', () => {
            treeZoomLocked = !treeZoomLocked;
            localStorage.setItem('tree-zoom-lock', String(treeZoomLocked));
            updateTreeZoomLockButton();
        });
    }

    if (elBtnLoadSceneGraph) {
        elBtnLoadSceneGraph.addEventListener('click', () => {
            sceneGraphFile = elSceneGraphFile.value.trim() || '/scene_graph_complete.json';
            localStorage.setItem('scene-graph-file', sceneGraphFile);
            loadSceneGraph(sceneGraphFile);
        });
    }
    Object.entries(sceneToggleElements).forEach(([key, element]) => {
        if (!element) return;
        element.addEventListener('change', () => {
            sceneOverlay[key] = element.checked;
            localStorage.setItem(`scene-show-${key}`, String(element.checked));
            drawMap();
        });
    });
    if (elBtnCloseSceneInfo) {
        elBtnCloseSceneInfo.addEventListener('click', () => clearSceneSelection());
    }

    if (elBtnToggleSpeech) {
        updateSpeechMuteButton();
        elBtnToggleSpeech.addEventListener('click', () => {
            speechMuted = !speechMuted;
            localStorage.setItem('speech-muted', String(speechMuted));
            if (speechMuted && window.speechSynthesis) {
                window.speechSynthesis.cancel();
            }
            updateSpeechMuteButton();
        });
    }

    if (elSpeechRate) {
        elSpeechRate.addEventListener('input', () => {
            speechRate = Number(elSpeechRate.value) || 1;
            localStorage.setItem('speech-rate', String(speechRate));
            updateSpeechSettingLabels();
        });
    }
    if (elSpeechPitch) {
        elSpeechPitch.addEventListener('input', () => {
            speechPitch = Number(elSpeechPitch.value) || 1;
            localStorage.setItem('speech-pitch', String(speechPitch));
            updateSpeechSettingLabels();
        });
    }
    if (elSpeechVoice) {
        elSpeechVoice.addEventListener('change', () => {
            speechVoiceName = elSpeechVoice.value;
            localStorage.setItem('speech-voice', speechVoiceName);
        });
    }

    if (elBtnSaveCameraSettings) {
        elBtnSaveCameraSettings.addEventListener('click', () => {
            localStorage.setItem('camera-topic', elCameraTopic.value.trim() || '/remote/head_rgbd_sensor/rgb/image_rect_color');
            localStorage.setItem('camera-message-type', elCameraMessageType.value.trim() || 'sensor_msgs/msg/Image');
            localStorage.setItem('camera-throttle-ms', elCameraThrottleMs.value.trim() || '500');
            if (subscriptionSpecs.cam) {
                unsubscribeWidget('cam');
                subscriptionSpecs.cam.topic = cameraTopic();
                subscriptionSpecs.cam.messageType = cameraMessageType();
                subscriptionSpecs.cam.options = { throttle_rate: cameraThrottleMs(), queue_length: 1 };
                refreshWidgetSubscription('cam');
            }
            addSystemLog("Camera subscription settings applied.");
        });
    }

    // Clear logs
    elBtnClearLogs.addEventListener('click', () => {
        elLogConsole.innerHTML = "";
        addSystemLog("Console logs cleared.");
    });

    // Log console filter buttons
    document.querySelectorAll('.log-filters button[data-filter]').forEach(btn => {
        btn.addEventListener('click', (e) => {
            document.querySelectorAll('.log-filters button[data-filter]').forEach(b => b.classList.remove('active'));
            e.target.classList.add('active');

            const filter = e.target.getAttribute('data-filter');
            const lines = elLogConsole.querySelectorAll('.log-line');
            lines.forEach(line => {
                if (filter === 'all') {
                    line.style.display = 'block';
                } else if (line.classList.contains(filter) || line.classList.contains('system')) {
                    line.style.display = 'block';
                } else {
                    line.style.display = 'none';
                }
            });
        });
    });

    // Follow-up send button
    elBtnCancelFollowup.addEventListener('click', cancelTaskAction);
    elBtnSendFollowup.addEventListener('click', submitFollowupResponse);

    // Behavior Tree reload trigger
    elBtnResetTree.addEventListener('click', () => {
        treePanX = 0;
        treePanY = 0;
        treeZoom = 1.0; treeVisibleKey = '';
        expandedTree.clear(); collapsedTree.clear(); selectedTreeNodeId = null;
        if (latestTreeData) {
            renderBehaviorTree(latestTreeData);
            elTreeFallback.classList.add('hidden');
        } else {
            elTreeSvg.innerHTML = "";
            elTreeFallback.classList.remove('hidden');
        }
        addSystemLog("Behavior Tree view reset.");
    });

    // Behavior Tree legend toggle
}

// -------------------------------------------------------------
// 3. ROS2 Connection Handling
// -------------------------------------------------------------
function setConnectButton(connected) {
    const label = connected ? 'Disconnect' : 'Connect';
    elConnectBtn.querySelector('.connect-label').textContent = label;
    elConnectBtn.title = label; elConnectBtn.setAttribute('aria-label', label);
}
function connectToROS() {
    if (!intentionalDisconnect) {
        intentionalDisconnect = true; clearTimeout(reconnectTimer); reconnectTimer = null;
        ++connectionGeneration;
        cleanupSubscriptions(); ros?.close(); ros = null;
        executeTaskAction = null; executeWorkflowAction = null; clearInterval(followupPollTimer); followupPollTimer = null;
        bridgeUnavailable(); updateConnectionStatus('disconnected');
        setConnectButton(false); return;
    }
    intentionalDisconnect = false; reconnectAttempt = 0;
    currentRobotId = elRobotId.value.trim();
    localStorage.setItem('robot-id', currentRobotId);
    loadWorkflowCache();
    localStorage.setItem('bridge-url', elBridgeUrl.value.trim());
    openROSConnection();
}
function scheduleReconnect(reason = 'socket closed') {
    if (intentionalDisconnect || reconnectTimer) return;
    console.info('[ROS reconnect]', {reason, inboundAgeMs: Math.round(performance.now() - lastRosMessageAt),
        heartbeatAgeMs: heartbeatAt === null ? null : Math.round(performance.now() - heartbeatAt),
        traffic: window.orchidTransportStats()});
    // Invalidate old callbacks before closing, so close cannot schedule a second loop.
    ++connectionGeneration;
    cleanupSubscriptions(); ros?.close(); ros = null; executeTaskAction = null; executeWorkflowAction = null;
    clearInterval(followupPollTimer); followupPollTimer = null;
    bridgeUnavailable(); updateConnectionStatus('disconnected');
    const delay = [500, 1000, 2000, 5000][Math.min(reconnectAttempt++, 3)];
    reconnectTimer = setTimeout(openROSConnection, delay);
}
function openROSConnection() {
    ensureMapCacheIdentity();
    if (catalogBridgeUrl !== elBridgeUrl.value.trim()) loadWorkflowCache();
    clearTimeout(reconnectTimer); reconnectTimer = null;
    cleanupSubscriptions();
    const generation = ++connectionGeneration;
    ros?.close();
    const socket = new ROSLIB.Ros({url: elBridgeUrl.value.trim()}); ros = socket;
    updateConnectionStatus('connecting'); setConnectButton(true);
    serverReady = false; heartbeatAt = null; bridgeOpenedAt = performance.now();
    lastRosMessageAt = bridgeOpenedAt; visualsPaused = true; recoveringTelemetry = false;
    resynchronizing = false; timelyHeartbeatCount = 0; updateAvailability();
    socket.on('connection', () => {
        if (generation !== connectionGeneration) return;
        bridgeOpenedAt = performance.now(); lastRosMessageAt = bridgeOpenedAt; updateConnectionStatus('connected');
        addSystemLog(`Bridge connected; waiting for ${currentRobotId}`);
        setupSubscriptions(); setupActionAndService(); updateAvailability();
    });
    socket.on('error', () => {
        if (generation !== connectionGeneration) return;
        updateConnectionStatus('degraded');
        if (!socket.isConnected) socket.close();
    });
    socket.on('close', () => {
        if (generation !== connectionGeneration) return;
        scheduleReconnect();
    });
}

function updateConnectionStatus(status) {
    elConnStatus.className = `status-badge ${status}`;
    elConnStatus.setAttribute('role', 'status');
    const dot = elConnStatus.querySelector('.badge-dot');
    const text = elConnStatus.querySelector('.badge-text');

    if (status === 'connected') {
        text.textContent = 'Connected';
    } else if (status === 'connecting') {
        text.textContent = 'Connecting...';
    } else if (status === 'degraded') {
        text.textContent = 'Connection Warning';
    } else {
        text.textContent = 'Disconnected';
    }
    elConnStatus.title = text.textContent;
    elConnStatus.setAttribute('aria-label', text.textContent);
}

// -------------------------------------------------------------
// 4. ROS2 Subscriptions (telemetry, camera, map, behavior tree, logs)
// -------------------------------------------------------------
function setupSubscriptions() {
    const placeholder = document.querySelector('.tree-placeholder-line');
    document.getElementById('tree-placeholder-topic').textContent = `/${currentRobotId}/behavior_tree`;
    placeholder.title = placeholder.getAttribute('aria-label').replace(/topic .*/, `topic /${currentRobotId}/behavior_tree`);
    placeholder.setAttribute('aria-label', placeholder.title);
    subscriptionSpecs = buildSubscriptionSpecs();
    Object.keys(subscriptionSpecs).forEach(refreshWidgetSubscription);
}

function cleanupSubscriptions() {
    Object.keys(subscriptionSpecs).forEach((widget) => unsubscribeWidget(widget));
    releaseCamera();
    subTaskStatus = null;
    subGuiLogs = null;
    subGuiState = null;
    subCamera = null;
    subPose = null;
    subMap = null;
    subPlan = null;
    subBehaviorTree = null;
    subBattery = null;
    subSpeech = null;
    if (batteryStaleTimer) {
        clearTimeout(batteryStaleTimer);
        batteryStaleTimer = null;
    }
    lastBatteryAt = 0;
    updateBatteryUnavailable();
}

function buildSubscriptionSpecs() {
    return {
        catalog: {assign: () => {}, topic: `/${currentRobotId}/workflow_catalog`, messageType: 'std_msgs/String', options: {queue_length:1}, callback: receiveWorkflowCatalog},
        heartbeat: {assign: () => {}, topic: `/${currentRobotId}/session_heartbeat`, messageType: 'std_msgs/String', options: {queue_length: 1}, callback: handleHeartbeat},
        ctrl: {
            assign: (topic) => { subTaskStatus = topic; },
            topic: `/${currentRobotId}/task_status`,
            messageType: 'std_msgs/String',
            options: { queue_length: 1, queue_size: 1 },
            callback: (msg) => updateTaskStatusUI(msg.data)
        },
        logs: {
            assign: (topic) => { subGuiLogs = topic; },
            topic: `/${currentRobotId}/gui_logs`,
            messageType: 'std_msgs/String',
            options: { queue_length: 1, queue_size: 1 },
            callback: (msg) => {
                try {
                    const data = JSON.parse(msg.data);
                    appendLog(data.level || 'info', data.message || msg.data, data.timestamp);
                } catch (e) {
                    appendLog('info', msg.data);
                }
            }
        },
        diag: {
            assign: (topic) => { subGuiState = topic; },
            topic: `/${currentRobotId}/gui_state`,
            messageType: 'std_msgs/String',
            options: { queue_length: 1, queue_size: 1 },
            callback: (msg) => {
                try {
                    updateDiagnostics(JSON.parse(msg.data));
                } catch (e) {
                    console.error("Error parsing GUI state JSON:", e);
                }
            }
        },
        cam: {
            assign: (topic) => { subCamera = topic; },
            topic: cameraTopic(),
            messageType: cameraMessageType(),
            options: { throttle_rate: cameraThrottleMs(), queue_length: 1, queue_size: 1 },
            callback: handleCameraMessage
        },
        previewSegmentation: {
            assign: () => {}, topic: `/${currentRobotId}/visualization/segmentation/compressed`,
            messageType: 'sensor_msgs/msg/CompressedImage', options: {queue_length:1,queue_size:1},
            callback: msg => handlePreviewMessage(msg,'segmentation'),
        },
        previewGrasp: {
            assign: () => {}, topic: `/${currentRobotId}/visualization/grasp/compressed`,
            messageType: 'sensor_msgs/msg/CompressedImage', options: {queue_length:1,queue_size:1},
            callback: msg => handlePreviewMessage(msg,'grasp'),
        },
        mapGrid: {
            assign: (topic) => { subMap = topic; },
            topic: mapTopic(),
            messageType: 'nav_msgs/msg/OccupancyGrid',
            options: { throttle_rate: 2000, queue_length: 1, queue_size: 1 },
            callback: handleMapMessage
        },
        map: {
            assign: (topic) => { subPose = topic; },
            topic: robotPoseTopic(),
            messageType: 'geometry_msgs/msg/PoseStamped',
            options: { throttle_rate: 100, queue_length: 1, queue_size: 1 },
            callback: handlePoseMessage
        },
        plan: {
            assign: (topic) => { subPlan = topic; },
            topic: planTopic(),
            messageType: 'nav_msgs/msg/Path',
            options: { throttle_rate: 1000, queue_length: 1, queue_size: 1 },
            callback: handlePlanMessage
        },
        tree: {
            assign: (topic) => { subBehaviorTree = topic; },
            topic: `/${currentRobotId}/behavior_tree`,
            messageType: 'std_msgs/String',
            options: { throttle_rate: 200, queue_length: 1, queue_size: 1 },
            callback: (msg) => {
                try {
                    const treeData = JSON.parse(msg.data);
                    queueVisual('tree', () => renderBehaviorTree(treeData));
                    elTreeFallback.classList.add('hidden');
                } catch (e) {
                    console.error("Error parsing behavior tree JSON:", e);
                }
            }
        },
        battery: {
            assign: (topic) => { subBattery = topic; },
            topic: `/battery_state`,
            messageType: 'sensor_msgs/msg/BatteryState',
            options: { queue_length: 1, queue_size: 1 },
            callback: handleBatteryMessage
        },
        speech: {
            assign: (topic) => { subSpeech = topic; },
            topic: `/${currentRobotId}/ui_speech`,
            messageType: 'std_msgs/String',
            options: { queue_length: 1, queue_size: 1 },
            callback: (msg) => speakUiText(msg.data || '')
        }
    };
}

function refreshWidgetSubscription(widget) {
    if (!ros || !ros.isConnected || !subscriptionSpecs[widget]) return;
    if (isWidgetVisible(widget)) {
        subscribeWidget(widget);
    } else {
        unsubscribeWidget(widget);
    }
    if (widget === 'cam') ['previewSegmentation','previewGrasp'].forEach(refreshWidgetSubscription);
}

function isWidgetVisible(widget) {
    if (widget.startsWith('preview')) return isWidgetVisible('cam');
    if (['ctrl', 'heartbeat', 'diag', 'battery', 'catalog'].includes(widget)) return true;
    if (document.hidden || visualsPaused) return false;
    if (widget === 'logs') return document.getElementById('chk-logs').checked;
    if (widget === 'tree') return currentPage === 'tree';
    if (widget === 'cam') return currentPage === 'map' && document.getElementById('chk-cam').checked;
    if (widget === 'mapGrid') return mapSource()==='topic' && currentPage === 'map' && (mapRefreshRequested || !hasValidOccupancyGrid());
    if (['map', 'mapGrid', 'plan'].includes(widget)) return currentPage === 'map';
    return true;
}

function subscribeWidget(widget) {
    const spec = subscriptionSpecs[widget];
    if (!spec || spec.instance) return;
    const generation = connectionGeneration;
    const topic = new ROSLIB.Topic({
        ros: ros,
        name: spec.topic,
        messageType: spec.messageType,
        ...(spec.options || {})
    });
    if (widget === 'cam') elCamStatus.textContent = 'Awaiting image';
    topic.subscribe((msg) => {
        if (generation !== connectionGeneration || spec.instance !== topic) return;
        recordRosTraffic(spec.topic, msg);
        lastRosMessageAt = performance.now();
        if (elConnStatus.classList.contains('degraded') && ros && ros.isConnected) {
            updateConnectionStatus('connected');
        }
        if (widget === 'map') {
            queueVisual('mapPose', () => {
                if (generation === connectionGeneration && spec.instance === topic) spec.callback(msg);
            });
            requestAnimationFrame(() => { const job = visualJobs.get('mapPose'); visualJobs.delete('mapPose'); job?.(); });
        }
        else if (['cam', 'mapGrid', 'plan', 'diag'].includes(widget)) queueVisual(widget, () => {
            if (generation === connectionGeneration && spec.instance === topic) spec.callback(msg);
        });
        else spec.callback(msg);
    });
    spec.instance = topic;
    spec.assign(topic);
    addSystemLog(`[ROS] Subscribed ${spec.topic}`);
}

function unsubscribeWidget(widget) {
    const spec = subscriptionSpecs[widget];
    if (!spec || !spec.instance) return;
    spec.instance.unsubscribe();
    spec.assign(null);
    addSystemLog(`[ROS] Unsubscribed ${spec.topic}`);
    spec.instance = null;
    if (widget === 'cam') {
        releaseCamera();
        elCamStatus.textContent = 'Paused';
        elCamStatus.className = 'badge badge-outline';
    }
}

function cameraTopic() {
    return localStorage.getItem('camera-topic') || '/orchid_ui/camera/compressed';
}

function cameraMessageType() {
    return localStorage.getItem('camera-message-type') || 'sensor_msgs/msg/CompressedImage';
}

function cameraThrottleMs() {
    return parseInt(localStorage.getItem('camera-throttle-ms') || '500', 10);
}

function mapTopic() {
    return localStorage.getItem('map-topic') || '/map';
}

function robotPoseTopic() {
    return localStorage.getItem('robot-pose-topic') || '/global_pose';
}

function planTopic() {
    return localStorage.getItem('plan-topic') || '/plan';
}

let pendingCamera = null, cameraDecoding = false, cameraTimer = null, cameraLastAt = 0;
let cameraUrl = null, cameraGeneration = 0, cameraRevision = 0;
let previewUntil = 0, previewTimer = null, latestLiveCamera = null;
let activePreviewKind = null, pendingPreview = null;
function handlePreviewMessage(msg,kind) {
    if (!isWidgetVisible('cam')) return;
    if (activePreviewKind && kind !== activePreviewKind) {
        // Keep only the newest other-topic result until the current preview expires.
        pendingPreview = {msg,kind};
        return;
    }
    activePreviewKind=kind;
    clearTimeout(previewTimer); previewUntil=Infinity;
    queueCameraFrame(msg.data,kind);
}
function resumeLiveCamera() {
    previewUntil=0; clearTimeout(previewTimer); previewTimer=null;
    activePreviewKind=null;
    if (pendingPreview) {
        const next=pendingPreview;pendingPreview=null;
        handlePreviewMessage(next.msg,next.kind);
        return;
    }
    document.getElementById('widget-cam').removeAttribute('data-preview');
    if (latestLiveCamera) handleCameraMessage(latestLiveCamera);
    else {elCamImage.hidden=true;if(cameraUrl)URL.revokeObjectURL(cameraUrl);cameraUrl=null;
        document.getElementById('widget-cam').classList.remove('has-image');elCamFallback.classList.remove('hidden');}
}
function releaseCamera() {
    ++cameraGeneration; ++cameraRevision; previewUntil=0;clearTimeout(previewTimer);previewTimer=null;latestLiveCamera=null;
    activePreviewKind=null;pendingPreview=null;
    document.getElementById('widget-cam').removeAttribute('data-preview');
    pendingCamera = null; clearTimeout(cameraTimer); cameraTimer = null;
    if (cameraUrl) URL.revokeObjectURL(cameraUrl); cameraUrl = null;
    elCamImage.removeAttribute('src'); elCamImage.hidden = true;
    if (cameraCanvas) cameraCanvas.hidden = true;
    document.getElementById('widget-cam').classList.remove('has-image');
    elCamFallback.classList.remove('hidden'); elCamFallback.textContent = 'No camera image';
    visualJobs.delete('cam');
}
function sizeCamera() {
    const pip = document.getElementById('widget-cam'), workspace = document.getElementById('workspace');
    const ratio = Number(pip.dataset.aspect || 4/3);
    const inset = (parseFloat(getComputedStyle(workspace).getPropertyValue('--workspace-inset')) || 0) + 8;
    pip.style.setProperty('--pip-width', `${Math.max(1,Math.min((innerWidth < 650 ? workspace.clientWidth*.60 : 420)*cameraWindowScale,workspace.clientWidth-2*inset,Math.min(workspace.clientHeight*.55,workspace.clientHeight-2*inset)*ratio))}px`);
    clampPicture(pip, workspace); clampPicture(elTreeDetails, workspace);
}
function queueCameraFrame(data,kind='live') {
    pendingCamera = {data,kind,revision:++cameraRevision};
    scheduleCameraFrame();
}
function scheduleCameraFrame() {
    if (cameraDecoding || cameraTimer || document.hidden || currentPage !== 'map' || !pendingCamera) return;
    cameraTimer = setTimeout(async () => {
        cameraTimer = null;
        if (document.hidden || currentPage !== 'map' || !pendingCamera) return;
        const generation = cameraGeneration, frame=pendingCamera;
        pendingCamera=null; cameraDecoding=true;
        let url=null;
        try {
            const bytes=typeof frame.data==='string'?Uint8Array.from(atob(frame.data),c=>c.charCodeAt(0)):new Uint8Array(frame.data);
            url=URL.createObjectURL(new Blob([bytes],{type:'image/jpeg'}));
            const image=new Image();image.src=url;await image.decode();
            if(generation===cameraGeneration && frame.revision===cameraRevision && !document.hidden && currentPage==='map') {
                const old=cameraUrl;cameraUrl=url;url=null;elCamImage.src=cameraUrl;elCamImage.hidden=false;
                if(cameraCanvas)cameraCanvas.hidden=true;
                const pip=document.getElementById('widget-cam');pip.dataset.aspect=image.naturalWidth/image.naturalHeight;
                pip.style.setProperty('--camera-aspect',pip.dataset.aspect);pip.classList.add('has-image');sizeCamera();
                elCamFallback.classList.add('hidden');
                if(frame.kind!=='live') {
                    pip.dataset.preview=frame.kind;previewUntil=performance.now()+previewDuration*1000;
                    clearTimeout(previewTimer);previewTimer=setTimeout(resumeLiveCamera,previewDuration*1000);
                } else {pip.removeAttribute('data-preview');}
                if(old)URL.revokeObjectURL(old);
            }
        } catch {
            if(frame.kind!=='live' && generation===cameraGeneration && frame.revision===cameraRevision)resumeLiveCamera();
        } finally {
            if(url)URL.revokeObjectURL(url);cameraLastAt=performance.now();cameraDecoding=false;
            scheduleCameraFrame();
        }
    },Math.max(0,500-(performance.now()-cameraLastAt)));
}
function handleCameraMessage(msg) {
    latestLiveCamera=msg;
    if(previewUntil>performance.now())return;
    document.getElementById('widget-cam').removeAttribute('data-preview');
    if (cameraMessageType().toLowerCase().includes('compressed')) {
        queueCameraFrame(msg.data);
        if (cameraCanvas) cameraCanvas.hidden = true;
        elCamFallback.classList.add('hidden');
        elCamStatus.textContent = 'Live Stream';
        elCamStatus.className = 'badge badge-success';
        return;
    }
    if (!cameraCanvas) {
        cameraCanvas = document.createElement('canvas');
        cameraCtx = cameraCanvas.getContext('2d');
    }
    if (cameraCanvas.width !== msg.width || cameraCanvas.height !== msg.height) {
        cameraCanvas.width = msg.width;
        cameraCanvas.height = msg.height;
    }
    const rawData = typeof msg.data === 'string'
        ? Uint8Array.from(atob(msg.data), c => c.charCodeAt(0))
        : msg.data;
    const imgData = cameraCtx.createImageData(msg.width, msg.height);
    let srcIdx = 0;
    let dstIdx = 0;
    while (srcIdx + 2 < rawData.length && dstIdx + 3 < imgData.data.length) {
        if (msg.encoding === 'bgr8') {
            imgData.data[dstIdx] = rawData[srcIdx + 2];
            imgData.data[dstIdx + 1] = rawData[srcIdx + 1];
            imgData.data[dstIdx + 2] = rawData[srcIdx];
        } else {
            imgData.data[dstIdx] = rawData[srcIdx];
            imgData.data[dstIdx + 1] = rawData[srcIdx + 1];
            imgData.data[dstIdx + 2] = rawData[srcIdx + 2];
        }
        imgData.data[dstIdx + 3] = 255;
        srcIdx += 3;
        dstIdx += 4;
    }
    cameraCtx.putImageData(imgData, 0, 0);
    const pip = document.getElementById('widget-cam');
    pip.dataset.aspect = msg.width / msg.height;
    pip.style.setProperty('--camera-aspect', pip.dataset.aspect);
    pip.classList.add('has-image'); sizeCamera();
    cameraCanvas.hidden = false;
    cameraCanvas.className = 'cam-stream-img';
    if (!cameraCanvas.parentElement) elCamImage.parentElement.appendChild(cameraCanvas);
    elCamImage.hidden = true;
    elCamFallback.classList.add('hidden');
    elCamStatus.textContent = 'Live Stream';
    elCamStatus.className = 'badge badge-success';
}

function handlePoseMessage(msg) {
    if (!msg?.pose?.position || !msg.pose.orientation) return;
    if (msg.header?.frame_id && msg.header.frame_id.replace(/^\//, '') !== 'map') return;
    const pose = msg.pose;
    robotPose.x = pose.position.x;
    robotPose.y = pose.position.y;
    const q = pose.orientation;
    const siny_cosp = 2 * (q.w * q.z + q.x * q.y);
    const cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z);
    robotPose.yaw = Math.atan2(siny_cosp, cosy_cosp);
    document.getElementById('pose-x').textContent = robotPose.x.toFixed(2);
    document.getElementById('pose-y').textContent = robotPose.y.toFixed(2);
    document.getElementById('pose-yaw').textContent = `${(robotPose.yaw * (180 / Math.PI)).toFixed(1)}°`;
    drawMap();
}

function handleMapMessage(msg) {
    if(mapSource()!=='topic')return;
    if (!msg?.info || !msg.data || !(msg.info.width > 0 && msg.info.height > 0) ||
        msg.data.length !== msg.info.width * msg.info.height || !(msg.info.resolution > 0)) return;
    occupancyGrid = {
        width: Number(msg.info.width) || 0,
        height: Number(msg.info.height) || 0,
        resolution: Number(msg.info.resolution) || 0.05,
        origin: msg.info.origin || { position: { x: 0, y: 0 }, orientation: { z: 0, w: 1 } },
        data: msg.data,
    };
    renderOccupancyGridCache();
    mapRefreshRequested = false;
    unsubscribeWidget('mapGrid');
    drawMap();
}

function handlePlanMessage(msg) {
    latestPlanPath = Array.isArray(msg?.poses)
        ? msg.poses.map((entry) => entry.pose || entry).filter(Boolean)
        : [];
    drawMap();
}

function handleBatteryMessage(msg) {
    if (!msg || msg.percentage === undefined || msg.percentage === null) return;
    lastBatteryAt = Date.now();
    scheduleBatteryStaleCheck();
    const percent = Math.round(Number(msg.percentage) * 100);
    updateBatteryDisplay(percent, msg.power_supply_status);
}

function scheduleBatteryStaleCheck() {
    if (batteryStaleTimer) clearTimeout(batteryStaleTimer);
    batteryStaleTimer = setTimeout(() => {
        if (!lastBatteryAt || Date.now() - lastBatteryAt >= batteryStaleMs) {
            updateBatteryUnavailable();
        }
    }, batteryStaleMs + 100);
}

function updateBatteryDisplay(percent, statusCode) {
    const item = document.getElementById('battery-item');
    if (item) item.classList.remove('hidden');
    const batteryVal = Math.min(100, Math.max(0, Number(percent) || 0));
    elLblBattery.textContent = `${Math.round(batteryVal)}%`;
    item?.setAttribute('aria-label', `Battery ${Math.round(batteryVal)} percent`);

    const radius = Number(elBatteryRing.getAttribute('r')) || 17;
    const circumference = 2 * Math.PI * radius;
    const offset = circumference - (batteryVal / 100) * circumference;
    elBatteryRing.style.strokeDasharray = `${circumference} ${circumference}`;
    elBatteryRing.style.strokeDashoffset = offset;

    if (batteryVal < 20) {
        elBatteryRing.style.stroke = "#ef4444";
        elLblBatteryStatus.textContent = "Critical";
        elLblBatteryStatus.className = "chip-subtext text-error";
    } else if (batteryVal < 50) {
        elBatteryRing.style.stroke = "#f59e0b";
        elLblBatteryStatus.textContent = batteryStatusText(statusCode) || "Discharging";
        elLblBatteryStatus.className = "chip-subtext text-warning";
    } else {
        elBatteryRing.style.stroke = "#10b981";
        elLblBatteryStatus.textContent = batteryStatusText(statusCode) || "Optimal";
        elLblBatteryStatus.className = "chip-subtext text-success";
    }
}

function updateBatteryUnavailable() {
    const item = document.getElementById('battery-item');
    if (item) item.classList.remove('hidden');
    elLblBattery.textContent = '/';
    item?.setAttribute('aria-label', 'Battery data unavailable');
    elBatteryRing.style.strokeDasharray = "0 1";
    elBatteryRing.style.strokeDashoffset = "0";
    elBatteryRing.style.stroke = "#94a3b8";
    elLblBatteryStatus.textContent = "No data";
    elLblBatteryStatus.className = "chip-subtext";
}

function batteryStatusText(statusCode) {
    const code = Number(statusCode);
    if (code === 1) return 'Charging';
    if (code === 2) return 'Discharging';
    if (code === 3) return 'Not charging';
    if (code === 4) return 'Full';
    return '';
}

function speakUiText(text) {
    const message = String(text || '').trim();
    if (!message || speechMuted || !window.speechSynthesis || !window.SpeechSynthesisUtterance) return;
    if (speechInterrupt) window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(message);
    utterance.lang = 'en-US';
    utterance.rate = speechRate;
    utterance.pitch = speechPitch;
    const voice = window.speechSynthesis.getVoices().find((item) => item.name === speechVoiceName);
    if (voice) utterance.voice = voice;
    window.speechSynthesis.speak(utterance);
}

function updateSpeechMuteButton() {
    if (!elBtnToggleSpeech) return;
    const icon = elBtnToggleSpeech.querySelector('i');
    if (icon) {
        icon.className = speechMuted ? 'fa-solid fa-volume-xmark' : 'fa-solid fa-volume-high';
    }
    elBtnToggleSpeech.title = speechMuted ? 'Unmute robot speech' : 'Mute robot speech';
    elBtnToggleSpeech.setAttribute('aria-label', elBtnToggleSpeech.title);
    elBtnToggleSpeech.setAttribute('aria-pressed', String(speechMuted));
    elBtnToggleSpeech.classList.toggle('active', !speechMuted);
}

// -------------------------------------------------------------
// 5. ROS2 Action & Service Setup (replaces the pub/sub workaround)
//
//  - Action:   /<robot_id>/execute_task   (orchid_ros_interfaces/action/ExecuteTask)
//  - Service:  /<robot_id>/get_user_input (orchid_ros_interfaces/srv/GetUserInput)
// -------------------------------------------------------------
function setupActionAndService() {
    if (!ros) return;

    // ---- Action client for ExecuteTask (ROS2-native via rosbridge send_action_goal) ----
    executeTaskAction = new ROSLIB.Action({
        ros: ros,
        name: `/${currentRobotId}/execute_task`,
        actionType: 'orchid_ros_interfaces/action/ExecuteTask'
    });

    executeWorkflowAction = new ROSLIB.Action({ros, name:`/${currentRobotId}/execute_workflow`, actionType:'orchid_ros_interfaces/action/ExecuteWorkflow'});

    // ---- Follow-up user input is brokered over REST by the web server
    //      (see src/main.rs). Start polling the broker for pending prompts. ----
    startFollowupPolling();

    addSystemLog(`Action \`/${currentRobotId}/execute_task\` client & follow-up polling ready.`);
}

// -------------------------------------------------------------
// 6. Task Execution via ROS2 Action
// -------------------------------------------------------------
function submitTaskAction(workflowRequest = null) {
    if (!ros || !ros.isConnected) {
        alert('Please connect to ROS2 bridge first.');
        return;
    }
    updateAvailability();
    if (!canSubmitTask()) return;
    const command = workflowRequest ? workflowRequest.title : elTxtCommand.value.trim();
    if (!command) {
        alert('Task description cannot be empty.');
        return;
    }
    // Store in command history (skip duplicates of the last entry)
    if (!workflowRequest && (commandHistory.length === 0 || commandHistory[commandHistory.length - 1] !== command)) {
        commandHistory.push(command);
        if (commandHistory.length > 100) commandHistory.shift();
    }
    historyIndex = -1;
    const actionClient = workflowRequest ? executeWorkflowAction : executeTaskAction;
    if (!actionClient) {
        addSystemLog('ERROR: Action client not initialised. Connect first.');
        return;
    }

    addSystemLog(`Submitting task goal: "${command}"`);

    activeActionKind = workflowRequest ? 'workflow' : 'task';
    setCurrentRequest('Current Request', command);
    showExecutionOverlay('Sending Task', 'Sending goal task...');
    setActionBadge('Sending');
    taskStage = 'Starting';taskRunning = true; taskAwaiting = true; updateSubmission();renderTaskStage();
    const generation = ++requestGeneration;
    const connection = connectionGeneration;
    const current = () => generation === requestGeneration && connection === connectionGeneration;

    const goalArgs = workflowRequest ? {workflow_id:workflowRequest.id, inputs_json:JSON.stringify(workflowRequest.inputs)} : {task_description:command};

    try { const id = actionClient.sendGoal(
        goalArgs,
        (result) => {
            // on success
            if (!current()) return;
            lastRosMessageAt = performance.now();
            const ok = result.success !== false;
            try {
                addSystemLog(`Goal ${ok ? 'completed' : 'failed'}: ${result.result_message || (ok ? 'Success' : 'Failed')}`);
            } catch (e) {
                addSystemLog(`Task result: ${JSON.stringify(result)}`);
            }
            finishTask(ok ? 'Completed' : 'Failed');
        },
        (feedback) => {
            // on feedback
            if (!current()) return;
            lastRosMessageAt = performance.now();
            taskAwaiting = false;
            try {
                let fbText = feedback.current_action || 'Processing...';
                if (feedback.percent_completed !== undefined) {
                    fbText += ` (${parseFloat(feedback.percent_completed).toFixed(0)}%)`;
                }
                const lifecycle = lifecycleFromFeedback(feedback.current_action);
                if (!followupActive) {
                    showExecutionOverlay(lifecycle, fbText);
                }
                setActionBadge(followupActive ? 'Waiting for Input' : lifecycle);
            } catch (e) {
                elOverlayFeedback.textContent = JSON.stringify(feedback);
            }
        },
        (error) => {
            // on error
            if (!current()) return;
            lastRosMessageAt = performance.now();
            addSystemLog(`Action error: ${error}`);
            finishTask('Failed', {preserve: true});
        },
    ); goalBinding = {socket:ros, id, client:actionClient}; } catch (error) {
        addSystemLog(`Task submission failed: ${error}`);
        finishTask('Failed', {preserve:true});
    }
    setTimeout(() => {
        if (current() && taskAwaiting) setActionBadge('Awaiting task acknowledgement');
    }, 10000);
}

function cancelTaskAction() {
    if (!taskRunning && !followupActive) return;
    if (!ros?.isConnected || !executeTaskAction) { setActionBadge('Reconnect to send cancellation'); return; }
    addSystemLog('Canceling current task goal...');
    if (followupActive) {
        cancelPendingFollowup();
    }
    const client = activeActionKind === 'workflow' ? executeWorkflowAction : executeTaskAction;
    client?.cancelAllGoals();
    finishTask('Cancelled', {preserve: true});
}

function cancelPendingFollowup() {
    fetch('/api/followup', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ cancel: true, prompt_id: followupPromptId })
    }).catch((e) => {
        addSystemLog(`[Follow-up] Failed to cancel pending prompt: ${e}`);
    });
}

function updateTaskStatusUI(statusMessage) {
    addSystemLog(`[Status Update]: ${statusMessage}`);
    // Action callbacks own ordinary completion. Status supports recovery when
    // an action client was lost; late generic messages cannot finish a new goal.
    if (!taskRunning || taskAwaiting || !observingRecoveredTask) return;
    const status = statusMessage.toLowerCase();
    if (/^(completed|success|succeeded)(:|$)/.test(status)) finishTask('Completed');
    else if (/^(failed|execution_failed|planning_failed|rejected|ignored_action)(:|$)/.test(status)) finishTask('Failed', {preserve:true});
    else if (/^cancelled(:|$)/.test(status)) finishTask('Cancelled', {preserve:true});
}

async function submitFollowupResponse() {
    const answer = elTxtFollowupAnswer.value.trim(); if (!answer || replyInFlight) return;
    replyInFlight = true; elBtnSendFollowup.disabled = true;
    const generation = requestGeneration;
    try {
        const response = await fetch('/api/followup', {method:'POST',headers:{'Content-Type':'application/json'},
            body:JSON.stringify({response:answer,prompt_id:followupPromptId}),signal:AbortSignal.timeout(5000)});
        const result = await response.json();
        if (!response.ok || !result.ok) throw new Error('Answer was not accepted; the question may have changed');
        if (generation !== requestGeneration) return;
        followupActive = false; followupPromptId = null;
        showExecutionOverlay('Running','Answer received. Waiting for updated plan…');setActionBadge('Running');
    } catch (error) {
        if (generation === requestGeneration) {
            addSystemLog(`Answer failed: ${error.message}`);
            elOverlayFeedback.textContent = error.message;
            document.getElementById('answer-error').textContent = error.message;
        }
    } finally { replyInFlight = false; elBtnSendFollowup.disabled = false; }
}

function startFollowupPolling() {
    if (followupPollTimer) return;
    followupPollTimer = setInterval(pollFollowup, 1000);
}

async function pollFollowup() {
    if (followupInFlight || followupActive || !taskRunning || !serverReady || document.hidden) return;
    followupInFlight = true;
    const generation = requestGeneration;
    try {
        const res = await fetch('/api/followup', {signal: AbortSignal.timeout(3000)});
        if (!res.ok) return;
        const data = await res.json();
        if (data.prompt && generation === requestGeneration && taskRunning && serverReady) {
            showFollowupModal(data.prompt, data.prompt_id, data.choices);
        }
    } catch (e) {
        // Server not reachable yet; ignore.
    } finally { followupInFlight = false; }
}

function showFollowupModal(prompt, promptId = null, choices = []) {
    followupPromptId = promptId;
    followupActive = true;
    addSystemLog(`[Follow-up Prompt]: ${prompt}`);
    elFollowupPrompt.textContent = prompt;
    const select = document.getElementById('followup-choices');
    select.replaceChildren(new Option('Choose a location, or type an answer', ''));
    for (const choice of choices || []) {
        if (typeof choice.value === 'string' && typeof choice.label === 'string') select.add(new Option(choice.label, choice.value));
    }
    select.classList.toggle('hidden', select.options.length <= 1);
    elTxtFollowupAnswer.value = '';
    showFollowupOverlay();
}

function setActionBadge(label) {
    elActionStatusBadge.textContent = label;
    const normalized = label.toLowerCase().replace(/\s+/g, '-');
    elActionStatusBadge.className = `badge badge-${normalized}`;
}

function setCurrentRequest(kind, text) {
    if(currentRequestText!==text)clearSceneHighlights();
    currentRequestKind = kind;
    currentRequestText = text;
    document.getElementById('ribbon-task').textContent = text || 'Idle';
    document.getElementById('ribbon-task').title = text || 'Idle';
    const label = elCurrentRequest.querySelector('.request-label');
    label.textContent = kind;
    elCurrentRequestText.textContent = text || 'None';
    elCurrentRequest.classList.toggle('hidden', !text);
}

function clearCurrentRequest() {
    currentRequestKind = 'none';
    currentRequestText = '';
    document.getElementById('ribbon-task').textContent = 'Idle';
    document.getElementById('ribbon-task').title = 'Idle';
    elCurrentRequestText.textContent = 'None';
    elCurrentRequest.classList.add('hidden');
}

function showExecutionOverlay(title='Running', feedback='') {
    elOverlayTitle.textContent=title;elOverlayFeedback.textContent=feedback;
    elExecutionFeedbackPanel.classList.remove('hidden');elOverlayExecuting.classList.remove('hidden');
    if (!followupActive) {
        elFollowupInlinePanel.classList.add('hidden');document.getElementById('answer-buttons').classList.add('hidden');
        document.getElementById('cmd-input-container').classList.remove('hidden');document.getElementById('task-buttons').classList.remove('hidden');
        elBtnCancelTask.classList.remove('hidden');elBtnSendTask.classList.add('hidden');
    }
}
function showFollowupOverlay() {
    document.getElementById('cmd-input-container').classList.add('hidden');document.getElementById('task-buttons').classList.add('hidden');
    document.getElementById('answer-buttons').classList.remove('hidden');elFollowupInlinePanel.classList.remove('hidden');
    elOverlayExecuting.classList.add('hidden');setActionBadge('Waiting for input');
    document.getElementById('answer-error').textContent='';
}
function hideExecutionOverlay() {
    elOverlayExecuting.classList.add('hidden');elFollowupInlinePanel.classList.add('hidden');
    document.getElementById('answer-buttons').classList.add('hidden');document.getElementById('task-buttons').classList.remove('hidden');
    document.getElementById('cmd-input-container').classList.remove('hidden');
    elBtnCancelTask.classList.add('hidden');elBtnSendTask.classList.remove('hidden');
}

function lifecycleFromFeedback(currentAction) {
    const text = (currentAction || '').toLowerCase();
    if (text.includes('planning')) return 'Planning';
    if (text.includes('build')) return 'Building Tree';
    if (text.includes('waiting') || text.includes('input')) return 'Waiting for Input';
    if (text.includes('cancel')) return 'Cancelled';
    if (text.includes('fail') || text.includes('error')) return 'Failed';
    if (text.includes('success') || text.includes('complete')) return 'Completed';
    return 'Running';
}

// -------------------------------------------------------------
// 8. Speech-to-Text (Native API)
// -------------------------------------------------------------
function startSpeechRecognition(targetTextarea, statusDiv) {
    if (!recognition) return;

    speechTargetTextarea = targetTextarea;
    statusDiv.classList.remove('hidden');

    // Ensure status shows listening state
    if (statusDiv === elSpeechStatus) {
        statusDiv.textContent = 'Listening... Speak now.';
    } else if (statusDiv === elFollowupSpeechStatus) {
        statusDiv.textContent = 'Listening... Speak now.';
    }

    elBtnMic.disabled = true;
    elBtnFollowupMic.disabled = true;

    try {
        recognition.start();
    } catch (e) {
        console.error("Speech recognition failed to start:", e);
        elBtnMic.disabled = false;
        elBtnFollowupMic.disabled = false;
    }
}

// -------------------------------------------------------------
// 9. 2D Map Drawing Logic (HTML5 Canvas)
// -------------------------------------------------------------
function initCanvas() {
    refreshStaticMap();drawMap();
    // Redraw map whenever the container changes size
    if (window.ResizeObserver) {
        const ro = new ResizeObserver(() => drawMap());
        if (elMapCanvas.parentElement) ro.observe(elMapCanvas.parentElement);
    }
}

function renderOccupancyGridCache() {
    if (!occupancyGrid || occupancyGrid.width <= 0 || occupancyGrid.height <= 0) {
        mapCanvasCache = null;
        mapCanvasCacheCtx = null;
        return;
    }
    mapCanvasCache = document.createElement('canvas');
    mapCanvasCache.width = occupancyGrid.width;
    mapCanvasCache.height = occupancyGrid.height;
    mapCanvasCacheCtx = mapCanvasCache.getContext('2d');
    const img = mapCanvasCacheCtx.createImageData(occupancyGrid.width, occupancyGrid.height);
    const data = occupancyGrid.data;
    for (let y = 0; y < occupancyGrid.height; y++) {
        for (let x = 0; x < occupancyGrid.width; x++) {
            const srcIdx = x + (occupancyGrid.height - 1 - y) * occupancyGrid.width;
            const dstIdx = (x + y * occupancyGrid.width) * 4;
            const occ = Number(data[srcIdx]);
            let value = 244;
            if (occ < 0) value = 205;
            else if (occ >= 65) value = 42;
            else if (occ > 0) value = Math.round(244 - (occ / 100) * 170);
            img.data[dstIdx] = value;
            img.data[dstIdx + 1] = value;
            img.data[dstIdx + 2] = value;
            img.data[dstIdx + 3] = 255;
        }
    }
    mapCanvasCacheCtx.putImageData(img, 0, 0);
}

function mapOriginYaw() {
    const q = occupancyGrid?.origin?.orientation || {};
    const x = Number(q.x) || 0;
    const y = Number(q.y) || 0;
    const z = Number(q.z) || 0;
    const w = Number(q.w);
    const safeW = Number.isFinite(w) ? w : 1;
    return Math.atan2(2 * (safeW * z + x * y), 1 - 2 * (y * y + z * z));
}

function hasValidOccupancyGrid() {
    return Boolean(occupancyGrid && occupancyGrid.width > 0 && occupancyGrid.height > 0 && mapCanvasCache);
}

function readNumericInput(element, fallback) {
    const raw = element?.value;
    if (raw === undefined || raw === null || String(raw).trim() === '') return fallback;
    const value = Number(raw);
    return Number.isFinite(value) ? value : fallback;
}

function fitMapToCanvas() {
    if (!occupancyGrid || occupancyGrid.width <= 0 || occupancyGrid.height <= 0) return false;
    const width = elMapCanvas.width || elMapCanvas.parentElement?.clientWidth || 500;
    const height = elMapCanvas.height || elMapCanvas.parentElement?.clientHeight || 400;
    const resolution = occupancyGrid.resolution || 0.05;
    const mapWidthM = occupancyGrid.width * resolution;
    const mapHeightM = occupancyGrid.height * resolution;
    if (mapWidthM <= 0 || mapHeightM <= 0) return false;

    const padding = 36;
    const fitScale = Math.min(
        Math.max(0.1, (width - padding * 2) / mapWidthM),
        Math.max(0.1, (height - padding * 2) / mapHeightM),
    );
    const scale = Math.max(0.1, Math.min(600, fitScale));
    const drawWidth = mapWidthM * scale;
    const drawHeight = mapHeightM * scale;

    elMapScale.value = scale >= 10 ? String(Math.floor(scale)) : scale.toFixed(1);
    elMapOriginX.value = String(Math.round((width - drawWidth) / 2));
    elMapOriginY.value = String(Math.round((height - drawHeight) / 2));
    saveMapView();
    mapAutoFitPending = false;
    return true;
}

function mapViewMetrics() {
    const width = elMapCanvas.width;
    const height = elMapCanvas.height;
    const scale = Math.max(0.1, readNumericInput(elMapScale, 50));
    const panX = readNumericInput(elMapOriginX, 0);
    const panY = readNumericInput(elMapOriginY, 0);
    const resolution = occupancyGrid?.resolution || 0.05;
    return {
        width,
        height,
        scale,
        panX,
        panY,
        resolution,
        drawScale: scale * resolution,
        rotationRad: readNumericInput(elMapRotation, 0) * Math.PI / 180,
        centerX: width / 2,
        centerY: height / 2,
    };
}

function rotateDisplayPoint(point, metrics = mapViewMetrics()) {
    const { rotationRad, centerX, centerY } = metrics;
    if (!rotationRad) return point;
    const dx = point.x - centerX;
    const dy = point.y - centerY;
    const c = Math.cos(rotationRad);
    const s = Math.sin(rotationRad);
    return { x: centerX + dx * c - dy * s, y: centerY + dx * s + dy * c };
}

function inverseRotateDisplayDelta(dx, dy, rotationRad) {
    const c = Math.cos(rotationRad);
    const s = Math.sin(rotationRad);
    return { x: dx * c + dy * s, y: -dx * s + dy * c };
}

function worldToCanvasPoint(wx, wy) {
    const metrics = mapViewMetrics();
    if (!occupancyGrid) {
        return rotateDisplayPoint({ x: metrics.panX + wx * metrics.scale, y: metrics.panY - wy * metrics.scale }, metrics);
    }
    const origin = occupancyGrid.origin?.position || {};
    const dx = Number(wx) - (Number(origin.x) || 0);
    const dy = Number(wy) - (Number(origin.y) || 0);
    const yaw = mapOriginYaw();
    const c = Math.cos(-yaw);
    const s = Math.sin(-yaw);
    const mapX = dx * c - dy * s;
    const mapY = dx * s + dy * c;
    const raw = {
        x: metrics.panX + (mapX / metrics.resolution) * metrics.drawScale,
        y: metrics.panY + (occupancyGrid.height - 1 - mapY / metrics.resolution) * metrics.drawScale,
    };
    return rotateDisplayPoint(raw, metrics);
}

function canvasToWorldPoint(canvasX, canvasY) {
    if (!occupancyGrid) return null;
    const metrics = mapViewMetrics();
    const raw = rotateDisplayPoint({ x: canvasX, y: canvasY }, { ...metrics, rotationRad: -metrics.rotationRad });
    const mapX = (raw.x - metrics.panX) / metrics.drawScale * metrics.resolution;
    const mapY = (occupancyGrid.height - 1 - (raw.y - metrics.panY) / metrics.drawScale) * metrics.resolution;
    const yaw = mapOriginYaw();
    const c = Math.cos(yaw);
    const s = Math.sin(yaw);
    const origin = occupancyGrid.origin?.position || {};
    return {
        x: (Number(origin.x) || 0) + c * mapX - s * mapY,
        y: (Number(origin.y) || 0) + s * mapX + c * mapY,
    };
}

function canvasEventPoint(event) {
    const rect = elMapCanvas.getBoundingClientRect();
    return {
        x: (event.clientX - rect.left) * elMapCanvas.width / rect.width,
        y: (event.clientY - rect.top) * elMapCanvas.height / rect.height,
    };
}

function drawMap() {
    if (currentPage !== 'map') return;
    queueVisual('mapDraw', drawMapNow);
}
function drawMapNow() {
    const container = elMapCanvas.parentElement;
    if (container) {
        const cw = container.clientWidth;
        const ch = container.clientHeight;
        if (cw > 0 && ch > 0) {
            if (elMapCanvas.width !== cw) elMapCanvas.width = cw;
            if (elMapCanvas.height !== ch) elMapCanvas.height = ch;
        }
    }

    if (mapAutoFitPending && hasValidOccupancyGrid()) {
        fitMapToCanvas();
    }

    const ctx = elMapCanvas.getContext('2d');
    const width = elMapCanvas.width;
    const height = elMapCanvas.height;
    const metrics = mapViewMetrics();
    const { scale, panX, panY, rotationRad, centerX, centerY, drawScale: z } = metrics;

    ctx.clearRect(0, 0, width, height);

    if (hasValidOccupancyGrid()) {
        ctx.save();
        if (rotationRad) {
            ctx.translate(centerX, centerY);
            ctx.rotate(rotationRad);
            ctx.translate(-centerX, -centerY);
        }
        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(
            mapCanvasCache,
            panX,
            panY,
            occupancyGrid.width * z,
            occupancyGrid.height * z,
        );
        ctx.restore();
    } else {
        const light = document.body.classList.contains('light-theme');
        ctx.fillStyle = light ? '#edf1f9' : '#0b0f19';
        ctx.fillRect(0, 0, width, height);
        ctx.fillStyle = light ? '#465a72' : '#94a3b8';
        ctx.font = '600 13px system-ui, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(mapSource()==='static'?(mapLoadMessage||'Loading static map…'):`Awaiting map topic ${mapTopic()}`, width / 2, height / 2);
    }

    if (!hasValidOccupancyGrid()) return;

    if (latestPlanPath.length > 1) {
        ctx.strokeStyle = '#a855f7';
        ctx.lineWidth = 3;
        ctx.shadowColor = 'rgba(168, 85, 247, 0.45)';
        ctx.shadowBlur = 5;
        ctx.beginPath();
        const s = worldToCanvasPoint(latestPlanPath[0].position.x, latestPlanPath[0].position.y);
        ctx.moveTo(s.x, s.y);
        for (let i = 1; i < latestPlanPath.length; i++) {
            const p = worldToCanvasPoint(latestPlanPath[i].position.x, latestPlanPath[i].position.y);
            ctx.lineTo(p.x, p.y);
        }
        ctx.stroke();
        ctx.shadowBlur = 0;
        drawPlanEndpoint(ctx, s, '#22c55e');
        const endPose = latestPlanPath[latestPlanPath.length - 1];
        drawPlanEndpoint(ctx, worldToCanvasPoint(endPose.position.x, endPose.position.y), '#ef4444');
    }

    drawSceneOverlay(ctx);

    const rp = worldToCanvasPoint(robotPose.x, robotPose.y);
    const rx = rp.x;
    const ry = rp.y;

    const robotRadius = Math.max(4, 0.25 * scale);
    const headingLength = Math.max(12, 0.4 * scale);
    ctx.fillStyle = 'rgba(168, 85, 247, 0.28)';
    ctx.beginPath();
    ctx.arc(rx, ry, robotRadius + 6, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = '#a855f7';
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(rx, ry, robotRadius, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(rx, ry);
    const headingWorld = {
        x: robotPose.x + headingLength / Math.max(scale, 0.1) * Math.cos(robotPose.yaw),
        y: robotPose.y + headingLength / Math.max(scale, 0.1) * Math.sin(robotPose.yaw),
    };
    const headingEnd = worldToCanvasPoint(headingWorld.x, headingWorld.y);
    const hx = headingEnd.x;
    const hy = headingEnd.y;
    ctx.lineTo(hx, hy);
    ctx.stroke();
}

function drawPlanEndpoint(ctx, point, color) {
    ctx.fillStyle = color;
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(point.x, point.y, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
}

function sceneLabel(object) {
    return String(object?.label || object?.name || `Object ${object?.id ?? ''}`).replaceAll('_', ' ');
}

function sceneObjectCategory(object) {
    return classifySceneObject(object, sceneClasses);
}

function sceneCategoryEnabled(category) {
    return sceneRepresentationEnabled(category,'Boxes') || sceneRepresentationEnabled(category,'Nodes');
}

function sceneRepresentationEnabled(category,kind) { return Boolean(sceneOverlay[category+kind]); }

function sceneColor(category, selected) {
    if (selected) return '#ffe04b';
    if (category === 'furniture') return '#c084fc';
    if (category === 'movable') return '#39e58c';
    return '#ff9f43';
}

function sceneCorners(object) {
    if (!Array.isArray(object?.center) || !Array.isArray(object?.size) || object.center.length < 2 || object.size.length < 2) return null;
    const cx = Number(object.center[0]);
    const cy = Number(object.center[1]);
    const sx = Number(object.size[0]);
    const sy = Number(object.size[1]);
    if (![cx, cy, sx, sy].every(Number.isFinite) || sx <= 0 || sy <= 0) return null;
    const rotation = object.rotation;
    const r00 = Number(rotation?.[0]?.[0]);
    const r01 = Number(rotation?.[0]?.[1]);
    const r10 = Number(rotation?.[1]?.[0]);
    const r11 = Number(rotation?.[1]?.[1]);
    const c = Number.isFinite(r00) && Number.isFinite(r01) && Number.isFinite(r10) && Number.isFinite(r11)
        ? { x: r00, y: r10, u: r01, v: r11 }
        : { x: 1, y: 0, u: 0, v: 1 };
    return [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([x, y]) => ({
        x: cx + c.x * x * sx / 2 + c.u * y * sy / 2,
        y: cy + c.y * x * sx / 2 + c.v * y * sy / 2,
    }));
}

function sceneRoomGroups() {
    const groups = new Map();
    for (const object of sceneGraph?.objects || []) {
        const roomId = object.room_id;
        if (!roomId) continue;
        if (!groups.has(roomId)) groups.set(roomId, []);
        groups.get(roomId).push(object);
    }
    return groups;
}

function sceneHighlighted(object){
    const matches=ref=>ref && (String(ref.id)===String(object.id) || (ref.boxer_id!=null && String(ref.boxer_id)===String(object.id)));
    return matches(sceneHighlights.navigation_target) || (sceneHighlights.known_objects||[]).some(matches);
}
function drawSceneOverlay(ctx) {
    if (!sceneGraph || !hasValidOccupancyGrid()) return;
    const visibleObjects = [];
    for (const object of sceneGraph.objects || []) {
        const category = sceneObjectCategory(object);
        if (!sceneCategoryEnabled(category)) continue;
        const corners = sceneCorners(object);
        if (!corners && !(sceneRepresentationEnabled(category,'Nodes') && Array.isArray(object.center) && object.center.slice(0,2).every(Number.isFinite))) continue;
        const points = (corners || [object.center].map(c=>({x:c[0],y:c[1]})) ).map((point) => worldToCanvasPoint(point.x, point.y));
        const selected = selectedSceneObject?.id === object.id;
        ctx.save();
        ctx.beginPath();
        ctx.moveTo(points[0].x, points[0].y);
        points.slice(1).forEach((point) => ctx.lineTo(point.x, point.y));
        ctx.closePath();
        const color = sceneColor(category, selected);
        ctx.strokeStyle = color;
        ctx.lineWidth = sceneOutlineWidth * (selected ? 2 : 1);
        ctx.setLineDash(category === 'structural' ? [6, 4] : []);
        ctx.globalAlpha = selected ? 1 : 0.78;
        const highlighted=sceneHighlighted(object);
        if (sceneRepresentationEnabled(category,'Boxes') && corners) {
            ctx.stroke();
            if(highlighted){
                ctx.save();ctx.clip();ctx.globalAlpha=.45;ctx.strokeStyle=sceneColor(category,false);ctx.lineWidth=2;ctx.setLineDash([]);
                const minX=Math.min(...points.map(p=>p.x)),maxX=Math.max(...points.map(p=>p.x));
                const minY=Math.min(...points.map(p=>p.y)),maxY=Math.max(...points.map(p=>p.y));
                const span=maxY-minY;
                for(let x=minX-span;x<maxX;x+=12){ctx.beginPath();ctx.moveTo(x,minY);ctx.lineTo(x+span,maxY);ctx.stroke();}
                ctx.restore();
            }
        }
        ctx.setLineDash([]);
        if (sceneRepresentationEnabled(category,'Nodes') && Array.isArray(object.center)) {
            const centre=worldToCanvasPoint(Number(object.center[0]),Number(object.center[1]));
            if(highlighted){
                ctx.save();const radius=sceneNodeRadius+22,color=sceneColor(category,false);
                const halo=ctx.createRadialGradient(centre.x,centre.y,sceneNodeRadius,centre.x,centre.y,radius);
                halo.addColorStop(0,color+'aa');halo.addColorStop(1,color+'00');ctx.globalAlpha=1;ctx.fillStyle=halo;
                ctx.beginPath();ctx.arc(centre.x,centre.y,radius,0,Math.PI*2);ctx.fill();ctx.restore();
            }
            ctx.beginPath();ctx.arc(centre.x,centre.y,sceneNodeRadius,0,Math.PI*2);
            ctx.fillStyle=sceneColor(category,false);ctx.globalAlpha=1;ctx.fill();
            ctx.strokeStyle=selected?'#ffe04b':'#152138';ctx.lineWidth=selected?3:1;ctx.stroke();
        }
        if (sceneOverlay.labels && sceneCategoryEnabled(category)) {
            const boxLabel = sceneRepresentationEnabled(category,'Boxes') && corners;
            const labelPoint = boxLabel
                ? points.reduce((best, point) => point.y < best.y ? point : best, points[0])
                : worldToCanvasPoint(Number(object.center[0]),Number(object.center[1]));
            ctx.font = selected ? '700 12px Outfit, sans-serif' : '600 10px Outfit, sans-serif';
            ctx.textAlign = 'left';
            ctx.textBaseline = boxLabel ? 'bottom' : 'middle';
            ctx.fillStyle = color;
            ctx.fillText(sceneLabel(object), labelPoint.x + (boxLabel ? 3 : sceneNodeRadius + 4), labelPoint.y - (boxLabel ? 3 : 0));
        }
        ctx.restore();
        visibleObjects.push({ object, category, corners });
    }

    if (sceneOverlay.visitPoses) {
        for (const object of sceneGraph.objects || []) {
            if (sceneObjectCategory(object) !== 'furniture') continue;
            for (const pose of validVisitPoses(object)) {
                const start=worldToCanvasPoint(pose.x,pose.y);
                const end=worldToCanvasPoint(pose.x+.35*Math.cos(pose.yaw),pose.y+.35*Math.sin(pose.yaw));
                const angle=Math.atan2(end.y-start.y,end.x-start.x);
                ctx.save();ctx.strokeStyle=sceneColor('furniture',selectedSceneObject?.id===object.id);
                ctx.fillStyle=ctx.strokeStyle;ctx.lineWidth=2;ctx.setLineDash([]);
                ctx.beginPath();ctx.arc(start.x,start.y,4,0,2*Math.PI);ctx.fill();
                ctx.beginPath();ctx.moveTo(start.x,start.y);ctx.lineTo(end.x,end.y);
                ctx.moveTo(end.x-7*Math.cos(angle-.5),end.y-7*Math.sin(angle-.5));
                ctx.lineTo(end.x,end.y);ctx.lineTo(end.x-7*Math.cos(angle+.5),end.y-7*Math.sin(angle+.5));ctx.stroke();ctx.restore();
            }
        }
    }

    for (const [roomId, objects] of sceneRoomGroups()) {
            const interior = objects.find((object) => Array.isArray(object.room_interior_point))?.room_interior_point;
            if (!interior || interior.length < 2) continue;
            ctx.save();
            const roomPoints = objects.flatMap((object) => (sceneCorners(object) || []).map((corner) => worldToCanvasPoint(corner.x, corner.y)));
            const point = roomPoints.length
                ? {
                    x: (Math.min(...roomPoints.map((item) => item.x)) + Math.max(...roomPoints.map((item) => item.x))) / 2,
                    y: (Math.min(...roomPoints.map((item) => item.y)) + Math.max(...roomPoints.map((item) => item.y))) / 2,
                }
                : worldToCanvasPoint(Number(interior[0]), Number(interior[1]));
            if (sceneOverlay.rooms && roomPoints.length) {
                const minX = Math.min(...roomPoints.map((item) => item.x));
                const maxX = Math.max(...roomPoints.map((item) => item.x));
                const minY = Math.min(...roomPoints.map((item) => item.y));
                const maxY = Math.max(...roomPoints.map((item) => item.y));
                ctx.strokeStyle = 'rgba(45, 212, 191, 0.78)';
                ctx.lineWidth = 1;
                ctx.setLineDash([3, 5]);
                ctx.strokeRect(minX, minY, maxX - minX, maxY - minY);
                ctx.setLineDash([]);
            }
            ctx.fillStyle = '#64748b';
            ctx.font = '700 11px Outfit, sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            const roomName = objects.find((object) => object.room_type)?.room_type || roomId;
            ctx.fillText(String(roomName).replaceAll('_', ' '), point.x, point.y);
            ctx.restore();
    }
}

function scenePointInPolygon(point, polygon) {
    let inside = false;
    for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
        const xi = polygon[i].x, yi = polygon[i].y;
        const xj = polygon[j].x, yj = polygon[j].y;
        const crosses = ((yi > point.y) !== (yj > point.y)) &&
            point.x < (xj - xi) * (point.y - yi) / ((yj - yi) || 1e-12) + xi;
        if (crosses) inside = !inside;
    }
    return inside;
}

function selectSceneAtCanvasPoint(canvasPoint) {
    if (!sceneGraph || !hasValidOccupancyGrid()) return clearSceneSelection();
    const worldPoint = canvasToWorldPoint(canvasPoint.x, canvasPoint.y);
    if (!worldPoint) return clearSceneSelection();
    const objects = (sceneGraph.objects || []).slice().reverse();
    const nodes=objects.filter(object=>sceneRepresentationEnabled(sceneObjectCategory(object),'Nodes')&&Array.isArray(object.center)).map(object=>{
        const point=worldToCanvasPoint(Number(object.center[0]),Number(object.center[1]));
        return {object,distance:Math.hypot(point.x-canvasPoint.x,point.y-canvasPoint.y)};
    }).filter(hit=>hit.distance<=Math.max(sceneNodeRadius,22)).sort((a,b)=>a.distance-b.distance);
    const match = nodes[0]?.object || objects.find((object) => {
        if (!sceneRepresentationEnabled(sceneObjectCategory(object),'Boxes')) return false;
        const corners = sceneCorners(object);
        return corners && scenePointInPolygon(worldPoint, corners);
    });
    if (match) {
        selectedSceneObject = match;
        renderSceneInfo(match);
    } else {
        clearSceneSelection();
    }
    drawMap();
}

function renderSceneInfo(object) {
    if (!elSceneInfo || !elSceneInfoContent) return;
    const category = sceneObjectCategory(object);
    const roomName = object.room_type || object.room_id || 'Unassigned';
    const supported = (sceneGraph?.objects || []).filter((candidate) => String(candidate.supported_by) === String(object.id));
    const support = sceneGraph?.objects?.find((candidate) => String(candidate.id) === String(object.supported_by));
    const rows = [
        ['Name', object.name || `${sceneLabel(object)} ${object.id}`],
        ['Type', category],
        ['Room', String(roomName).replaceAll('_', ' ')],
    ];
    if (support) rows.push(['Supported by', sceneLabel(support)]);
    if (supported.length) rows.push(['Contains', supported.map(sceneLabel).join(', ')]);
    if (object.manually_added) rows.push(['Source', object.geometry_source || 'Manually added']);
    if (object.confidence !== undefined) rows.push(['Confidence', `${Math.round(Number(object.confidence) * 100)}%`]);
    if (Array.isArray(object.base_access_poses)) rows.push(['Valid access poses', String(validVisitPoses(object).length)]);
    elSceneInfoContent.innerHTML = `<div class="scene-info-title">${escapeHtml(sceneLabel(object))}</div>` +
        rows.map(([key, value]) => `<div class="scene-info-row"><span>${escapeHtml(key)}</span><strong>${escapeHtml(value)}</strong></div>`).join('');
    elSceneInfo.classList.remove('hidden');
}

function clearSceneSelection() {
    selectedSceneObject = null;
    elSceneInfo?.classList.add('hidden');
    drawMap();
}

function setMapStatus(message) {
    if (!elMapStatus) return;
    elMapStatus.textContent = message || '';
    elMapStatus.classList.toggle('hidden', !message);
}

async function loadSceneGraph(path) {
    const generation = ++sceneGraphLoadGeneration;
    sceneGraphAbort?.abort();
    sceneGraphAbort = new AbortController();
    setMapStatus(`Loading scene graph ${path}...`);
    try {
        const loaded = await loadSceneDataset(path, {signal:sceneGraphAbort.signal});
        if (generation !== sceneGraphLoadGeneration) return;
        // Commit the graph and its classification as one successful dataset.
        sceneGraph = loaded.graph;
        sceneClasses = loaded.classes;
        const poseCount=sceneGraph.objects.filter(o=>sceneObjectCategory(o)==='furniture').reduce((sum,o)=>sum+validVisitPoses(o).length,0);
        document.getElementById('scene-visit-pose-count').textContent=String(poseCount);
        selectedSceneObject = null;
        elSceneInfo?.classList.add('hidden');
        setMapStatus(`Scene overlay: ${sceneGraph.objects.length} objects — ${path}`);
        drawMap();
    } catch (error) {
        if (generation !== sceneGraphLoadGeneration || error.name === 'AbortError') return;
        setMapStatus(`Scene overlay ${sceneGraph ? 'update failed (showing previous graph)' : 'unavailable'}: ${error.message}`);
        drawMap();
    }
}

function updateMapZoomLockButton() {
    if (!elBtnMapZoomLock) return;
    const icon = elBtnMapZoomLock.querySelector('i');
    const label = elBtnMapZoomLock.querySelector('span');
    if (icon) icon.className = mapZoomLocked ? 'fa-solid fa-lock-open' : 'fa-solid fa-lock';
    if (label) label.textContent = mapZoomLocked ? 'Unlock zoom' : 'Lock zoom';
    elBtnMapZoomLock.title = mapZoomLocked ? 'Unlock map zoom' : 'Lock map zoom';
    elBtnMapZoomLock.classList.toggle('active', mapZoomLocked);
}

function updateTreeZoomLockButton() {
    if (!elBtnTreeZoomLock) return;
    const icon = elBtnTreeZoomLock.querySelector('i');
    if (icon) icon.className = treeZoomLocked ? 'fa-solid fa-lock-open' : 'fa-solid fa-lock';
    elBtnTreeZoomLock.title = treeZoomLocked ? 'Unlock BT zoom' : 'Lock BT zoom';
    elBtnTreeZoomLock.classList.toggle('active', treeZoomLocked);
}

// -------------------------------------------------------------
// 10. PyTrees SVG behavior tree rendering
// -------------------------------------------------------------
function renderBehaviorTree(node) {
    if (!node) return;
    latestTreeData = node;
    if (currentPage !== 'tree') return;
    const topology = topologyKey(node);
    if (topology !== treeTopology) {
        treeTopology = topology; treeVisibleKey = '';
        expandedTree.clear(); collapsedTree.clear(); selectedTreeNodeId = null;
    }
    const items = projectTree(node, treeDepth, expandedTree, collapsedTree);
    const visibleKey = items.map(item => `${item.path}:${item.open}`).join('|');
    const ns = 'http://www.w3.org/2000/svg';
    if (visibleKey !== treeVisibleKey || !document.getElementById('tree-viewport')) {
        treeVisibleKey = visibleKey; treeElements.clear();
        elTreeSvg.replaceChildren();
        const viewport = document.createElementNS(ns, 'g'); viewport.id = 'tree-viewport';
        viewport.setAttribute('transform', `translate(${treePanX}, ${treePanY}) scale(${treeZoom})`);
        elTreeSvg.appendChild(viewport);
        const byPath = new Map(items.map(item => [item.path, item]));
        let row = 0;
        const place = item => {
            const children = items.filter(child => child.parent === item.path);
            children.forEach(place);
            item.x = 20 + item.level * 210;
            item.y = children.length ? children.reduce((sum, child) => sum + child.y, 0) / children.length : 20 + row++ * 76;
        };
        place(items[0]);
        const centerOffset = elTreeSvg.parentElement.clientHeight / 2 - 29 - items[0].y;
        items.forEach(item => { item.y += centerOffset; });
        const height = 40 + Math.max(row, 1) * 76;
        elTreeSvg.setAttribute('width', Math.max(elTreeSvg.parentElement.clientWidth, 40 + (Math.max(...items.map(i => i.level)) + 1) * 210));
        elTreeSvg.setAttribute('height', Math.max(elTreeSvg.parentElement.clientHeight, height));
        for (const item of items) {
            let link = null;
            if (item.parent) {
                const parent = byPath.get(item.parent); link = document.createElementNS(ns, 'path');
                const x = parent.x + 190, y = parent.y + 29, endY = item.y + 29, mid = (x + item.x) / 2;
                link.setAttribute('d', `M${x},${y} C${mid},${y} ${mid},${endY} ${item.x},${endY}`);
                viewport.appendChild(link);
            }
            const group = document.createElementNS(ns, 'g'); group.dataset.nodeId = item.path;
            const rect = document.createElementNS(ns, 'rect');
            for (const [key, value] of Object.entries({x:item.x, y:item.y, width:190, height:58, rx:7})) rect.setAttribute(key, value);
            group.appendChild(rect);
            const text = (y, className) => {
                const t = document.createElementNS(ns, 'text'); t.setAttribute('x', item.x + 8); t.setAttribute('y', item.y + y); t.setAttribute('class', className); group.appendChild(t); return t;
            };
            const type = text(15, 'tree-node-type'), label = text(32, 'tree-node-text'), status = text(49, 'tree-node-text tree-node-status');
            group.addEventListener('click', event => {
                event.stopPropagation();
                const current = findTreeNodeByPath(latestTreeData, item.path);
                if ((current?.children || []).length) {
                    if (item.open) { expandedTree.delete(item.path); collapsedTree.add(item.path); }
                    else { collapsedTree.delete(item.path); expandedTree.add(item.path); }
                }
                selectedTreeNodeId = item.path;
                updateTreeDetails(current);
                renderBehaviorTree(latestTreeData);
            });
            viewport.appendChild(group); treeElements.set(item.path, {group, rect, type, label, status, link});
        }
    }
    for (const item of items) {
        const view = treeElements.get(item.path), n = item.node;
        const status = n.status || 'INVALID';
        view.group.setAttribute('class', `tree-node ${item.active ? 'active-path' : ''} ${selectedTreeNodeId === item.path ? 'selected' : ''}`);
        view.rect.setAttribute('class', `tree-node-rect ${status} ${(n.node_type || 'generic').toLowerCase()}`);
        view.type.textContent = compactNodeType(n.node_type) + ((n.children || []).length ? (item.open ? ' −' : ` +${item.hidden}`) : '');
        view.label.textContent = (n.name || '').length > 25 ? n.name.slice(0, 23) + '…' : n.name;
        view.status.textContent = `${status} ${formatNodeDuration(n)}`;
        if (view.link) view.link.setAttribute('class', `tree-link ${item.active ? 'active-path' : ''}`);
    }
    // Layout anchors the root at left centre. Expanding active paths changes topology,
    // never the user's viewport transform or explicit pan/zoom.
    const selected = selectedTreeNodeId ? findTreeNodeByPath(node, selectedTreeNodeId) : null;
    if (!selected) selectedTreeNodeId = null;
    updateTreeDetails(selectedTreeNodeId ? selected : null);
}

function compactNodeType(nodeType) {
    const text = (nodeType || 'Generic').replace(/^Bt/, '').replace(/Leaf$/, '');
    return text || 'Generic';
}

function findTreeNodeByPath(node, path) {
    if (!node || !path) return null;
    let current = node;
    for (const index of path.split('.').slice(1)) current = current?.children?.[Number(index)];
    return current || null;
}

function findDeepestRunningNode(node) {
    const result = findDeepestRunningNodeResult(node, null, 0);
    return result ? result.node : null;
}

function findDeepestRunningNodeResult(node, best = null, depth = 0) {
    if (!node) return best;
    const status = String(node.status || '').toUpperCase();
    if (status === 'RUNNING' && (!best || depth >= best.depth)) {
        best = { node, depth };
    }
    for (const child of node.children || []) {
        best = findDeepestRunningNodeResult(child, best, depth + 1);
    }
    return best;
}

function findTreeNodeByName(node, name) {
    if (!node || !name) return null;
    const needle = String(name).toLowerCase();
    const candidates = [
        node.name,
        node.display_name,
        node.action_type,
        node.operation?.current_action_name,
    ].filter(Boolean).map(v => String(v).toLowerCase());
    if (candidates.some(v => v === needle || needle.includes(v) || v.includes(needle))) {
        return node;
    }
    for (const child of node.children || []) {
        const found = findTreeNodeByName(child, name);
        if (found) return found;
    }
    return null;
}

function formatNodeDuration(node) {
    if (!node) return '';
    if (node.duration_s !== undefined && node.duration_s !== null) {
        return `${Number(node.duration_s).toFixed(1)}s`;
    }
    if (node.elapsed_s !== undefined && node.elapsed_s !== null) {
        return `${Number(node.elapsed_s).toFixed(1)}s`;
    }
    return '';
}

function updateTreeDetails(node) {
    if (!elTreeDetails) return;
    const show = Boolean(node);
    elTreeDetails.hidden = !show;
    if (currentPage !== 'tree') elTreeDetails.hidden = true;
    if (!show) {
        elTreeDetails.innerHTML = '';
        delete elTreeDetails.dataset.nodeKey;
        return;
    }
    const detailKey = String(node.id || node.uuid || selectedTreeNodeId);
    const sameSelection = elTreeDetails.dataset.nodeKey === detailKey;
    const scrollTop = sameSelection ? (elTreeDetails.querySelector('.leaf-scroll')?.scrollTop || 0) : 0;
    const rawOpen = sameSelection && Boolean(elTreeDetails.querySelector('.tree-raw')?.open);
    elTreeDetails.dataset.nodeKey = detailKey;
    const operation = node.operation || {};
    const displayType = node.action_type || operation.action_type || node.node_type || 'generic';
    const detailRows = [
        ['Name', node.display_name || node.name || 'unknown'],
        ['Type', displayType],
        ['Status', node.status || 'INVALID'],
        ['Children', String((node.children || []).length)],
    ];
    if (node.elapsed_s !== undefined) detailRows.push(['Elapsed', `${Number(node.elapsed_s).toFixed(2)}s`]);
    if (node.duration_s !== undefined) detailRows.push(['Duration', `${Number(node.duration_s).toFixed(2)}s`]);
    if (operation.current_action_name) {
        detailRows.push(['Current Child', `${operation.current_action_name} (${operation.current_action_index || 1}/${operation.total_actions || 1})`]);
    }
    if (operation.current_action_elapsed_s !== undefined) {
        detailRows.push(['Child Elapsed', `${Number(operation.current_action_elapsed_s).toFixed(2)}s`]);
    }

    const rows = detailRows.map(([key, value]) => `
        <div class="tree-detail-row">
            <span>${escapeHtml(key)}</span>
            <strong>${escapeHtml(value)}</strong>
        </div>
    `).join('');
    const rawNode = JSON.stringify(stripTransientTreeFields(node), null, 2);
    elTreeDetails.innerHTML = `
        <button id="close-tree-details" type="button" title="Close node details" aria-label="Close node details">&times;</button><div class="tree-details-title">Node details</div><div class="leaf-scroll">
        ${rows}
        <details class="tree-raw">
            <summary>Raw metadata</summary>
            <pre>${escapeHtml(rawNode)}</pre>
        </details></div>
    `;
    elTreeDetails.querySelector('.leaf-scroll').scrollTop = scrollTop;
    elTreeDetails.querySelector('.tree-raw').open = rawOpen;
    document.getElementById('close-tree-details').onclick = () => { selectedTreeNodeId = null; updateTreeDetails(null); };
    clampPicture(elTreeDetails, document.getElementById('workspace'));
}

function stripTransientTreeFields(node) {
    const copy = { ...node };
    delete copy.__treePath;
    if (copy.children) copy.children = copy.children.map(child => ({id:child.id, name:child.name, node_type:child.node_type, status:child.status}));
    return copy;
}

function escapeHtml(value) {
    return String(value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

// -------------------------------------------------------------
// 11. Diagnostics Widgets and Logs
// -------------------------------------------------------------
function updateDiagnostics(state) {
    if(state.session_id && serverInstance && state.session_id!==serverInstance)return;
    if(Object.prototype.hasOwnProperty.call(state,'task_stage')) {
        taskStage=typeof state.task_stage==='string' ? state.task_stage : null;
        renderTaskStage();
    }
    if(state.scene_highlights){sceneHighlights=state.scene_highlights;drawMap();}
    // Battery Progress ring calculations
    if (state.battery !== undefined) {
        updateBatteryDisplay(state.battery);
    }

    if (state.robot_status) {
        operationStatus = state.robot_status;
        updateAvailability();
    }

    if (state.eta) {
        elLblTaskTime.textContent = state.task_elapsed_s !== undefined
            ? formatDuration(state.task_elapsed_s)
            : state.eta.replace(/^Elapsed\s+/i, '');
    }

    if (state.task_elapsed_s !== undefined) {
        elLblTaskTime.textContent = formatDuration(state.task_elapsed_s);
    }

    const operation = state.operation || {};
    if (operation.elapsed_s !== undefined) {
        elLblCurrentActionTime.textContent = formatDuration(operation.elapsed_s);
    } else {
        elLblCurrentActionTime.textContent = '--';
    }
    if (operation.sequence_label || state.active_node) {
        elLblCurrentActionName.textContent = operation.sequence_label || state.active_node;
    }
    latestActiveNodeName = operation.current_action_name || state.active_node || operation.sequence_label || latestActiveNodeName;
    if (latestTreeData) {
        queueVisual('tree', () => renderBehaviorTree(latestTreeData));
    }

    // The map-frame pose topic is authoritative; diagnostics may use another frame.

}

function formatDuration(seconds) {
    const safeSeconds = Math.max(0, Number(seconds) || 0);
    if (safeSeconds < 60) return `${safeSeconds.toFixed(1)}s`;
    const mins = Math.floor(safeSeconds / 60);
    const secs = Math.floor(safeSeconds % 60);
    return `${mins}m ${secs}s`;
}

const logQueue = [];
setInterval(() => {
    if (document.hidden || !document.body.classList.contains('logs-open') || !logQueue.length) return;
    const atBottom = elLogConsole.scrollHeight - elLogConsole.scrollTop - elLogConsole.clientHeight < 40;
    const fragment = document.createDocumentFragment();
    logQueue.splice(0).forEach(line => fragment.appendChild(line));
    elLogConsole.appendChild(fragment);
    while (elLogConsole.children.length > 500) elLogConsole.firstElementChild.remove();
    if (atBottom) elLogConsole.scrollTop = elLogConsole.scrollHeight;
}, 100);
function appendLog(level, message, timestamp) {
    const timeStr = timestamp ? formatTimestamp(timestamp) : formatCurrentTime();
    const logLine = document.createElement('div');
    logLine.className = `log-line ${level.toLowerCase()}`;

    const timeSpan = document.createElement('span');
    timeSpan.className = 'log-time';
    timeSpan.textContent = `[${timeStr}]`;

    logLine.appendChild(timeSpan);
    logLine.appendChild(document.createTextNode(` [${level.toUpperCase()}] ${message}`));

    logQueue.push(logLine);
    if (logQueue.length > 500) logQueue.shift();
}

function addSystemLog(message) {
    appendLog('system', message);
}

function formatCurrentTime() {
    const d = new Date();
    return d.toTimeString().split(' ')[0];
}

function formatTimestamp(ts) {
    const d = new Date(ts * 1000);
    return d.toTimeString().split(' ')[0];
}

// -------------------------------------------------------------
// 12. Custom Interactive Zoom, Pan, and Theme Helpers
// -------------------------------------------------------------
function initTheme() {
    const storedTheme = localStorage.getItem('theme');
    if (storedTheme === 'light') {
        document.body.classList.add('light-theme');
        const icon = elBtnThemeToggle.querySelector('i');
        icon.className = 'fa-solid fa-sun';
    } else {
        document.body.classList.remove('light-theme');
        const icon = elBtnThemeToggle.querySelector('i');
        icon.className = 'fa-solid fa-moon';
    }
}

function initMapInteraction() {
    let frame=null;
    mapGestures=bindViewport(elMapCanvas,{
        get:()=>({scale:readNumericInput(elMapScale,64),x:readNumericInput(elMapOriginX,0),y:readNumericInput(elMapOriginY,0)}),
        local:event=>{const point=canvasEventPoint(event),m=mapViewMetrics();return rotateDisplayPoint(point,{...m,rotationRad:-m.rotationRad});},
        min:.1,max:600,zoomAllowed:()=>!mapZoomLocked,
        set:view=>{mapAutoFitPending=false;elMapScale.value=String(view.scale);elMapOriginX.value=String(view.x);elMapOriginY.value=String(view.y);
            if(frame===null)frame=requestAnimationFrame(()=>{frame=null;drawMapNow();});},end:saveMapView
    });
    elMapCanvas.addEventListener('click',event=>{if(!mapGestures.suppressClick)selectSceneAtCanvasPoint(canvasEventPoint(event));});
}
function initTreeInteraction() {
    treeGestures=bindViewport(elTreeSvg,{min:.3,max:3,zoomAllowed:()=>!treeZoomLocked,get:()=>({scale:treeZoom,x:treePanX,y:treePanY}),
        set:view=>{treeZoom=view.scale;treePanX=view.x;treePanY=view.y;
            document.getElementById('tree-viewport')?.setAttribute('transform',`translate(${treePanX}, ${treePanY}) scale(${treeZoom})`);}});
}
function initPages() {
    const workspace=document.getElementById('workspace');
    const change=()=>{
        currentPage=['#tree','#tasks'].includes(location.hash)?location.hash.slice(1):'map';document.body.dataset.page=currentPage;
        for(const name of ['map','tree','tasks'])document.getElementById(`page-${name}`).setAttribute('aria-current',name===currentPage?'page':'false');
        visualJobs.clear();Object.keys(subscriptionSpecs).forEach(refreshWidgetSubscription);
        elTreeDetails.hidden=currentPage!=='tree'||!selectedTreeNodeId;
        if(currentPage==='map'){refreshStaticMap();drawMap();}else if(currentPage==='tree'&&latestTreeData)renderBehaviorTree(latestTreeData);
        sizeCamera();
    };
    addEventListener('hashchange',change);change();
    bindPicture(document.getElementById('widget-cam'),workspace);bindPicture(elTreeDetails,workspace);
    const resize=()=>{
        document.documentElement.style.setProperty('--available-height',`${window.visualViewport?.height||innerHeight}px`);
        sizeCamera();if(currentPage==='map')drawMap();
    };
    addEventListener('resize',resize);window.visualViewport?.addEventListener('resize',resize);resize();
    setInterval(async()=>{
        if(document.hidden)return;
        const label=document.getElementById('http-status');
        const began=performance.now();
        try{const response=await fetch('/healthz',{cache:'no-store',signal:AbortSignal.timeout(2500)});
            console.info('[dashboard-http]',{status:response.status,elapsed_ms:Math.round(performance.now()-began)});
            if(!response.ok)throw Error('HTTP error');label.textContent='Dashboard HTTP available';}
        catch(error){console.warn('[dashboard-http]',{error:error.name,elapsed_ms:Math.round(performance.now()-began)});label.textContent='Dashboard HTTP unreachable; bridge status is separate';}
    },10000);
}


// Catalog is retained server-side and cached per bridge/robot/schema; no requests or polling.
function workflowCacheKey() { return `workflow-catalog:1:${catalogBridgeUrl ?? elBridgeUrl.value.trim()}:${currentRobotId}`; }
function validWorkflowCatalog(data) {
    return data?.schema_version === 1 && data.robot_id === currentRobotId && typeof data.revision === 'string' &&
        Array.isArray(data.workflows) && Array.isArray(data.items) && Array.isArray(data.locations);
}
function loadWorkflowCache() {
    catalogBridgeUrl = elBridgeUrl.value.trim();
    workflowCatalog = null;
    try { const data = JSON.parse(localStorage.getItem(workflowCacheKey())); if (validWorkflowCatalog(data)) workflowCatalog = data; } catch {}
    renderWorkflows(Boolean(workflowCatalog));
}
function receiveWorkflowCatalog(message) {
    let data; try { data = JSON.parse(message.data); } catch { return; }
    if (!validWorkflowCatalog(data)) return;
    if (workflowCatalog?.revision === data.revision) {
        workflowCatalog = data;
        try { localStorage.setItem(workflowCacheKey(), JSON.stringify(data)); } catch {}
        document.getElementById('workflow-catalog-status').textContent = 'Workflow catalog ready';
        updateWorkflowButtons(); return;
    }
    workflowCatalog = data;
    try { localStorage.setItem(workflowCacheKey(), JSON.stringify(data)); } catch {}
    renderWorkflows(false);
}
function updateWorkflowButtons() {
    for (const button of document.querySelectorAll('.workflow-launch')) button.disabled = !canSubmitTask() || workflowCatalog?.session_id !== serverInstance || button.dataset.available !== 'true';
}
function renderWorkflows(cached) {
    const cards = document.getElementById('workflow-cards'); if (!cards) return;
    const previous = new Map([...cards.querySelectorAll('select')].map(el => [el.dataset.field, el.value]));
    cards.replaceChildren();
    document.getElementById('workflow-catalog-status').textContent = workflowCatalog ? (cached ? 'Cached choices — connect for live readiness' : 'Workflow catalog ready') : 'Connect to receive available workflows';
    for (const workflow of workflowCatalog?.workflows || []) {
        const form = document.createElement('form'); form.className = 'workflow-card';
        const heading = document.createElement('h3'); heading.textContent = workflow.title; form.append(heading);
        const fields = [];
        for (const field of workflow.fields || []) {
            const label = document.createElement('label'); label.textContent = field.label;
            const input = document.createElement(field.type === 'select' ? 'select' : 'input');
            input.dataset.field = `${workflow.id}.${field.name}`;input.required = Boolean(field.required);
            if (field.type === 'select') {
                input.add(new Option('Choose an item', ''));
                const choices = field.enum ? field.enum.map(value=>({id:String(value),label:String(value)})) : (workflowCatalog[field.choices] || []);
                for (const item of choices) input.add(new Option(item.label, item.id));
                const saved = previous.get(input.dataset.field) ?? field.default;
                if ([...input.options].some(option => option.value === saved)) input.value = saved;
            } else {
                input.type = ['number','checkbox'].includes(field.type) ? field.type : 'text';
                if (field.type === 'checkbox') input.required = false;
                if (field.type === 'checkbox') input.checked = Boolean(field.default);
                else if (field.default !== undefined) input.value = String(field.default);
                if (field.min !== undefined) input.min = field.min;
                if (field.max !== undefined) input.max = field.max;
            }
            label.append(input);form.append(label);fields.push([field.name, input]);
        }
        const reason = document.createElement('p'); reason.className = 'workflow-reason';reason.textContent = workflow.reason || '';
        const button = document.createElement('button');button.type='submit';button.className='btn btn-primary workflow-launch';button.textContent=workflow.title;button.dataset.available=String(workflow.available===true);
        form.append(reason, button);
        form.addEventListener('submit',event=>{event.preventDefault();if(!canSubmitTask()||workflowCatalog?.session_id!==serverInstance||workflow.available!==true||!form.reportValidity())return;
            const inputs=Object.fromEntries(fields.map(([name,input])=>[name,input.type==='checkbox'?input.checked:input.type==='number'?Number(input.value):input.value]));
            const item=workflow.id==='fetch' ? workflowCatalog.items.find(item=>item.id===inputs.item_id) : null;
            submitTaskAction({id:workflow.id,title:item ? `${workflow.title} ${item.label}` : workflow.title,inputs});});
        cards.append(form);
    }
    updateWorkflowButtons();
}
