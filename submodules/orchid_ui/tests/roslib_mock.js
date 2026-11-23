// Browser-only mock: no socket, ROS publisher, or robot command is created.
const topics = new Set(), sockets = [], actions = [];
window.uiMock = {
    sockets, actions, topics, goals: [],
    emitTopic(name, data) { for (const topic of topics) if (topic.name === name) topic.callback?.({data}); },
    heartbeat(state) { this.emitTopic('/hsrb_bot/session_heartbeat', JSON.stringify(state)); },
};
export class Ros {
    constructor() { this.listeners = {}; this.isConnected = false; sockets.push(this); setTimeout(() => {this.isConnected=true;this.emit('connection');}, 10); }
    on(name, cb) { (this.listeners[name] ||= []).push(cb); }
    emit(name, value) { for (const cb of this.listeners[name] || []) cb(value); }
    close() { this.isConnected=false;this.emit('close'); }
}
export class Topic {
    constructor(options) { Object.assign(this, options); }
    subscribe(cb) { this.callback=cb;topics.add(this); }
    unsubscribe() { topics.delete(this); }
}
export class Action {
    constructor(options) { Object.assign(this,options); actions.push(this); }
    sendGoal(args, result, feedback, error) { this.goal={args,result,feedback,error}; window.uiMock.goals.push(args); window.uiMock.lastAction=this; }
    cancelAllGoals() { this.cancelled=true; }
}
