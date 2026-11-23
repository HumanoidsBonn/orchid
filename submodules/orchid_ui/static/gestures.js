// Affine CSS-pixel gesture controller shared by map and tree. No page zoom suppression outside the viewport.
export function bindViewport(element, {get, set, local = p => ({x:p.clientX-element.getBoundingClientRect().left,y:p.clientY-element.getBoundingClientRect().top}), min=.1, max=600, zoomAllowed=()=>true, end=()=>{}}) {
    const pointers=new Map();let baseline=null, moved=false, blockedUntil=0, frame=null, pending=null;
    const sample=()=>{
        const points=[...pointers.values()];
        const center=points.length>1?{x:(points[0].x+points[1].x)/2,y:(points[0].y+points[1].y)/2}:points[0];
        return {center,distance:points.length>1?Math.hypot(points[1].x-points[0].x,points[1].y-points[0].y):0};
    };
    const flush=()=>{frame=null;if(pending){set(pending);pending=null;}};
    const update=v=>{pending=v;if(frame===null)frame=requestAnimationFrame(flush);};
    const rebase=()=>{if(pending)flush();baseline=pointers.size?{...sample(),view:{...get()}}:null;};
    element.addEventListener('pointerdown',e=>{
        if(e.pointerType==='mouse' && e.button!==0)return;
        if(!pointers.size)moved=false;
        pointers.set(e.pointerId,local(e));
        if(pointers.size>1)moved=true;rebase();
    });
    element.addEventListener('pointermove',e=>{
        if(!pointers.has(e.pointerId)||!baseline)return;
        e.preventDefault();pointers.set(e.pointerId,local(e));const now=sample(),start=baseline;
        const scale=Math.min(max,Math.max(min,start.view.scale*(zoomAllowed()&&start.distance&&now.distance?now.distance/start.distance:1)));
        if(Math.hypot(now.center.x-start.center.x,now.center.y-start.center.y)>5||now.distance){
            moved=true;try{element.setPointerCapture(e.pointerId);}catch{}
        }
        update({scale,x:now.center.x-(start.center.x-start.view.x)*scale/start.view.scale,y:now.center.y-(start.center.y-start.view.y)*scale/start.view.scale});
    });
    const release=e=>{if(!pointers.has(e.pointerId))return;pointers.delete(e.pointerId);if(moved)blockedUntil=performance.now()+350;rebase();if(!pointers.size)end();};
    for(const name of ['pointerup','pointercancel','lostpointercapture'])element.addEventListener(name,release);
    element.addEventListener('click',e=>{if(performance.now()<blockedUntil){e.preventDefault();e.stopImmediatePropagation();}},true);
    element.addEventListener('wheel',e=>{
        e.preventDefault();if(!zoomAllowed())return;const p=local(e),v=get(),scale=Math.min(max,Math.max(min,v.scale*Math.exp(-e.deltaY*.0015)));
        update({scale,x:p.x-(p.x-v.x)*scale/v.scale,y:p.y-(p.y-v.y)*scale/v.scale});
    },{passive:false});
    return {get active(){return pointers.size>0;},get suppressClick(){return performance.now()<blockedUntil;}};
}

export function bindPicture(element, workspace) {
    let drag=null;
    element.addEventListener('pointerdown',e=>{
        if(e.target.closest('button,textarea,input,select,a,summary,.leaf-scroll'))return;
        const rect=element.getBoundingClientRect(),parent=workspace.getBoundingClientRect();
        drag={id:e.pointerId,x:e.clientX,y:e.clientY,left:rect.left-parent.left,top:rect.top-parent.top};
        element.setPointerCapture(e.pointerId);e.preventDefault();e.stopPropagation();
    });
    element.addEventListener('pointermove',e=>{
        if(!drag||drag.id!==e.pointerId)return;
        element.style.right='auto';element.style.bottom='auto';
        element.style.left=`${drag.left+e.clientX-drag.x}px`;
        element.style.top=`${drag.top+e.clientY-drag.y}px`;
        clampPicture(element, workspace);
        e.stopPropagation();
    });
    for(const name of ['pointerup','pointercancel','lostpointercapture'])element.addEventListener(name,()=>{drag=null;});
}

// Keep floating windows inside the visible framed workspace, including after resizing.
export function clampPicture(element, workspace) {
    if (!element.style.left || element.hidden) return;
    const inset = (parseFloat(getComputedStyle(workspace).getPropertyValue('--workspace-inset')) || 0) + 8;
    element.style.left = `${Math.max(inset,Math.min(parseFloat(element.style.left),workspace.clientWidth-element.offsetWidth-inset))}px`;
    element.style.top = `${Math.max(inset,Math.min(parseFloat(element.style.top),workspace.clientHeight-element.offsetHeight-inset))}px`;
}
