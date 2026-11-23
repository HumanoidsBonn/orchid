import {load,JSON_SCHEMA} from './vendor/js-yaml/js-yaml-4.1.1.mjs';

export function parseMapMetadata(text) {
    const value=load(text,{schema:JSON_SCHEMA});
    if(!value || !Number.isFinite(value.resolution) || value.resolution<=0 ||
       !Array.isArray(value.origin) || value.origin.length!==3 || !value.origin.every(Number.isFinite)) {
        throw new Error('Map YAML requires positive resolution and finite [x, y, yaw] origin');
    }
    const [x,y,yaw]=value.origin;
    return {resolution:value.resolution,origin:{position:{x,y,z:0},orientation:{x:0,y:0,z:Math.sin(yaw/2),w:Math.cos(yaw/2)}}};
}

export async function loadStaticMap(pngPath,yamlPath,{signal,baseUrl=location.href,fetcher=fetch}={}) {
    const [png,yaml]=await Promise.all([pngPath,yamlPath].map(path=>fetcher(new URL(path,baseUrl).href,{signal,cache:'no-cache'})));
    if(!png.ok || !yaml.ok)throw new Error(`Static map HTTP error: PNG ${png.status}, YAML ${yaml.status}`);
    const metadata=parseMapMetadata(await yaml.text());
    const blob=await png.blob();
    const signature=new Uint8Array(await blob.slice(0,8).arrayBuffer());
    if(signature.join(',')!=='137,80,78,71,13,10,26,10')throw new Error('Static map image must be PNG');
    const url=URL.createObjectURL(blob);
    try {
        const image=new Image();image.src=url;await image.decode();
        if(signal?.aborted)throw new DOMException('Map load cancelled','AbortError');
        const canvas=document.createElement('canvas');canvas.width=image.naturalWidth;canvas.height=image.naturalHeight;
        if(!canvas.width || !canvas.height || canvas.width*canvas.height>32000000)throw new Error('Unsupported static map dimensions');
        canvas.getContext('2d').drawImage(image,0,0);
        return {grid:{width:canvas.width,height:canvas.height,...metadata},canvas};
    } finally {URL.revokeObjectURL(url);}
}
