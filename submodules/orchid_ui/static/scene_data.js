// Scene semantics come exclusively from the graph's referenced YAML file.
import {load, JSON_SCHEMA} from './vendor/js-yaml/js-yaml-4.1.1.mjs';

export function sceneObjectCategory(object, classes) {
    const label = object?.label || '';
    if (classes.furniture.has(label)) return 'furniture';
    if (classes.structural.has(label)) return 'structural';
    return 'movable';
}

// One filter for both the displayed count and arrows. Do not infer validity
// from ranking or diagnostic candidate entries.
export function validVisitPoses(object) {
    return (Array.isArray(object?.base_access_poses) ? object.base_access_poses : []).flatMap(pose => {
        if (pose?.valid !== true || (pose.frame_id || 'map') !== 'map') return [];
        const {x, y} = pose.position || {};
        const q = pose.orientation;
        if (![x,y,q?.x,q?.y,q?.z,q?.w].every(Number.isFinite)) return [];
        const norm = Math.hypot(q.x,q.y,q.z,q.w);
        if (norm < 1e-9) return [];
        const qx=q.x/norm,qy=q.y/norm,qz=q.z/norm,qw=q.w/norm;
        const yaw=Math.atan2(2*(qw*qz+qx*qy),1-2*(qy*qy+qz*qz));
        return [{x,y,yaw,id:pose.pose_id}];
    });
}

export async function loadSceneDataset(path, {baseUrl = location.href, fetcher = fetch, signal} = {}) {
    const url = new URL(path, baseUrl);
    const response = await fetcher(url.href, {cache:'no-cache', signal});
    if (!response.ok) throw new Error(`Scene graph ${url.pathname}: HTTP ${response.status}`);
    const graph = await response.json();
    const frame = graph?.metadata?.world_frame || 'map';
    if (frame !== 'map') throw new Error(`Scene graph world frame '${frame}' is not map`);
    if (!Array.isArray(graph.objects)) throw new Error('Scene graph objects list missing');
    const reference = graph.metadata?.relationship_config;
    if (typeof reference !== 'string' || !reference.trim()) throw new Error('Scene graph metadata.relationship_config missing');
    const relationshipUrl = new URL(reference, url);
    const relationships = await fetcher(relationshipUrl.href, {cache:'no-cache', signal});
    if (!relationships.ok) throw new Error(`Relationships ${relationshipUrl.pathname}: HTTP ${relationships.status}`);
    const configuration = load(await relationships.text(), {schema:JSON_SCHEMA});
    for (const key of ['static_furniture', 'null_classes']) {
        if (!Array.isArray(configuration?.[key]) || !configuration[key].every(label => typeof label === 'string')) {
            throw new Error(`Relationships ${relationshipUrl.pathname}: ${key} must be a list of strings`);
        }
    }
    return {graph, classes:{furniture:new Set(configuration.static_furniture), structural:new Set(configuration.null_classes)}};
}
