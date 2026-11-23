// Pure display projection. The full tree remains available for details/search.
export function projectTree(root, depth, expanded, collapsed = new Set()) {
    const nodes = [], active = new Set();
    const visitActive = (node, path, ancestors) => {
        const children = node.children || [];
        if (node.status === 'RUNNING' && !children.some(c => c.status === 'RUNNING')) {
            [...ancestors, path].forEach(p => active.add(p));
        }
        children.forEach((c, i) => visitActive(c, `${path}.${i}`, [...ancestors, path]));
    };
    visitActive(root, '0', []);
    const count = n => (n.children || []).reduce((s, c) => s + 1 + count(c), 0);
    const walk = (node, path, level, parent) => {
        const children = node.children || [];
        const open = children.length && !collapsed.has(path) && (active.has(path) || level < depth || expanded.has(path));
        const item = { node, path, level, parent, open, hidden: open ? 0 : count(node), active: active.has(path) };
        nodes.push(item);
        if (open) children.forEach((c, i) => walk(c, `${path}.${i}`, level + 1, path));
    };
    walk(root, '0', 0, null);
    return nodes;
}

export function topologyKey(root) {
    const parts = [];
    function visit(node, path) {
        parts.push([path, node.id || node.uuid || '', node.name, node.node_type, (node.children || []).length]);
        (node.children || []).forEach((child, index) => visit(child, `${path}.${index}`));
    }
    visit(root, '0');
    return JSON.stringify(parts);
}
