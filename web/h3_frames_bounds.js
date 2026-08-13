import { app } from "../../scripts/app.js";

const NODE_NAME = "WeeToddH3Frames";
const NODE_HEIGHT = 455;
const NODE_WIDTH = 620;

function isFramesNode(node) {
    return (
        node?.type === NODE_NAME ||
        node?.comfyClass === NODE_NAME ||
        node?.constructor?.comfyClass === NODE_NAME ||
        node?.title === "WeeTodd H3 Frames"
    );
}

function compactFramesBounds(node) {
    if (!isFramesNode(node)) return;
    const width = Math.max(Number(node.size?.[0]) || 0, NODE_WIDTH);
    node.setSize?.([width, NODE_HEIGHT]);
    if (node.size) {
        node.size[0] = width;
        node.size[1] = NODE_HEIGHT;
    }
    node.graph?.setDirtyCanvas?.(true, true);
}

app.registerExtension({
    name: "WeeTodd.H3.FramesBoundsV3",
    setup() {
        const apply = () => {
            for (const node of app.graph?._nodes ?? []) compactFramesBounds(node);
        };
        for (const delay of [0, 50, 250, 1000, 2500]) setTimeout(apply, delay);
    },
    nodeCreated(node) {
        compactFramesBounds(node);
    },
});
