import { app } from "../../scripts/app.js";

const NODE_NAME = "WeeToddH3TimedKeyframeEncode";
const PROMPT_HEIGHT = 160;
const MINIMUM_NODE_WIDTH = 480;

function isTimedEncoder(node) {
    return (
        node?.type === NODE_NAME ||
        node?.comfyClass === NODE_NAME ||
        node?.constructor?.comfyClass === NODE_NAME
    );
}

function enhanceTimedEncoder(node) {
    if (!isTimedEncoder(node) || node._weeToddTimedEncoderUI) return;
    const promptWidget = node.widgets?.find((widget) => widget.name === "prompt");
    if (!promptWidget) return;
    node._weeToddTimedEncoderUI = true;
    promptWidget.label = "PROMPT";

    const fitNodeToContent = () => {
        requestAnimationFrame(() => {
            if (node._weeToddTimedEncoderSizing) return;
            node._weeToddTimedEncoderSizing = true;
            const computed = node.computeSize?.() ?? node.size ?? [MINIMUM_NODE_WIDTH, 220];
            const width = Math.max(Number(node.size?.[0]) || 0, MINIMUM_NODE_WIDTH);
            const height = Math.max(Number(computed?.[1]) || 0, PROMPT_HEIGHT + 60);
            node.setSize?.([width, height]);
            if (node.size) {
                node.size[0] = width;
                node.size[1] = height;
            }
            node._weeToddTimedEncoderSizing = false;
            node.graph?.setDirtyCanvas?.(true, true);
        });
    };

    promptWidget.options = {
        ...(promptWidget.options ?? {}),
        getMinHeight: () => PROMPT_HEIGHT,
        getMaxHeight: () => PROMPT_HEIGHT,
        getHeight: () => PROMPT_HEIGHT,
        afterResize: fitNodeToContent,
    };
    if (promptWidget.element) {
        promptWidget.element.setAttribute("aria-label", "H3 generation prompt");
        promptWidget.element.style.minHeight = `${PROMPT_HEIGHT}px`;
        promptWidget.element.style.maxHeight = `${PROMPT_HEIGHT}px`;
    }
    for (const delay of [0, 50, 250, 1000]) setTimeout(fitNodeToContent, delay);
}

app.registerExtension({
    name: "WeeTodd.H3.TimedEncoderUIV1",
    setup() {
        const apply = () => {
            for (const node of app.graph?._nodes ?? []) enhanceTimedEncoder(node);
        };
        for (const delay of [0, 50, 250, 1000]) setTimeout(apply, delay);
    },
    nodeCreated(node) {
        enhanceTimedEncoder(node);
    },
});
