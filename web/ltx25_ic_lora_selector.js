import { app } from "../../scripts/app.js";

const LTX25_IC_LORA_LOADER = "WeeToddLTX25ICLoRALoader";

function enhanceICLoRASelector(node, nodeData) {
    if (!Array.isArray(node.widgets)) return;
    const widget = node.widgets.find((candidate) => candidate.name === "ic_lora");
    const discovered = nodeData?.input?.required?.ic_lora?.[0];
    if (!widget || !Array.isArray(discovered) || discovered.length === 0) return;

    // Older workflows may have serialized this input while it was a free-form STRING.
    // Comfy can retain that text widget even after the backend changes the contract to a
    // combo. Upgrade the live widget so existing graphs receive the installed-model menu.
    const current = typeof widget.value === "string" ? widget.value : "";
    const values = [...new Set([...discovered, ...(current ? [current] : [])])];
    widget.type = "combo";
    widget.label = "installed IC-LoRA";
    widget.options = { ...(widget.options ?? {}), values };
    if (!current) widget.value = values[0];
    node.setDirtyCanvas?.(true, true);
}

app.registerExtension({
    name: "WeeTodd.LTX25.ICLoRASelectorMigration",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== LTX25_IC_LORA_LOADER) return;
        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function onNodeCreated() {
            const result = originalOnNodeCreated?.apply(this, arguments);
            enhanceICLoRASelector(this, nodeData);
            return result;
        };
        const originalOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function onConfigure(info) {
            const result = originalOnConfigure?.apply(this, arguments);
            enhanceICLoRASelector(this, nodeData);
            return result;
        };
    },
});
