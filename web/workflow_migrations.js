import { app } from "../../scripts/app.js";

const GENERATION_CONFIG = "WeeToddH3GenerationConfig";
const RESOLUTION_MODES = new Set(["preset", "custom"]);
const SEED_CONTROLS = new Set(["fixed", "increment", "decrement", "randomize"]);
const CONFIG_WIDGETS = [
    "duration_seconds",
    "steps",
    "seed",
    "control_after_generate",
    "resolution_mode",
    "resolution_tier",
    "aspect_ratio",
    "custom_width",
    "custom_height",
    "drop_adaln",
    "memory_mode",
    "attention_chunk_size",
    "projection_backend",
];
const RESOLUTION_TIERS = new Map([
    ["384P (fast mode)", "384P (fast smoke)"],
    ["384P (fast smoke)", "384P (fast smoke)"],
    ["512P (balanced)", "512P (balanced)"],
    ["640P (quality preview)", "640P (quality preview)"],
    ["768P (native quality)", "768P (native quality)"],
    ["2K (experimental, very high memory)", "2K (experimental, very high memory)"],
]);

export function migrateGenerationConfig(node, info) {
    const saved = info?.widgets_values;
    if (!Array.isArray(saved) || !Array.isArray(node.widgets)) {
        return;
    }

    const hasSeedControl = SEED_CONTROLS.has(saved[3]);
    const modeIndex = hasSeedControl ? 4 : 3;
    if (RESOLUTION_MODES.has(saved[modeIndex])) {
        return;
    }

    let migrated;
    if (RESOLUTION_TIERS.has(saved[modeIndex])) {
        migrated = [...saved];
        migrated.splice(modeIndex, 0, "preset");
        migrated[modeIndex + 1] = RESOLUTION_TIERS.get(saved[modeIndex]);
    } else if (
        Number.isInteger(saved[modeIndex]) &&
        Number.isInteger(saved[modeIndex + 1])
    ) {
        migrated = [
            ...saved.slice(0, modeIndex),
            "custom",
            "768P (native quality)",
            "16:9",
            ...saved.slice(modeIndex),
        ];
    } else {
        return;
    }

    const widgetNames = hasSeedControl
        ? CONFIG_WIDGETS
        : CONFIG_WIDGETS.filter((name) => name !== "control_after_generate");
    for (let index = 0; index < widgetNames.length; index += 1) {
        const widget = node.widgets.find((candidate) => candidate.name === widgetNames[index]);
        if (widget && migrated[index] !== undefined) {
            widget.value = migrated[index];
        }
    }
    console.info("WeeTodd H3: migrated a legacy Generation Config workflow.");
}

app.registerExtension({
    name: "WeeTodd.H3.WorkflowMigrations",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== GENERATION_CONFIG) {
            return;
        }
        const originalOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function onConfigure(info) {
            const result = originalOnConfigure?.apply(this, arguments);
            migrateGenerationConfig(this, info);
            return result;
        };
    },
});
