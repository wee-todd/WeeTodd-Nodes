import { app } from "../../scripts/app.js";

const GENERATION_CONFIG = "WeeToddH3GenerationConfig";
const RATIO_MODE = "ratio + size";
const EXACT_MODE = "exact dimensions";
const CUSTOM_RATIO = "custom — exact dimensions";
const SIZE_SLIDER = "Use size slider — 32 px steps";
const MODERN_MODES = [RATIO_MODE, EXACT_MODE];
const ALL_RESOLUTION_MODES = new Set([...MODERN_MODES, "preset", "custom"]);
const SEED_CONTROLS = new Set(["fixed", "increment", "decrement", "randomize"]);
const CONFIG_WIDGETS = [
    "duration_seconds",
    "steps",
    "seed",
    "control_after_generate",
    "resolution_mode",
    "resolution_tier",
    "aspect_ratio",
    "short_edge",
    "custom_width",
    "custom_height",
    "drop_adaln",
    "memory_mode",
    "attention_chunk_size",
    "projection_backend",
];

const RESOLUTION_PRESETS = new Map([
    [SIZE_SLIDER, null],
    ["384 px short edge — fast smoke", 384],
    ["480 px short edge — fast preview", 480],
    ["512 px short edge — balanced preview", 512],
    ["576 px short edge — detailed preview", 576],
    ["640 px short edge — quality preview", 640],
    ["672 px short edge — quality preview+", 672],
    ["704 px short edge — high quality", 704],
    ["736 px short edge — near-native", 736],
    ["768 px short edge — native", 768],
    ["896 px short edge — high detail / high memory", 896],
    ["1024 px short edge — very high memory", 1024],
    ["1088 px short edge — maximum slider size", 1088],
]);
const LEGACY_PRESETS = new Map([
    ["384P (fast mode)", 384],
    ["384P (fast smoke)", 384],
    ["512P (balanced)", 512],
    ["640P (quality preview)", 640],
    ["768P (native quality)", 768],
    ["2K (experimental, very high memory)", 1088],
]);
const PRESET_BY_SIZE = new Map(
    [...RESOLUTION_PRESETS].map(([label, shortEdge]) => [shortEdge, label]),
);

const ASPECT_RATIOS = new Map([
    ["21:9 — ultrawide landscape", [21, 9]],
    ["16:9 — widescreen landscape", [16, 9]],
    ["5:3 — wide landscape", [5, 3]],
    ["3:2 — classic landscape", [3, 2]],
    ["4:3 — standard landscape", [4, 3]],
    ["5:4 — near-square landscape", [5, 4]],
    ["1:1 — square", [1, 1]],
    ["4:5 — near-square portrait", [4, 5]],
    ["3:4 — standard portrait", [3, 4]],
    ["2:3 — classic portrait", [2, 3]],
    ["3:5 — tall portrait", [3, 5]],
    ["9:16 — vertical portrait", [9, 16]],
    ["9:21 — ultratall portrait", [9, 21]],
]);
const ASPECT_BY_KEY = new Map(
    [...ASPECT_RATIOS].map(([label]) => [label.split(" — ", 1)[0], label]),
);

function normalizeMode(value) {
    if (value === "preset") return RATIO_MODE;
    if (value === "custom") return EXACT_MODE;
    return MODERN_MODES.includes(value) ? value : RATIO_MODE;
}

function normalizePreset(value) {
    if (RESOLUTION_PRESETS.has(value)) return value;
    const size = LEGACY_PRESETS.get(value);
    return PRESET_BY_SIZE.get(size) ?? "768 px short edge — native";
}

function normalizeAspect(value) {
    if (value === "custom" || value === CUSTOM_RATIO) return CUSTOM_RATIO;
    if (ASPECT_RATIOS.has(value)) return value;
    return ASPECT_BY_KEY.get(value) ?? "16:9 — widescreen landscape";
}

function roundEven(value) {
    const lower = Math.floor(value);
    const fraction = value - lower;
    if (fraction < 0.5) return lower;
    if (fraction > 0.5) return lower + 1;
    return lower % 2 === 0 ? lower : lower + 1;
}

function resolveCanvas(aspectLabel, shortEdge) {
    const ratio = ASPECT_RATIOS.get(normalizeAspect(aspectLabel));
    if (!ratio) return null;
    const [ratioWidth, ratioHeight] = ratio;
    const ratioKey = `${ratioWidth}:${ratioHeight}`;
    if (ratioKey === "16:9" || ratioKey === "9:16") {
        const longEdge = roundEven((shortEdge * 7) / 4 / 32) * 32;
        return ratioKey === "16:9" ? [longEdge, shortEdge] : [shortEdge, longEdge];
    }
    if (ratioWidth >= ratioHeight) {
        return [roundEven((shortEdge * ratioWidth) / ratioHeight / 32) * 32, shortEdge];
    }
    return [shortEdge, roundEven((shortEdge * ratioHeight) / ratioWidth / 32) * 32];
}

function maximumShortEdge(aspectLabel) {
    let maximum = 32;
    for (let candidate = 32; candidate <= 1088; candidate += 32) {
        const canvas = resolveCanvas(aspectLabel, candidate);
        if (canvas && Math.max(...canvas) <= 1920) maximum = candidate;
    }
    return maximum;
}

function matchAspect(width, height) {
    if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
        return CUSTOM_RATIO;
    }
    const actual = width / height;
    let bestLabel = CUSTOM_RATIO;
    let bestDistance = Number.POSITIVE_INFINITY;
    for (const [label, [ratioWidth, ratioHeight]] of ASPECT_RATIOS) {
        const distance = Math.abs(Math.log(actual / (ratioWidth / ratioHeight)));
        if (distance < bestDistance) {
            bestDistance = distance;
            bestLabel = label;
        }
    }
    return bestDistance <= 0.035 ? bestLabel : CUSTOM_RATIO;
}

function moveWidgetAfter(node, widgetName, precedingName) {
    const sourceIndex = node.widgets?.findIndex((widget) => widget.name === widgetName) ?? -1;
    if (sourceIndex < 0) return;
    const [widget] = node.widgets.splice(sourceIndex, 1);
    const precedingIndex = node.widgets.findIndex((candidate) => candidate.name === precedingName);
    node.widgets.splice(precedingIndex + 1, 0, widget);
}

function chainCallback(widget, handler) {
    if (!widget || widget._weeToddResolutionCallback) return;
    const original = widget.callback;
    widget.callback = function callback(value) {
        const result = original?.apply(this, arguments);
        handler(value);
        return result;
    };
    widget._weeToddResolutionCallback = true;
}

function enhanceResolutionWidgets(node) {
    if (!Array.isArray(node.widgets)) return;
    moveWidgetAfter(node, "short_edge", "aspect_ratio");
    const widget = (name) => node.widgets.find((candidate) => candidate.name === name);
    const mode = widget("resolution_mode");
    const preset = widget("resolution_tier");
    const aspect = widget("aspect_ratio");
    const shortEdge = widget("short_edge");
    const width = widget("custom_width");
    const height = widget("custom_height");
    if (![mode, preset, aspect, shortEdge, width, height].every(Boolean)) return;

    mode.options.values = MODERN_MODES;
    preset.options.values = [...RESOLUTION_PRESETS.keys()];
    aspect.options.values = [...ASPECT_RATIOS.keys(), CUSTOM_RATIO];
    preset.label = "size shortcut";
    shortEdge.label = "short edge (32 px steps)";

    const updateLabels = () => {
        const exact = normalizeMode(mode.value) === EXACT_MODE;
        width.label = exact ? "exact width" : "resolved width";
        height.label = exact ? "exact height" : "resolved height";
    };
    const updateExactAspect = () => {
        if (normalizeMode(mode.value) === EXACT_MODE) {
            aspect.value = matchAspect(Number(width.value), Number(height.value));
        }
    };
    const updateCanvas = () => {
        mode.value = normalizeMode(mode.value);
        preset.value = normalizePreset(preset.value);
        aspect.value = normalizeAspect(aspect.value);
        const exact = mode.value === EXACT_MODE;
        shortEdge.disabled = exact;
        shortEdge.options.disabled = exact;
        preset.disabled = exact;
        preset.options.disabled = exact;
        if (exact) {
            shortEdge.options.max = 1088;
            preset.options.values = [...RESOLUTION_PRESETS.keys()];
            updateExactAspect();
        } else {
            if (aspect.value === CUSTOM_RATIO) {
                aspect.value = "16:9 — widescreen landscape";
            }
            const allowedMaximum = maximumShortEdge(aspect.value);
            shortEdge.options.max = allowedMaximum;
            preset.options.values = [...RESOLUTION_PRESETS]
                .filter(([, size]) => size === null || size <= allowedMaximum)
                .map(([label]) => label);
            const snapped = Math.max(
                32,
                Math.min(allowedMaximum, Math.round(Number(shortEdge.value) / 32) * 32),
            );
            shortEdge.value = snapped;
            preset.value = PRESET_BY_SIZE.get(snapped) ?? SIZE_SLIDER;
            const canvas = resolveCanvas(aspect.value, snapped);
            if (canvas) [width.value, height.value] = canvas;
        }
        updateLabels();
        node.setDirtyCanvas?.(true, true);
    };

    chainCallback(mode, () => {
        mode.value = normalizeMode(mode.value);
        if (mode.value === EXACT_MODE) aspect.value = matchAspect(width.value, height.value);
        updateCanvas();
    });
    chainCallback(preset, () => {
        preset.value = normalizePreset(preset.value);
        const shortcut = RESOLUTION_PRESETS.get(preset.value);
        if (shortcut !== null && shortcut !== undefined) shortEdge.value = shortcut;
        mode.value = RATIO_MODE;
        updateCanvas();
    });
    chainCallback(aspect, () => {
        aspect.value = normalizeAspect(aspect.value);
        if (aspect.value === CUSTOM_RATIO) mode.value = EXACT_MODE;
        updateCanvas();
    });
    chainCallback(shortEdge, updateCanvas);
    chainCallback(width, updateExactAspect);
    chainCallback(height, updateExactAspect);
    updateCanvas();
}

export function migrateGenerationConfig(node, info) {
    const saved = info?.widgets_values;
    if (!Array.isArray(saved) || !Array.isArray(node.widgets)) return;

    const hasSeedControl = SEED_CONTROLS.has(saved[3]);
    const modeIndex = hasSeedControl ? 4 : 3;
    let migrated = [...saved];
    if (!ALL_RESOLUTION_MODES.has(migrated[modeIndex])) {
        if (RESOLUTION_PRESETS.has(migrated[modeIndex]) || LEGACY_PRESETS.has(migrated[modeIndex])) {
            migrated.splice(modeIndex, 0, RATIO_MODE);
        } else if (
            Number.isInteger(migrated[modeIndex]) &&
            Number.isInteger(migrated[modeIndex + 1])
        ) {
            migrated = [
                ...migrated.slice(0, modeIndex),
                EXACT_MODE,
                "768 px short edge — native",
                CUSTOM_RATIO,
                ...migrated.slice(modeIndex),
            ];
        } else {
            return;
        }
    }

    migrated[modeIndex] = normalizeMode(migrated[modeIndex]);
    migrated[modeIndex + 1] = normalizePreset(migrated[modeIndex + 1]);
    migrated[modeIndex + 2] =
        migrated[modeIndex] === EXACT_MODE
            ? CUSTOM_RATIO
            : normalizeAspect(migrated[modeIndex + 2]);

    const sizeIndex = modeIndex + 3;
    const hasShortEdge =
        Number.isInteger(migrated[sizeIndex]) &&
        Number.isInteger(migrated[sizeIndex + 1]) &&
        Number.isInteger(migrated[sizeIndex + 2]);
    if (!hasShortEdge) {
        const presetSize = RESOLUTION_PRESETS.get(migrated[modeIndex + 1]);
        const exactSize = Math.min(migrated[sizeIndex] ?? 768, migrated[sizeIndex + 1] ?? 768);
        migrated.splice(sizeIndex, 0, migrated[modeIndex] === EXACT_MODE ? exactSize : presetSize);
    }

    const widgetNames = hasSeedControl
        ? CONFIG_WIDGETS
        : CONFIG_WIDGETS.filter((name) => name !== "control_after_generate");
    for (let index = 0; index < widgetNames.length; index += 1) {
        const target = node.widgets.find((candidate) => candidate.name === widgetNames[index]);
        if (target && migrated[index] !== undefined) target.value = migrated[index];
    }
    enhanceResolutionWidgets(node);
    console.info("WeeTodd H3: migrated a legacy Generation Config workflow.");
}

app.registerExtension({
    name: "WeeTodd.H3.ResolutionControls",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== GENERATION_CONFIG) return;
        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function onNodeCreated() {
            const result = originalOnNodeCreated?.apply(this, arguments);
            enhanceResolutionWidgets(this);
            return result;
        };
        const originalOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function onConfigure(info) {
            const result = originalOnConfigure?.apply(this, arguments);
            migrateGenerationConfig(this, info);
            return result;
        };
    },
});
