import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_NAME = "WeeToddH3Frames";
const MAX_FRAMES = 8;
const MINIMUM_NODE_WIDTH = 620;
const EDITOR_NODE_HEIGHT = 455;

function applyCompactNodeSize(node) {
    if (!node) return;
    const width = Math.max(Number(node.size?.[0]) || 0, MINIMUM_NODE_WIDTH);
    node.setSize?.([width, EDITOR_NODE_HEIGHT]);
    if (node.size) {
        node.size[0] = width;
        node.size[1] = EDITOR_NODE_HEIGHT;
    }
    node.graph?.setDirtyCanvas?.(true, true);
}

function alignedFrameCount(durationSeconds) {
    let frames = Math.max(1, Math.round(Number(durationSeconds) * 24));
    while (frames % 17 !== 5) frames += 1;
    return frames;
}

function connectedLastFrame(node) {
    const input = node.inputs?.find((candidate) => candidate.name === "config");
    const link = input?.link == null ? null : app.graph?.links?.[input.link];
    const origin = link ? app.graph?.getNodeById?.(link.origin_id) : null;
    const duration = origin?.widgets?.find((widget) => widget.name === "duration_seconds")?.value;
    return Number.isFinite(Number(duration)) ? alignedFrameCount(duration) : null;
}

function manifestEntries(value) {
    try {
        const parsed = JSON.parse(value || "[]");
        if (Array.isArray(parsed) && parsed.length) {
            return parsed.slice(0, MAX_FRAMES).map((entry, index) => ({
                id: crypto.randomUUID?.() ?? `${Date.now()}-${index}`,
                role: ["first", "middle", "last"].includes(entry.role)
                    ? entry.role
                    : "middle",
                frame: Number.isInteger(entry.frame) ? entry.frame : 2,
                image: typeof entry.image === "string" ? entry.image : "",
            }));
        }
    } catch (error) {
        console.warn("WeeTodd H3 Frames: ignored malformed saved editor data.", error);
    }
    return [
        { id: "first", role: "first", frame: 1, image: "" },
        { id: "last", role: "last", frame: null, image: "" },
    ];
}

function serializedEntries(entries) {
    return entries.map(({ role, frame, image }) => ({
        role,
        ...(role === "middle" ? { frame } : {}),
        image,
    }));
}

function imageUrl(imageName) {
    const normalized = imageName.replaceAll("\\", "/");
    const parts = normalized.split("/");
    const filename = parts.pop();
    const params = new URLSearchParams({
        filename,
        subfolder: parts.join("/"),
        type: "input",
        rand: String(Date.now()),
    });
    return api.apiURL(`/view?${params.toString()}`);
}

async function uploadImage(file) {
    const body = new FormData();
    body.append("image", file);
    const response = await api.fetchApi("/upload/image", { method: "POST", body });
    if (!response.ok) throw new Error(`Image upload failed (${response.status}).`);
    const result = await response.json();
    return result.subfolder ? `${result.subfolder}/${result.name}` : result.name;
}

function makeButton(label, className, callback) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = className;
    button.textContent = label;
    button.addEventListener("click", callback);
    return button;
}

function enhanceFramesNode(node) {
    if (node._weeToddFramesEditor || !Array.isArray(node.widgets)) return;
    const manifestWidget = node.widgets.find((widget) => widget.name === "frame_manifest");
    if (!manifestWidget) return;
    node._weeToddFramesEditor = true;

    manifestWidget.computeSize = () => [0, -4];
    manifestWidget.type = "hidden";
    manifestWidget.hidden = true;
    manifestWidget.options = {
        ...(manifestWidget.options ?? {}),
        hidden: true,
        getMinHeight: () => 0,
        getMaxHeight: () => 0,
        getHeight: () => 0,
    };
    if (manifestWidget.element) {
        manifestWidget.element.hidden = true;
        manifestWidget.element.setAttribute("aria-hidden", "true");
        manifestWidget.element.style.display = "none";
        manifestWidget.element.style.pointerEvents = "none";
    }
    let entries = manifestEntries(manifestWidget.value);
    let selectedId = entries[0].id;
    let lastFrame = connectedLastFrame(node);

    const root = document.createElement("div");
    root.className = "weetodd-h3-frames";
    root.innerHTML = `
      <style>
        .weetodd-h3-frames { box-sizing:border-box; width:100%; height:350px; padding:8px;
          display:grid; grid-template-columns:150px minmax(250px,1fr); grid-template-rows:1fr 82px;
          gap:8px; color:var(--fg-color, #ddd); background:var(--comfy-menu-bg, #202020);
          border:1px solid var(--border-color, #444); border-radius:4px; overflow:hidden;
          font:12px Arial, sans-serif; }
        .weetodd-h3-frame-list { min-width:0; overflow:auto; background:var(--bg-color, #181818);
          border:1px solid var(--border-color, #444); border-radius:4px; padding:7px; }
        .weetodd-h3-frame-list h3 { margin:0 0 6px; color:var(--fg-color, #ddd);
          font:600 12px Arial, sans-serif; letter-spacing:.04em; }
        .weetodd-h3-frame-row { width:100%; display:grid; grid-template-columns:48px 1fr 22px;
          align-items:center; gap:4px; padding:4px; margin:2px 0; color:inherit; background:transparent;
          border:1px solid transparent; border-radius:3px; cursor:pointer; text-align:left; }
        .weetodd-h3-frame-row:hover { background:var(--input-bg, #2b2b2b); }
        .weetodd-h3-frame-row.is-selected { border-color:var(--p-button-primary-border-color, #4d8ed8);
          background:color-mix(in srgb, var(--p-button-primary-background, #2f6ea9) 35%, transparent); }
        .weetodd-h3-frame-row input { width:46px; box-sizing:border-box; color:var(--input-text, #ddd);
          background:var(--input-bg, #2b2b2b); border:1px solid var(--border-color, #555);
          border-radius:3px; padding:3px; }
        .weetodd-h3-frame-role { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
        .weetodd-h3-frame-remove { border:0; background:transparent;
          color:var(--error-text, #d88); cursor:pointer; }
        .weetodd-h3-add { width:100%; margin-top:6px; padding:5px;
          border:1px solid var(--border-color, #555); border-radius:3px;
          color:var(--fg-color, #ddd); background:var(--input-bg, #2b2b2b); cursor:pointer; }
        .weetodd-h3-add:hover { filter:brightness(1.15); }
        .weetodd-h3-preview { position:relative; min-width:0; min-height:0; display:flex; align-items:center;
          justify-content:center; overflow:hidden; border:1px solid var(--border-color, #444);
          border-radius:4px; background:var(--bg-color, #181818); cursor:pointer; }
        .weetodd-h3-preview img { width:100%; height:100%; object-fit:contain; }
        .weetodd-h3-preview-empty { color:var(--descrip-text, #999); text-align:center; padding:12px; }
        .weetodd-h3-filmstrip { grid-column:1 / -1; display:flex; align-items:stretch; gap:10px;
          overflow-x:auto; overflow-y:hidden; padding:2px 0 3px; }
        .weetodd-h3-thumb { position:relative; flex:0 0 112px; overflow:hidden;
          border:1px solid var(--border-color, #555); border-radius:4px;
          background:var(--bg-color, #181818); cursor:pointer; padding:0; }
        .weetodd-h3-thumb.is-selected { border:2px solid var(--p-button-primary-border-color, #4d8ed8); }
        .weetodd-h3-thumb img { width:100%; height:100%; object-fit:cover; }
        .weetodd-h3-thumb-label { position:absolute; left:5px; bottom:4px; padding:2px 5px;
          border-radius:2px; background:#000b; color:#fff; font:600 10px Arial, sans-serif; }
        .weetodd-h3-thumb-empty { color:var(--descrip-text, #999); font:11px Arial, sans-serif; }
      </style>
      <section class="weetodd-h3-frame-list"><h3>FRAMES</h3><div data-rows></div></section>
      <section class="weetodd-h3-preview" title="Click to choose or replace the selected image"></section>
      <section class="weetodd-h3-filmstrip" aria-label="Selected frame images"></section>`;
    const list = root.querySelector("[data-rows]");
    const listPanel = root.querySelector(".weetodd-h3-frame-list");
    const preview = root.querySelector(".weetodd-h3-preview");
    const filmstrip = root.querySelector(".weetodd-h3-filmstrip");
    const fileInput = document.createElement("input");
    fileInput.type = "file";
    fileInput.accept = "image/*";
    fileInput.hidden = true;
    root.append(fileInput);

    const frameLabel = (entry) => {
        if (entry.role === "first") return { number: "001", role: "First" };
        if (entry.role === "last") {
            return { number: lastFrame ? String(lastFrame).padStart(3, "0") : "END", role: "Last" };
        }
        return { number: String(entry.frame).padStart(3, "0"), role: "Middle" };
    };
    const selectedEntry = () => entries.find((entry) => entry.id === selectedId) ?? entries[0];
    const sync = () => {
        const previous = manifestWidget.value;
        manifestWidget.value = JSON.stringify(serializedEntries(entries));
        node.onWidgetChanged?.(
            manifestWidget.name,
            manifestWidget.value,
            previous,
            manifestWidget,
        );
        node.graph?.setDirtyCanvas?.(true, true);
    };
    const chooseImage = () => fileInput.click();
    const fitNodeSize = () => {
        requestAnimationFrame(() => {
            if (node._weeToddFramesSizing) return;
            const width = Math.max(Number(node.size?.[0]) || 0, MINIMUM_NODE_WIDTH);
            if (
                Math.abs(width - Number(node.size?.[0])) < 0.5 &&
                Math.abs(EDITOR_NODE_HEIGHT - Number(node.size?.[1])) < 0.5
            ) {
                return;
            }
            node._weeToddFramesSizing = true;
            applyCompactNodeSize(node);
            node._weeToddFramesSizing = false;
        });
    };
    const settleNodeSize = () => {
        // Workflow deserialization restores `size` in more than one phase in
        // current ComfyUI. Re-apply the content height after each phase so old
        // workflows cannot retain the original oversized editor rectangle.
        for (const delay of [0, 50, 250, 1000]) {
            setTimeout(fitNodeSize, delay);
        }
    };

    function render() {
        list.replaceChildren();
        filmstrip.replaceChildren();
        listPanel.querySelector(".weetodd-h3-add")?.remove();
        for (const entry of entries) {
            const label = frameLabel(entry);
            const row = document.createElement("div");
            row.role = "button";
            row.tabIndex = 0;
            row.className = `weetodd-h3-frame-row${entry.id === selectedId ? " is-selected" : ""}`;
            if (entry.role === "middle") {
                const input = document.createElement("input");
                input.type = "number";
                input.min = "2";
                input.max = lastFrame ? String(lastFrame - 1) : "9999";
                input.step = "1";
                input.value = String(entry.frame);
                input.title = "One-based frame number";
                input.addEventListener("click", (event) => event.stopPropagation());
                input.addEventListener("change", () => {
                    entry.frame = Math.max(2, Math.round(Number(input.value) || 2));
                    sync();
                    render();
                });
                row.append(input);
            } else {
                const number = document.createElement("span");
                number.textContent = label.number;
                row.append(number);
            }
            const role = document.createElement("span");
            role.className = "weetodd-h3-frame-role";
            role.textContent = label.role;
            row.append(role);
            if (entry.role === "middle") {
                row.append(makeButton("×", "weetodd-h3-frame-remove", (event) => {
                    event.stopPropagation();
                    entries = entries.filter((candidate) => candidate.id !== entry.id);
                    selectedId = entries[0].id;
                    sync();
                    render();
                }));
            } else {
                row.append(document.createElement("span"));
            }
            row.addEventListener("click", () => {
                selectedId = entry.id;
                render();
            });
            row.addEventListener("keydown", (event) => {
                if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    selectedId = entry.id;
                    render();
                }
            });
            list.append(row);

            const thumb = makeButton("", `weetodd-h3-thumb${entry.id === selectedId ? " is-selected" : ""}`, () => {
                selectedId = entry.id;
                render();
            });
            if (entry.image) {
                const image = document.createElement("img");
                image.src = imageUrl(entry.image);
                image.alt = `${label.role} frame`;
                thumb.append(image);
            } else {
                const empty = document.createElement("span");
                empty.className = "weetodd-h3-thumb-empty";
                empty.textContent = "+ Choose image";
                thumb.append(empty);
                thumb.addEventListener("dblclick", chooseImage);
            }
            const caption = document.createElement("span");
            caption.className = "weetodd-h3-thumb-label";
            caption.textContent = `${label.number} ${label.role}`;
            thumb.append(caption);
            filmstrip.append(thumb);
        }
        const add = makeButton("+ Add Frame", "weetodd-h3-add", () => {
            if (entries.length >= MAX_FRAMES) return;
            const used = new Set(entries.filter((entry) => entry.role === "middle").map((entry) => entry.frame));
            const limit = lastFrame ?? 121;
            let candidate = Math.max(2, Math.round(limit / 2));
            while (used.has(candidate) && candidate < limit - 1) candidate += 1;
            const entry = {
                id: crypto.randomUUID?.() ?? String(Date.now()),
                role: "middle",
                frame: candidate,
                image: "",
            };
            const lastIndex = entries.findIndex((item) => item.role === "last");
            entries.splice(lastIndex < 0 ? entries.length : lastIndex, 0, entry);
            selectedId = entry.id;
            sync();
            render();
            chooseImage();
        });
        add.disabled = entries.length >= MAX_FRAMES;
        listPanel.append(add);

        preview.replaceChildren();
        const selected = selectedEntry();
        if (selected?.image) {
            const image = document.createElement("img");
            image.src = imageUrl(selected.image);
            image.alt = "Selected H3 frame preview";
            preview.append(image);
        } else {
            const empty = document.createElement("div");
            empty.className = "weetodd-h3-preview-empty";
            empty.textContent = "Select a frame, then click here to choose its image";
            preview.append(empty);
        }
        fitNodeSize();
    }

    fileInput.addEventListener("change", async () => {
        const file = fileInput.files?.[0];
        if (!file) return;
        try {
            selectedEntry().image = await uploadImage(file);
            sync();
            render();
        } catch (error) {
            alert(`WeeTodd H3 Frames: ${error.message}`);
        } finally {
            fileInput.value = "";
        }
    });
    preview.addEventListener("click", chooseImage);
    node.addDOMWidget("frames_editor", "div", root, {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: () => 350,
        getMaxHeight: () => 350,
        getHeight: () => 350,
        afterResize(currentNode) {
            if (
                !currentNode._weeToddFramesSizing &&
                Math.abs(Number(currentNode.size?.[1]) - EDITOR_NODE_HEIGHT) > 0.5
            ) {
                currentNode._weeToddFramesSizing = true;
                applyCompactNodeSize(currentNode);
                currentNode._weeToddFramesSizing = false;
            }
        },
    });
    fitNodeSize();

    const originalConnectionsChange = node.onConnectionsChange;
    node.onConnectionsChange = function onConnectionsChange() {
        const result = originalConnectionsChange?.apply(this, arguments);
        const updated = connectedLastFrame(node);
        if (updated !== lastFrame) {
            lastFrame = updated;
            render();
        }
        return result;
    };
    node._weeToddFramesReload = () => {
        entries = manifestEntries(manifestWidget.value);
        selectedId = entries[0].id;
        lastFrame = connectedLastFrame(node);
        render();
    };
    node._weeToddFramesFit = fitNodeSize;
    node._weeToddFramesSettle = settleNodeSize;
    render();
    settleNodeSize();
    if (manifestWidget.value === "[]" || !manifestWidget.value) sync();
}

app.registerExtension({
    name: "WeeTodd.H3.FramesEditorV2",
    setup() {
        // Also migrate nodes that were instantiated by an older cached editor
        // module before this extension registered.
        for (const delay of [0, 50, 250, 1000]) {
            setTimeout(() => {
                for (const node of app.graph?._nodes ?? []) {
                    if (node.type === NODE_NAME || node.comfyClass === NODE_NAME) {
                        applyCompactNodeSize(node);
                    }
                }
            }, delay);
        }
    },
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;
        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function onNodeCreated() {
            const result = originalOnNodeCreated?.apply(this, arguments);
            enhanceFramesNode(this);
            return result;
        };
        const originalOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function onConfigure() {
            const result = originalOnConfigure?.apply(this, arguments);
            this._weeToddFramesReload?.();
            this._weeToddFramesSettle?.();
            return result;
        };
        const originalOnSerialize = nodeType.prototype.onSerialize;
        nodeType.prototype.onSerialize = function onSerialize(serialized) {
            const result = originalOnSerialize?.apply(this, arguments);
            if (serialized) {
                const width = Math.max(
                    Number(serialized.size?.[0]) || Number(this.size?.[0]) || 0,
                    MINIMUM_NODE_WIDTH,
                );
                serialized.size = [width, EDITOR_NODE_HEIGHT];
            }
            return result;
        };
    },
});
