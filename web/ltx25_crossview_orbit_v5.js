import { app } from "../../scripts/app.js";

const NODE_NAME = "WeeToddLTX25CrossViewCameraOrbit";
const MINIMUM_NODE_WIDTH = 430;
const VIEW_HEIGHT = 390;

function clamp(value, low, high) {
    return Math.min(high, Math.max(low, Number(value)));
}

function wrapDegrees(value) {
    return ((Number(value) + 180) % 360 + 360) % 360 - 180;
}

function unwrapDegrees(values) {
    if (!values.length) return [];
    const result = [Number(values[0])];
    for (const value of values.slice(1)) {
        result.push(result.at(-1) + wrapDegrees(Number(value) - result.at(-1)));
    }
    return result;
}

function easeFraction(value, mode) {
    if (mode === "ease_in") return value * value;
    if (mode === "ease_out") return 1 - (1 - value) ** 2;
    if (mode === "ease_in_out") return 0.5 - 0.5 * Math.cos(Math.PI * value);
    return value;
}

function catmull(values, index, value) {
    const p1 = values[index];
    const p2 = values[index + 1];
    const p0 = index ? values[index - 1] : 2 * p1 - p2;
    const p3 = index + 2 < values.length ? values[index + 2] : 2 * p2 - p1;
    return 0.5 * (2 * p1 + (-p0 + p2) * value
        + (2 * p0 - 5 * p1 + 4 * p2 - p3) * value ** 2
        + (-p0 + 3 * p1 - 3 * p2 + p3) * value ** 3);
}

function enhanceOrbitNode(node) {
    if (node._weeToddCrossViewOrbit || !Array.isArray(node.widgets)) return;
    const widget = (name) => node.widgets.find((candidate) => candidate.name === name);
    const azimuth = widget("azimuth");
    const elevation = widget("elevation");
    const distance = widget("distance");
    const fov = widget("horizontal_fov");
    const verticalShift = widget("vertical_shift");
    const pivotX = widget("pivot_x");
    const pivotY = widget("pivot_y");
    const pivotZ = widget("pivot_z");
    const interpolation = widget("interpolation");
    const keyframeText = widget("camera_keyframes");
    if (!azimuth || !elevation || !distance || !fov || !keyframeText) return;
    node._weeToddCrossViewOrbit = true;
    node.properties ??= {};

    const root = document.createElement("div");
    root.className = "weetodd-crossview-orbit";
    root.innerHTML = `
      <style>
        .weetodd-crossview-orbit { box-sizing:border-box; width:100%; height:${VIEW_HEIGHT}px;
          padding:8px; display:grid; grid-template-rows:minmax(0,1fr) auto auto auto; gap:7px;
          color:var(--fg-color, #ddd); background:var(--comfy-menu-bg, #202020);
          border:1px solid var(--border-color, #444); border-radius:4px;
          font:12px Arial, sans-serif; user-select:none; }
        .weetodd-crossview-stage { position:relative; min-height:0; overflow:hidden;
          border:1px solid var(--border-color, #444); border-radius:4px;
          background:var(--bg-color, #181818); touch-action:none; cursor:grab; }
        .weetodd-crossview-stage.is-dragging { cursor:grabbing; }
        .weetodd-crossview-stage canvas { width:100%; height:100%; display:block; }
        .weetodd-crossview-readout { position:absolute; left:8px; top:7px; padding:4px 6px;
          border-radius:3px; color:var(--fg-color, #ddd); background:#0009;
          pointer-events:none; line-height:1.35; }
        .weetodd-crossview-warning { color:var(--warning-text, #e6b85c); }
        .weetodd-crossview-help { position:absolute; right:8px; bottom:7px;
          color:var(--descrip-text, #999); background:#0008; padding:3px 5px;
          border-radius:3px; pointer-events:none; }
        .weetodd-crossview-toolbar, .weetodd-crossview-timeline,
        .weetodd-crossview-actions { display:flex; align-items:center; gap:6px; min-width:0; }
        .weetodd-crossview-orbit button { padding:4px 8px; color:var(--fg-color, #ddd);
          background:var(--input-bg, #2b2b2b); border:1px solid var(--border-color, #555);
          border-radius:3px; cursor:pointer; }
        .weetodd-crossview-orbit button:hover { filter:brightness(1.15); }
        .weetodd-crossview-orbit button.is-selected { border-color:#74b9ff; color:#a9d7ff; }
        .weetodd-crossview-orbit input { box-sizing:border-box; height:24px;
          color:var(--fg-color, #ddd); background:var(--input-bg, #2b2b2b);
          border:1px solid var(--border-color, #555); border-radius:3px; }
        .weetodd-crossview-timeline input[type=range] { flex:1; min-width:60px; }
        .weetodd-crossview-number { width:58px; padding:2px 5px; }
        .weetodd-crossview-frames { display:flex; gap:4px; min-width:0; overflow-x:auto; }
        .weetodd-crossview-frame { white-space:nowrap; }
        .weetodd-crossview-frame.is-selected { border-color:#74b9ff; color:#a9d7ff; }
        .weetodd-crossview-legend { margin-left:auto; min-width:0; overflow:hidden;
          text-overflow:ellipsis; white-space:nowrap; color:var(--descrip-text, #999); }
      </style>
      <div class="weetodd-crossview-stage">
        <canvas aria-label="CrossView orbit camera sphere"></canvas>
        <div class="weetodd-crossview-readout"></div>
        <div class="weetodd-crossview-help">drag orbit · wheel distance · click a green point</div>
      </div>
      <div class="weetodd-crossview-toolbar">
        <button type="button" data-source>Source</button>
        <button type="button" data-recommended>Recommended</button>
        <button type="button" data-interaction>Mode: place camera</button>
        <button type="button" data-reset-view>Reset view</button>
        <span class="weetodd-crossview-legend">green points = camera keyframes</span>
      </div>
      <div class="weetodd-crossview-timeline">
        <label>Frame</label><input class="weetodd-crossview-number" data-frame type="number" min="1" value="1">
        <input data-scrub type="range" min="1" value="1">
        <label>of</label><input class="weetodd-crossview-number" data-total type="number" min="9" max="10000" value="121">
      </div>
      <div class="weetodd-crossview-actions">
        <button type="button" data-add>+ Add / Update</button>
        <button type="button" data-delete>Delete</button>
        <button type="button" data-clear>Clear</button>
        <div class="weetodd-crossview-frames"></div>
      </div>`;

    const stage = root.querySelector(".weetodd-crossview-stage");
    const canvas = root.querySelector("canvas");
    const readout = root.querySelector(".weetodd-crossview-readout");
    const help = root.querySelector(".weetodd-crossview-help");
    const frameInput = root.querySelector("[data-frame]");
    const scrub = root.querySelector("[data-scrub]");
    const totalInput = root.querySelector("[data-total]");
    const frameList = root.querySelector(".weetodd-crossview-frames");
    const interactionButton = root.querySelector("[data-interaction]");
    const legend = root.querySelector(".weetodd-crossview-legend");
    const context = canvas.getContext("2d");
    let dragging = null;
    let markerPositions = [];
    let selectedFrame = 1;
    let totalFrames = clamp(node.properties.crossview_timeline_frames || 121, 9, 10000);
    let interactionMode = node.properties.crossview_interaction_mode === "rotate_view"
        ? "rotate_view" : "place_camera";
    let viewYaw = clamp(node.properties.crossview_view_yaw || 0, -180, 180);
    let viewPitch = clamp(node.properties.crossview_view_pitch || 0, -89, 89);
    let viewRoll = wrapDegrees(node.properties.crossview_view_roll || 0);

    function syncInteractionMode() {
        const rotatesView = interactionMode === "rotate_view";
        interactionButton.textContent = rotatesView ? "Mode: rotate view" : "Mode: place camera";
        interactionButton.classList.toggle("is-selected", rotatesView);
        help.textContent = rotatesView
            ? "drag yaw/pitch · option-drag rolls · keyframes stay unchanged"
            : "drag places camera · shift/right-drag rotates view · option-drag rolls";
    }

    function setView(yaw, pitch, roll = viewRoll) {
        viewYaw = wrapDegrees(yaw);
        viewPitch = clamp(pitch, -89, 89);
        viewRoll = wrapDegrees(roll);
        node.properties.crossview_view_yaw = viewYaw;
        node.properties.crossview_view_pitch = viewPitch;
        node.properties.crossview_view_roll = viewRoll;
        node.graph?.setDirtyCanvas?.(true, true);
        draw();
    }

    function parseKeyframes() {
        try {
            const payload = JSON.parse(String(keyframeText.value || "[]"));
            if (!Array.isArray(payload)) return [];
            return payload.map((item) => ({
                f: Math.max(1, Math.round(Number(item.f ?? item.frame))),
                az: Number(item.az ?? item.azimuth ?? azimuth.value),
                el: Number(item.el ?? item.elevation ?? elevation.value),
                dist: Number(item.dist ?? item.distance ?? distance.value),
                vs: Number(item.vs ?? item.vertical_shift ?? verticalShift?.value ?? 0),
                px: Number(item.px ?? item.pivot_x ?? pivotX?.value ?? 0),
                py: Number(item.py ?? item.pivot_y ?? pivotY?.value ?? 0),
                pz: Number(item.pz ?? item.pivot_z ?? pivotZ?.value ?? 1.05),
            })).filter((item) => Number.isFinite(item.f) && Number.isFinite(item.az)
                && Number.isFinite(item.el) && Number.isFinite(item.dist))
                .sort((left, right) => left.f - right.f);
        } catch (_error) {
            return [];
        }
    }

    function notifyWidget(target, value) {
        const previous = target.value;
        if (previous === value) return;
        target.value = value;
        target.callback?.(value, app.canvas, node, target);
        node.onWidgetChanged?.(target.name, value, previous, target);
        node.graph?.setDirtyCanvas?.(true, true);
    }

    function setNumberWidget(target, value) {
        const rounded = Math.round(Number(value) * 100) / 100;
        if (Math.abs(Number(target.value) - rounded) < 1e-9) return;
        notifyWidget(target, rounded);
    }

    function setPose(az, el, dist = Number(distance.value)) {
        setNumberWidget(azimuth, clamp(az, -180, 180));
        setNumberWidget(elevation, clamp(el, -90, 90));
        setNumberWidget(distance, Math.round(clamp(dist, 0.1, 3) * 20) / 20);
        draw();
    }

    function setSelectedFrame(frame, loadPose = true) {
        selectedFrame = Math.round(clamp(frame, 1, totalFrames));
        frameInput.value = String(selectedFrame);
        scrub.value = String(selectedFrame);
        if (loadPose) {
            const item = parseKeyframes().find((candidate) => candidate.f === selectedFrame);
            if (item) setPose(item.az, item.el, item.dist);
        }
        renderFrameList();
        draw();
    }

    function serializeKeyframes(items) {
        notifyWidget(keyframeText, JSON.stringify(items.map((item) => ({
            f: item.f, az: item.az, el: item.el, dist: item.dist,
            vs: item.vs, px: item.px, py: item.py, pz: item.pz,
        }))));
        renderFrameList();
        draw();
    }

    function currentPose() {
        return {
            f: selectedFrame, az: Number(azimuth.value), el: Number(elevation.value),
            dist: Number(distance.value), vs: Number(verticalShift?.value ?? 0),
            px: Number(pivotX?.value ?? 0), py: Number(pivotY?.value ?? 0),
            pz: Number(pivotZ?.value ?? 1.05),
        };
    }

    function renderFrameList() {
        frameList.replaceChildren();
        for (const item of parseKeyframes()) {
            const button = document.createElement("button");
            button.type = "button";
            button.className = "weetodd-crossview-frame";
            if (item.f === selectedFrame) button.classList.add("is-selected");
            button.textContent = `F${item.f}`;
            button.title = `Frame ${item.f}: az ${item.az}°, el ${item.el}°, distance ${item.dist}`;
            button.addEventListener("click", () => setSelectedFrame(item.f, true));
            frameList.append(button);
        }
    }

    function resizeCanvas() {
        const ratio = Math.max(1, window.devicePixelRatio || 1);
        // LiteGraph zooms DOM widgets with a CSS transform. getBoundingClientRect()
        // therefore reports the transformed graph size, while draw() uses the logical
        // client size. Keep the bitmap on that same logical coordinate system.
        const width = Math.max(1, Math.round(stage.clientWidth * ratio));
        const height = Math.max(1, Math.round(stage.clientHeight * ratio));
        if (canvas.width !== width || canvas.height !== height) {
            canvas.width = width;
            canvas.height = height;
        }
        context.setTransform(ratio, 0, 0, ratio, 0, 0);
        draw();
    }

    function rotateForView(point) {
        const yaw = viewYaw * Math.PI / 180;
        const pitch = viewPitch * Math.PI / 180;
        const roll = viewRoll * Math.PI / 180;
        const yawX = point.x * Math.cos(yaw) + point.z * Math.sin(yaw);
        const yawZ = -point.x * Math.sin(yaw) + point.z * Math.cos(yaw);
        const pitchY = point.y * Math.cos(pitch) - yawZ * Math.sin(pitch);
        const pitchZ = point.y * Math.sin(pitch) + yawZ * Math.cos(pitch);
        return {
            x: yawX * Math.cos(roll) - pitchY * Math.sin(roll),
            y: yawX * Math.sin(roll) + pitchY * Math.cos(roll),
            z: pitchZ,
        };
    }

    function sphericalPoint(azimuthDegrees, elevationDegrees, radial = 1) {
        const az = Number(azimuthDegrees) * Math.PI / 180;
        const el = Number(elevationDegrees) * Math.PI / 180;
        return {
            x: radial * Math.sin(az) * Math.cos(el),
            y: radial * Math.sin(el),
            z: radial * Math.cos(az) * Math.cos(el),
        };
    }

    function projectPoint(point, cx, cy, radius) {
        const rotated = rotateForView(point);
        const perspective = 0.9 + 0.1 * rotated.z;
        return {
            x: cx + radius * perspective * rotated.x,
            y: cy - radius * perspective * rotated.y,
            depth: rotated.z,
        };
    }

    function project(item, cx, cy, radius) {
        const az = Number(item.az) * Math.PI / 180;
        const el = Number(item.el) * Math.PI / 180;
        const radial = 0.88 + 0.08 * (Number(item.dist) - 1);
        return projectPoint({
            x: radial * Math.sin(az) * Math.cos(el),
            y: radial * Math.sin(el),
            z: radial * Math.cos(az) * Math.cos(el),
        }, cx, cy, radius);
    }

    function strokeSphereLine(points, cx, cy, radius, color, width = 1) {
        const projected = points.map((point) => projectPoint(point, cx, cy, radius));
        context.beginPath();
        projected.forEach((point, index) => {
            index ? context.lineTo(point.x, point.y) : context.moveTo(point.x, point.y);
        });
        context.strokeStyle = color;
        context.lineWidth = width;
        context.stroke();
    }

    function samplePath(keyframes, interpolation) {
        if (keyframes.length < 2) return [...keyframes];
        const fields = ["az", "el", "dist", "vs", "px", "py", "pz"];
        const channels = Object.fromEntries(fields.map((field) => [
            field, keyframes.map((item) => Number(item[field])),
        ]));
        channels.az = unwrapDegrees(channels.az);
        const result = [];
        keyframes.slice(0, -1).forEach((left, segment) => {
            const right = keyframes[segment + 1];
            const samples = Math.max(12, Math.min(72, right.f - left.f));
            for (let sample = 0; sample < samples; sample += 1) {
                const rawFraction = sample / samples;
                const smooth = interpolation === "smooth" && keyframes.length > 2;
                const fraction = smooth ? rawFraction : easeFraction(rawFraction, interpolation);
                const item = { f: left.f + (right.f - left.f) * rawFraction };
                for (const field of fields) {
                    const values = channels[field];
                    item[field] = smooth
                        ? catmull(values, segment, fraction)
                        : values[segment] + (values[segment + 1] - values[segment]) * fraction;
                }
                item.az = wrapDegrees(item.az);
                item.el = clamp(item.el, -90, 90);
                item.dist = clamp(item.dist, 0.1, 3);
                result.push(item);
            }
        });
        result.push({ ...keyframes.at(-1) });
        return result;
    }

    function draw() {
        const width = canvas.clientWidth;
        const height = canvas.clientHeight;
        if (!width || !height) return;
        context.clearRect(0, 0, width, height);
        const cx = width / 2;
        const cy = height / 2 + 7;
        const radius = Math.max(35, Math.min(width * 0.29, height * 0.39));
        const styles = getComputedStyle(root);
        const grid = styles.getPropertyValue("--border-color").trim() || "#555";
        const foreground = styles.getPropertyValue("--fg-color").trim() || "#ddd";
        const gradient = context.createRadialGradient(
            cx - radius * 0.35, cy - radius * 0.4, radius * 0.08, cx, cy, radius,
        );
        gradient.addColorStop(0, "#536271");
        gradient.addColorStop(0.65, "#28323b");
        gradient.addColorStop(1, "#11171c");
        context.beginPath();
        context.arc(cx, cy, radius, 0, Math.PI * 2);
        context.fillStyle = gradient;
        context.fill();
        context.strokeStyle = grid;
        context.lineWidth = 1.4;
        context.stroke();
        for (const latitude of [-60, -30, 0, 30, 60]) {
            const points = [];
            for (let longitude = -180; longitude <= 180; longitude += 5) {
                points.push(sphericalPoint(longitude, latitude));
            }
            strokeSphereLine(points, cx, cy, radius, "#ffffff2b");
        }
        for (const longitude of [-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150]) {
            const points = [];
            for (let latitude = -90; latitude <= 90; latitude += 4) {
                points.push(sphericalPoint(longitude, latitude));
            }
            strokeSphereLine(points, cx, cy, radius, "#ffffff24");
        }
        const reliableRegion = [];
        for (let index = 0; index <= 48; index += 1) {
            reliableRegion.push(sphericalPoint(-45 + 90 * index / 48, -20, 0.91));
        }
        for (let index = 48; index >= 0; index -= 1) {
            reliableRegion.push(sphericalPoint(-45 + 90 * index / 48, 30, 0.91));
        }
        const projectedRegion = reliableRegion.map((point) => projectPoint(point, cx, cy, radius));
        context.beginPath();
        projectedRegion.forEach((point, index) => {
            index ? context.lineTo(point.x, point.y) : context.moveTo(point.x, point.y);
        });
        context.closePath();
        context.fillStyle = "#45b87824";
        context.fill();
        context.strokeStyle = "#65d595a8";
        context.stroke();

        const keyframes = parseKeyframes();
        markerPositions = keyframes.map((item) => ({ item, ...project(item, cx, cy, radius) }));
        if (markerPositions.length > 1) {
            const interpolationMode = String(interpolation?.value || "linear");
            const sampledPath = samplePath(keyframes, interpolationMode)
                .map((item) => project(item, cx, cy, radius));
            context.beginPath();
            sampledPath.forEach((point, index) => {
                index ? context.lineTo(point.x, point.y) : context.moveTo(point.x, point.y);
            });
            context.strokeStyle = "#65d595c0";
            context.lineWidth = 2;
            context.stroke();
            // Equal-time marks expose the speed profile: ease-in points bunch near
            // the start, ease-out points bunch near the end, and smooth follows
            // the same Catmull-Rom trajectory used by the backend.
            const markerStride = Math.max(1, Math.floor(sampledPath.length / 12));
            for (let index = markerStride; index < sampledPath.length - 1; index += markerStride) {
                const point = sampledPath[index];
                context.beginPath();
                context.arc(point.x, point.y, 2.25, 0, Math.PI * 2);
                context.fillStyle = "#b3f5c8d8";
                context.fill();
            }
            legend.textContent = `${interpolationMode.replaceAll("_", " ")} path · dots = equal time`;
        } else {
            legend.textContent = "green points = camera keyframes";
        }
        for (const marker of markerPositions) {
            context.beginPath();
            context.arc(marker.x, marker.y, marker.item.f === selectedFrame ? 7 : 5, 0, Math.PI * 2);
            context.fillStyle = marker.item.f === selectedFrame ? "#74b9ff" : "#65d595";
            context.fill();
            context.strokeStyle = foreground;
            context.stroke();
            context.fillStyle = foreground;
            context.font = "10px Arial";
            context.fillText(String(marker.item.f), marker.x + 7, marker.y - 6);
        }
        const current = project({ az: azimuth.value, el: elevation.value, dist: distance.value }, cx, cy, radius);
        context.beginPath();
        context.moveTo(cx, cy);
        context.lineTo(current.x, current.y);
        context.strokeStyle = "#74b9ffb0";
        context.stroke();
        context.beginPath();
        context.arc(current.x, current.y, 7, 0, Math.PI * 2);
        context.fillStyle = "#74b9ff";
        context.fill();
        context.strokeStyle = foreground;
        context.stroke();
        const reliable = Math.abs(Number(azimuth.value)) <= 45
            && Number(elevation.value) >= -20 && Number(elevation.value) <= 30
            && Math.abs(Number(distance.value) - 1) <= 0.05;
        readout.innerHTML = `frame ${selectedFrame}/${totalFrames} · ${keyframes.length} keyframe${keyframes.length === 1 ? "" : "s"}<br>`
            + `az ${Number(azimuth.value).toFixed(1)}° · el ${Number(elevation.value).toFixed(1)}° · distance ${Number(distance.value).toFixed(2)}`
            + `<br>view yaw ${viewYaw.toFixed(0)}° · pitch ${viewPitch.toFixed(0)}° · roll ${viewRoll.toFixed(0)}°`
            + (reliable ? "" : '<br><span class="weetodd-crossview-warning">outside best-tested range</span>');
    }

    stage.addEventListener("pointerdown", (event) => {
        const bounds = stage.getBoundingClientRect();
        const x = event.clientX - bounds.left;
        const y = event.clientY - bounds.top;
        const selected = markerPositions.find((marker) => Math.hypot(marker.x - x, marker.y - y) <= 11);
        if (selected) return setSelectedFrame(selected.item.f, true);
        const rotateView = interactionMode === "rotate_view" || event.shiftKey || event.button !== 0;
        dragging = { pointer: event.pointerId, x: event.clientX, y: event.clientY,
            rotateView, rollView: event.altKey, viewYaw, viewPitch, viewRoll,
            az: Number(azimuth.value), el: Number(elevation.value) };
        stage.setPointerCapture(event.pointerId);
        stage.classList.add("is-dragging");
    });
    stage.addEventListener("pointermove", (event) => {
        if (!dragging || event.pointerId !== dragging.pointer) return;
        if (dragging.rotateView) {
            if (dragging.rollView) {
                setView(dragging.viewYaw, dragging.viewPitch,
                    dragging.viewRoll + (event.clientX - dragging.x) * 0.45);
            } else {
                setView(dragging.viewYaw + (event.clientX - dragging.x) * 0.45,
                    dragging.viewPitch - (event.clientY - dragging.y) * 0.45,
                    dragging.viewRoll);
            }
        } else {
            setPose(dragging.az + (event.clientX - dragging.x) * 0.45,
                dragging.el - (event.clientY - dragging.y) * 0.45);
        }
    });
    const stopDrag = (event) => {
        if (!dragging || event.pointerId !== dragging.pointer) return;
        dragging = null;
        stage.classList.remove("is-dragging");
    };
    stage.addEventListener("pointerup", stopDrag);
    stage.addEventListener("pointercancel", stopDrag);
    stage.addEventListener("contextmenu", (event) => event.preventDefault());
    stage.addEventListener("wheel", (event) => {
        event.preventDefault();
        setPose(Number(azimuth.value), Number(elevation.value),
            Number(distance.value) + Math.sign(event.deltaY) * 0.05);
    }, { passive: false });
    stage.addEventListener("dblclick", () => {
        if (interactionMode === "rotate_view") setView(0, 0, 0);
        else setPose(0, 0, 1);
    });
    root.querySelector("[data-source]").addEventListener("click", () => setPose(0, 0, 1));
    root.querySelector("[data-recommended]").addEventListener("click", () => setPose(-30, 15, 1));
    interactionButton.addEventListener("click", () => {
        interactionMode = interactionMode === "place_camera" ? "rotate_view" : "place_camera";
        node.properties.crossview_interaction_mode = interactionMode;
        node.graph?.setDirtyCanvas?.(true, true);
        syncInteractionMode();
    });
    root.querySelector("[data-reset-view]").addEventListener("click", () => setView(0, 0, 0));
    root.querySelector("[data-add]").addEventListener("click", () => {
        const path = parseKeyframes().filter((item) => item.f !== selectedFrame);
        path.push(currentPose());
        path.sort((left, right) => left.f - right.f);
        serializeKeyframes(path);
    });
    root.querySelector("[data-delete]").addEventListener("click", () => {
        serializeKeyframes(parseKeyframes().filter((item) => item.f !== selectedFrame));
    });
    root.querySelector("[data-clear]").addEventListener("click", () => serializeKeyframes([]));
    frameInput.addEventListener("change", () => setSelectedFrame(Number(frameInput.value), true));
    scrub.addEventListener("input", () => setSelectedFrame(Number(scrub.value), true));
    totalInput.addEventListener("change", () => {
        totalFrames = Math.round(clamp(totalInput.value, 9, 10000));
        node.properties.crossview_timeline_frames = totalFrames;
        totalInput.value = String(totalFrames);
        scrub.max = String(totalFrames);
        frameInput.max = String(totalFrames);
        setSelectedFrame(selectedFrame, false);
    });
    for (const target of [azimuth, elevation, distance, fov, interpolation, keyframeText]) {
        if (!target) continue;
        const previousCallback = target.callback;
        target.callback = function orbitWidgetCallback() {
            const result = previousCallback?.apply(this, arguments);
            renderFrameList();
            draw();
            return result;
        };
    }

    totalInput.value = String(totalFrames);
    scrub.max = String(totalFrames);
    frameInput.max = String(totalFrames);
    syncInteractionMode();
    renderFrameList();
    node.addDOMWidget("crossview_orbit", "div", root, {
        serialize: false, hideOnZoom: false,
        getMinHeight: () => VIEW_HEIGHT, getMaxHeight: () => VIEW_HEIGHT,
        getHeight: () => VIEW_HEIGHT, afterResize: resizeCanvas,
    });
    const width = Math.max(Number(node.size?.[0]) || 0, MINIMUM_NODE_WIDTH);
    const computed = node.computeSize?.() ?? [width, Number(node.size?.[1]) || 0];
    node.setSize?.([width, Math.max(Number(computed[1]) || 0, VIEW_HEIGHT + 250)]);
    const observer = new ResizeObserver(resizeCanvas);
    observer.observe(stage);
    node._weeToddCrossViewRedraw = () => {
        totalFrames = clamp(node.properties?.crossview_timeline_frames || totalFrames, 9, 10000);
        interactionMode = node.properties?.crossview_interaction_mode === "rotate_view"
            ? "rotate_view" : "place_camera";
        viewYaw = clamp(node.properties?.crossview_view_yaw || 0, -180, 180);
        viewPitch = clamp(node.properties?.crossview_view_pitch || 0, -89, 89);
        viewRoll = wrapDegrees(node.properties?.crossview_view_roll || 0);
        totalInput.value = String(totalFrames);
        scrub.max = String(totalFrames);
        frameInput.max = String(totalFrames);
        syncInteractionMode();
        renderFrameList();
        resizeCanvas();
    };
    node._weeToddCrossViewObserver = observer;
    requestAnimationFrame(node._weeToddCrossViewRedraw);
}

app.registerExtension({
    name: "WeeTodd.LTX25.CrossViewOrbit",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;
        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function onNodeCreated() {
            const result = originalOnNodeCreated?.apply(this, arguments);
            enhanceOrbitNode(this);
            return result;
        };
        const originalOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function onConfigure() {
            const result = originalOnConfigure?.apply(this, arguments);
            requestAnimationFrame(() => this._weeToddCrossViewRedraw?.());
            return result;
        };
        const originalOnRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function onRemoved() {
            this._weeToddCrossViewObserver?.disconnect();
            return originalOnRemoved?.apply(this, arguments);
        };
    },
});
