// LoadImagePaint - inline mask painting on the LoadImage node body.
// Companion JS for nodes/load_image_paint.py.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_NAME = "LoadImagePaint";

// Display size of the embedded canvas. Width matches the node body width;
// height is computed per-image to preserve aspect ratio, capped so portrait
// images don't overflow the node. The painted mask is always stored at the
// image's NATIVE resolution; this canvas is just the view.
const CANVAS_WIDTH = 320;
const MAX_CANVAS_HEIGHT = 480;
const TOOLBAR_HEIGHT = 32;

function loadImageURL(filename, subfolder, type) {
    const params = new URLSearchParams();
    params.set("filename", filename);
    if (subfolder) params.set("subfolder", subfolder);
    params.set("type", type || "input");
    return `/view?${params.toString()}`;
}

function parseFilenameFromWidget(value) {
    // ComfyUI's LoadImage widget may store "name [subfolder/type]" forms in
    // some versions, or just "name". Handle both.
    if (!value) return { filename: "", subfolder: "", type: "input" };
    const m = /^(.+?)\s*\[(input|output|temp)\]\s*$/.exec(value);
    if (m) return { filename: m[1].trim(), subfolder: "", type: m[2] };
    const slashIdx = value.lastIndexOf("/");
    if (slashIdx >= 0) {
        return {
            filename: value.substring(slashIdx + 1),
            subfolder: value.substring(0, slashIdx),
            type: "input",
        };
    }
    return { filename: value, subfolder: "", type: "input" };
}

app.registerExtension({
    name: "nanobanana.LoadImagePaint",

    async beforeRegisterNodeDef(nodeType, nodeData, _app) {
        if (nodeData.name !== NODE_NAME) return;

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            origOnNodeCreated?.apply(this, arguments);

            const self = this;

            // Locate widgets
            const imageWidget = this.widgets.find((w) => w.name === "image");
            const maskDataWidget = this.widgets.find((w) => w.name === "mask_data");
            const brushSizeWidget = this.widgets.find((w) => w.name === "brush_size");
            if (!imageWidget || !maskDataWidget || !brushSizeWidget) return;

            // Hide the mask_data widget from the UI (still serializes with workflow)
            maskDataWidget.computeSize = () => [0, -4];
            maskDataWidget.type = "hidden";

            // --- Build the inline canvas widget ---
            const container = document.createElement("div");
            container.style.cssText = `
                position: relative;
                width: 100%;
                background: #1a1a1a;
                border: 1px solid #444;
                border-radius: 4px;
                overflow: hidden;
                margin-top: 4px;
            `;

            const imageCanvas = document.createElement("canvas");
            imageCanvas.width = CANVAS_WIDTH;
            imageCanvas.height = CANVAS_WIDTH;
            // Explicit display size (no width:100%) so we can clamp height
            // and feed it back to LiteGraph for node resize.
            imageCanvas.style.cssText = "display:block;cursor:crosshair;";

            const maskCanvas = document.createElement("canvas");
            maskCanvas.width = CANVAS_WIDTH;
            maskCanvas.height = CANVAS_WIDTH;
            maskCanvas.style.cssText = "position:absolute;left:0;top:0;pointer-events:none;opacity:0.55;mix-blend-mode:screen;";

            container.appendChild(imageCanvas);
            container.appendChild(maskCanvas);

            // --- Toolbar ---
            const toolbar = document.createElement("div");
            toolbar.style.cssText = `
                display: flex;
                gap: 4px;
                padding: 4px;
                background: #222;
                font-size: 11px;
                color: #ccc;
            `;
            const clearBtn = document.createElement("button");
            clearBtn.textContent = "Clear Mask";
            clearBtn.style.cssText = "padding:2px 8px;background:#333;border:1px solid #555;color:#ccc;cursor:pointer;border-radius:3px;";
            const infoLabel = document.createElement("span");
            infoLabel.textContent = "Left: paint • Right/Shift+Left: erase";
            infoLabel.style.cssText = "margin-left:auto;align-self:center;color:#888;";
            toolbar.appendChild(clearBtn);
            toolbar.appendChild(infoLabel);

            const root = document.createElement("div");
            root.appendChild(toolbar);
            root.appendChild(container);

            // Track image state
            const state = {
                imageW: 0,
                imageH: 0,
                imageLoaded: false,
                drawing: false,
                erasing: false,
                lastX: 0,
                lastY: 0,
            };

            // --- Mask canvas operations ---
            // We keep the painted mask at the IMAGE's NATIVE resolution in a
            // hidden offscreen canvas. The visible maskCanvas shows the same
            // mask scaled down. Painting writes to the native canvas; we
            // refresh the visible canvas after each stroke.
            const nativeMaskCanvas = document.createElement("canvas");
            // Initialize at view size; resized to native dims when image loads.
            nativeMaskCanvas.width = CANVAS_WIDTH;
            nativeMaskCanvas.height = CANVAS_WIDTH;

            // Tracks the on-screen height of the canvas pair (without toolbar).
            // Used to dynamically size the node body so images of any aspect
            // ratio stay inside the node body.
            let currentCanvasHeight = CANVAS_WIDTH;

            function applyCanvasDisplaySize(displayW, displayH) {
                // Set CSS pixel dimensions (display) AND drawing-surface
                // dimensions (internal pixel buffer). Match them so the
                // rendered image is crisp without scaling artifacts.
                imageCanvas.width = displayW;
                imageCanvas.height = displayH;
                imageCanvas.style.width = displayW + "px";
                imageCanvas.style.height = displayH + "px";
                maskCanvas.width = displayW;
                maskCanvas.height = displayH;
                maskCanvas.style.width = displayW + "px";
                maskCanvas.style.height = displayH + "px";
                container.style.width = displayW + "px";
                container.style.height = displayH + "px";
                currentCanvasHeight = displayH;

                // Tell LiteGraph the node needs more (or less) vertical room.
                try {
                    const newH = displayH + TOOLBAR_HEIGHT + 180; // 180 = widgets above
                    if (Math.abs(self.size[1] - newH) > 4) {
                        self.setSize([Math.max(self.size[0], displayW + 20), newH]);
                    }
                    self.setDirtyCanvas?.(true, true);
                } catch (_) {}
            }

            function refreshVisibleMask() {
                const ctx = maskCanvas.getContext("2d");
                ctx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
                ctx.drawImage(nativeMaskCanvas, 0, 0, maskCanvas.width, maskCanvas.height);
            }

            function saveMaskToWidget() {
                if (!state.imageLoaded) return;
                // Encode native-resolution mask as PNG base64, write to widget
                const dataUrl = nativeMaskCanvas.toDataURL("image/png");
                maskDataWidget.value = dataUrl;
                if (typeof self.onWidgetChanged === "function") {
                    self.onWidgetChanged(maskDataWidget.name, dataUrl, "", maskDataWidget);
                }
            }

            // --- Load image into background canvas ---
            function loadImage(filename) {
                if (!filename || filename === "(no files)") return;
                const { filename: f, subfolder, type } = parseFilenameFromWidget(filename);
                const url = loadImageURL(f, subfolder, type);
                const img = new Image();
                img.onload = () => {
                    state.imageW = img.naturalWidth;
                    state.imageH = img.naturalHeight;
                    state.imageLoaded = true;

                    // Resize native mask canvas to match image
                    // Preserve old mask only if dims match (rare on file change)
                    const oldMaskData = nativeMaskCanvas.width === img.naturalWidth
                        && nativeMaskCanvas.height === img.naturalHeight
                        ? nativeMaskCanvas.getContext("2d").getImageData(0, 0, nativeMaskCanvas.width, nativeMaskCanvas.height)
                        : null;
                    nativeMaskCanvas.width = img.naturalWidth;
                    nativeMaskCanvas.height = img.naturalHeight;
                    if (oldMaskData) {
                        nativeMaskCanvas.getContext("2d").putImageData(oldMaskData, 0, 0);
                    } else {
                        nativeMaskCanvas.getContext("2d").clearRect(0, 0, nativeMaskCanvas.width, nativeMaskCanvas.height);
                        // Try to restore mask_data widget value (workflow reload)
                        if (maskDataWidget.value) {
                            const saved = new Image();
                            saved.onload = () => {
                                const ctx = nativeMaskCanvas.getContext("2d");
                                ctx.clearRect(0, 0, nativeMaskCanvas.width, nativeMaskCanvas.height);
                                // Scale to image dims if needed
                                ctx.drawImage(saved, 0, 0, nativeMaskCanvas.width, nativeMaskCanvas.height);
                                refreshVisibleMask();
                            };
                            saved.src = maskDataWidget.value;
                        }
                    }

                    // Compute display size: preserve aspect, cap height to
                    // MAX_CANVAS_HEIGHT so portrait images don't overflow.
                    const aspect = img.naturalHeight / img.naturalWidth;
                    let dispW = CANVAS_WIDTH;
                    let dispH = Math.round(CANVAS_WIDTH * aspect);
                    if (dispH > MAX_CANVAS_HEIGHT) {
                        dispH = MAX_CANVAS_HEIGHT;
                        dispW = Math.round(MAX_CANVAS_HEIGHT / aspect);
                    }
                    applyCanvasDisplaySize(dispW, dispH);

                    // Draw image into the now-correctly-sized canvas
                    const ctx = imageCanvas.getContext("2d");
                    ctx.clearRect(0, 0, imageCanvas.width, imageCanvas.height);
                    ctx.drawImage(img, 0, 0, imageCanvas.width, imageCanvas.height);
                    refreshVisibleMask();
                };
                img.onerror = () => {
                    state.imageLoaded = false;
                };
                img.src = url;
            }

            // --- Painting handlers (operate in NATIVE image coords) ---
            function viewToNative(clientX, clientY) {
                const rect = maskCanvas.getBoundingClientRect();
                const x = (clientX - rect.left) / rect.width * state.imageW;
                const y = (clientY - rect.top) / rect.height * state.imageH;
                return [Math.round(x), Math.round(y)];
            }

            function paintAt(nx, ny, erase) {
                const r = brushSizeWidget.value;
                const ctx = nativeMaskCanvas.getContext("2d");
                ctx.globalCompositeOperation = erase ? "destination-out" : "source-over";
                ctx.fillStyle = erase ? "rgba(0,0,0,1)" : "rgba(255,255,255,1)";
                ctx.beginPath();
                ctx.arc(nx, ny, r, 0, Math.PI * 2);
                ctx.fill();
                if (state.lastX !== null) {
                    // Connect via a line stroke for continuous brush
                    ctx.lineWidth = r * 2;
                    ctx.lineCap = "round";
                    ctx.strokeStyle = erase ? "rgba(0,0,0,1)" : "rgba(255,255,255,1)";
                    ctx.beginPath();
                    ctx.moveTo(state.lastX, state.lastY);
                    ctx.lineTo(nx, ny);
                    ctx.stroke();
                }
                state.lastX = nx;
                state.lastY = ny;
            }

            imageCanvas.addEventListener("contextmenu", (e) => e.preventDefault());

            imageCanvas.addEventListener("pointerdown", (e) => {
                if (!state.imageLoaded) return;
                e.preventDefault();
                e.stopPropagation();
                imageCanvas.setPointerCapture(e.pointerId);
                state.drawing = true;
                state.erasing = e.button === 2 || e.shiftKey;
                state.lastX = null;
                const [nx, ny] = viewToNative(e.clientX, e.clientY);
                paintAt(nx, ny, state.erasing);
                refreshVisibleMask();
            });

            imageCanvas.addEventListener("pointermove", (e) => {
                if (!state.drawing) return;
                e.preventDefault();
                const [nx, ny] = viewToNative(e.clientX, e.clientY);
                paintAt(nx, ny, state.erasing);
                refreshVisibleMask();
            });

            const endStroke = (e) => {
                if (!state.drawing) return;
                state.drawing = false;
                state.lastX = null;
                try { imageCanvas.releasePointerCapture(e.pointerId); } catch (_) {}
                saveMaskToWidget();
            };
            imageCanvas.addEventListener("pointerup", endStroke);
            imageCanvas.addEventListener("pointercancel", endStroke);

            clearBtn.addEventListener("click", (e) => {
                e.preventDefault();
                e.stopPropagation();
                const ctx = nativeMaskCanvas.getContext("2d");
                ctx.clearRect(0, 0, nativeMaskCanvas.width, nativeMaskCanvas.height);
                refreshVisibleMask();
                maskDataWidget.value = "";
            });

            // Watch image widget for changes
            const origImageCb = imageWidget.callback;
            imageWidget.callback = function (value) {
                if (origImageCb) origImageCb.call(this, value);
                loadImage(value);
            };

            // Initial load if image already selected
            if (imageWidget.value && imageWidget.value !== "(no files)") {
                // Defer to next tick so node is fully constructed
                setTimeout(() => loadImage(imageWidget.value), 0);
            }

            // Add the DOM widget to the node. getMinHeight feeds the canvas's
            // current rendered height back to LiteGraph so the node body
            // grows to fit portrait / landscape images correctly.
            this.addDOMWidget("paint_canvas", "div", root, {
                serialize: false,
                hideOnZoom: false,
                getMinHeight: () => currentCanvasHeight + TOOLBAR_HEIGHT + 8,
            });

            // Initial node size — final height is set by applyCanvasDisplaySize
            // once the image loads.
            this.setSize([Math.max(this.size[0], 360), Math.max(this.size[1], 520)]);
        };
    },
});
