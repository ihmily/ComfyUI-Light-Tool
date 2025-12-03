import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

// Displays any type of data on a node
app.registerExtension({
    name: "Light-Tool: ShowAnything",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "Light-Tool: ShowAnything") {

            function populate(text) {
                // console.log("[ShowAnything] Received text:", text);
                // console.log("[ShowAnything] text type:", typeof text);
                // console.log("[ShowAnything] is Array:", Array.isArray(text));
                // if (Array.isArray(text)) {
                //     console.log("[ShowAnything] Array length:", text.length);
                //     console.log("[ShowAnything] Array content:", text);
                // }
                
                if (this.widgets) {
                    for (let i = 0; i < this.widgets.length; i++) {
                        this.widgets[i].onRemove?.();
                    }
                    this.widgets.length = 0;
                }

                const displayText = Array.isArray(text) && text.length > 0 
                    ? text[0] 
                    : typeof text === 'string' 
                        ? text 
                        : String(text);
                
                // console.log("[ShowAnything] Final displayText:", displayText);
                
                const w = ComfyWidgets["STRING"](this, "output", ["STRING", { multiline: true }], app).widget;
                w.inputEl.readOnly = true;
                w.inputEl.style.opacity = 0.6;
                w.inputEl.style.fontFamily = "monospace";
                w.value = displayText || "";

                // Adjust size and refresh canvas
                requestAnimationFrame(() => {
                    const sz = this.computeSize();
                    if (sz[0] < this.size[0]) {
                        sz[0] = this.size[0];
                    }
                    if (sz[1] < this.size[1]) {
                        sz[1] = this.size[1];
                    }
                    this.onResize?.(sz);
                    app.graph.setDirtyCanvas(true, false);
                });
            }

            // Override onExecuted to display data in the widget
            const originalOnExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                originalOnExecuted?.apply(this, arguments);
                populate.call(this, message.text);
            };

        }
    },
});
