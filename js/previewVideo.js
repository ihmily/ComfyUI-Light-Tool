// Reference: https://github.com/ArtVentureX/comfyui-animatediff/blob/main/web/js/vid_preview.js
import { app, ANIM_PREVIEW_WIDGET } from '../../../scripts/app.js';
import { createImageHost } from "../../../scripts/ui/imagePreview.js"

const BASE_SIZE = 768;

// Set video dimensions
function setVideoDimensions(videoElement, width, height) {
    videoElement.style.width = `${width}px`;
    videoElement.style.height = `${height}px`;
}

// Resize video maintaining aspect ratio
export function resizeVideoAspectRatio(videoElement, maxWidth, maxHeight) {
    const aspectRatio = videoElement.videoWidth / videoElement.videoHeight;
    let newWidth;
    let newHeight;

    if (videoElement.videoWidth / maxWidth > videoElement.videoHeight / maxHeight) {
        newWidth = maxWidth;
        newHeight = newWidth / aspectRatio;
    } else {
        newHeight = maxHeight;
        newWidth = newHeight * aspectRatio;
    }

    setVideoDimensions(videoElement, newWidth, newHeight);
}

export function chainCallback(object, property, callback) {
    if (!object) {
        console.error("Tried to add callback to non-existent object");
        return;
    }
    const originalCallback = object[property];
    object[property] = function () {
        originalCallback?.apply(this, arguments);
        callback.apply(this, arguments);
    };
};


function isValidUrl(string) {
    try {
        new URL(string);
        return true;
    } catch (e) {
        return false;
    }
}


// Create video node
function createVideoNode(url) {
    return new Promise((resolve, reject) => {
        const videoEl = document.createElement('video');
        videoEl.addEventListener('loadedmetadata', () => {
            videoEl.controls = false;
            videoEl.loop = true;
            videoEl.muted = true;
            resizeVideoAspectRatio(videoEl, BASE_SIZE, BASE_SIZE);
            resolve(videoEl);
        });
        videoEl.addEventListener('error', () => {
            reject(new Error('Failed to load video'));
        });
        if (!isValidUrl(url)){
            const origin = window.location.origin;
            let pathname = window.location.pathname;
            if (pathname.endsWith('/') && url.startsWith('/')) {
                pathname = pathname.slice(0, -1);
            }else if (!pathname.endsWith('/') && !url.startsWith('/')){
                pathname = `${pathname}/`;
            }
            const urlWithoutQueryAndHash = origin + pathname;
            videoEl.src = urlWithoutQueryAndHash + url;
        }else{
            videoEl.src = url;
        }
        
    });
}


export function addVideoPreview(nodeType, options = {}) {

    nodeType.prototype.onDrawBackground = function (ctx) {
        if (this.flags.collapsed) return;

        const imageURLs = this.images ?? [];
        let imagesChanged = false;

        if (JSON.stringify(this.displayingImages) !== JSON.stringify(imageURLs)) {
            this.displayingImages = imageURLs;
            imagesChanged = true;
        }

        if (!imagesChanged) {
            return;
        }

        if (!imageURLs.length) {
            this.imgs = null;
            this.animatedImages = false;
            return;
        }

        const promises = imageURLs.map((url) => {
            return createVideoNode(url);
        });

        Promise.all(promises)
            .then((imgs) => {
                this.imgs = imgs.filter(Boolean);
            })
            .then(() => {
                if (!this.imgs.length) return;

                this.animatedImages = true;
                const widgetIdx = this.widgets?.findIndex((w) => w.name === ANIM_PREVIEW_WIDGET);

                this.size[0] = BASE_SIZE;
                this.size[1] = BASE_SIZE;

                if (widgetIdx > -1) {
                    const widget = this.widgets[widgetIdx];
                    widget.options.host.updateImages(this.imgs);
                } else {
                    const host = createImageHost(this);
                    const widget = this.addDOMWidget(ANIM_PREVIEW_WIDGET, 'img', host.el, {
                        host,
                        getHeight: host.getHeight,
                        onDraw: host.onDraw,
                        hideOnZoom: false,
                    });
                    widget.serializeValue = () => ({
                        height: BASE_SIZE,
                    });

                    widget.options.host.updateImages(this.imgs);
                }

                // biome-ignore lint/complexity/noForEach: <explanation>
                this.imgs.forEach((img) => {
                    if (img instanceof HTMLVideoElement) {
                        img.muted = true;
                        img.autoplay = true;
                        img.play();
                    }
                });

                this.setDirtyCanvas(true, true);
            });
    };

    chainCallback(nodeType.prototype, "onExecuted", function (message) {
        if (message?.video_url) {
            this.images = message?.video_url;
            this.setDirtyCanvas(true);
        }
    });
}

app.registerExtension({
    name: "Light-Tool: PreviewVideo",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "Light-Tool: PreviewVideo" || nodeData.name === "Light-Tool: SaveVideo") {
            addVideoPreview(nodeType);
        }
    },
});