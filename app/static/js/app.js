"use strict";

// Keep API coordinates and image pixels in the same canvas coordinate system.
// CSS scales the complete canvas (image, boxes and labels) together on resize.
const MAX_BYTES = 10 * 1024 * 1024;
const $ = (id) => document.getElementById(id);
const canvas = $("result-canvas");
const context = canvas.getContext("2d");
let selectedFile = null;
let previewImage = null;
let imageURL = null;
let selectionVersion = 0;
let busy = false;
let modelReady = false;
let requestController = null;

function showError(message = "") {
  $("error-message").textContent = message;
  $("error-message").hidden = !message;
}

function updateButtons() {
  $("detect-button").disabled = !selectedFile || !previewImage || busy || !modelReady;
  $("choose-image").disabled = busy;
  $("file-input").disabled = busy;
  $("loading").hidden = !busy;
}

async function checkHealth() {
  try {
    const response = await fetch("/health", { cache: "no-store", signal: AbortSignal.timeout(10000) });
    const data = await response.json();
    modelReady = response.ok && data.status === "ok";
  } catch {
    modelReady = false;
  }
  $("status-text").textContent = modelReady ? "Model Ready" : "Model Unavailable";
  $("model-status").className = `model-status ${modelReady ? "ready" : "unavailable"}`;
  if (!modelReady && selectedFile && !busy) {
    showError("YOLOv4 model is unavailable. Please check models/yolov4.weights.");
  }
  updateButtons();
}

function resetSelection() {
  // A late response from an earlier selection must not annotate a new image.
  selectionVersion += 1;
  requestController?.abort();
  requestController = null;
  busy = false;
  selectedFile = null;
  previewImage = null;
  if (imageURL) URL.revokeObjectURL(imageURL);
  imageURL = null;
  $("file-input").value = "";
  $("selected-filename").hidden = true;
  $("clear-button").hidden = true;
  $("results-section").hidden = true;
  canvas.hidden = true;
  canvas.width = canvas.height = 1;
  $("empty-state").hidden = false;
  $("image-heading").textContent = "Image preview";
  $("image-caption").textContent = "Your image and detection results will appear here.";
  $("image-badge").textContent = "NO IMAGE";
  $("image-badge").className = "tag neutral";
  $("image-dimensions").textContent = "Original resolution preserved";
  showError();
  updateButtons();
}

async function selectFile(file) {
  if (!file || busy) return;
  resetSelection();
  if (!/\.(jpe?g|png)$/i.test(file.name) || !["image/jpeg", "image/png"].includes(file.type)) {
    showError("Please upload a JPG, JPEG, or PNG image.");
    return;
  }
  if (file.size > MAX_BYTES) {
    showError("Image must be 10 MiB or smaller.");
    return;
  }
  const version = selectionVersion;
  selectedFile = file;
  $("selected-filename").textContent = file.name;
  $("selected-filename").hidden = false;
  $("clear-button").hidden = false;
  imageURL = URL.createObjectURL(file);
  const image = new Image();
  image.src = imageURL;
  try {
    await image.decode();
    if (version !== selectionVersion) return;
    previewImage = image;
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    context.drawImage(image, 0, 0);
    canvas.hidden = false;
    $("empty-state").hidden = true;
    $("image-badge").textContent = "PREVIEW";
    $("image-caption").textContent = "Image loaded. Ready for object detection.";
    $("image-dimensions").textContent = `${canvas.width} × ${canvas.height} px`;
    if (!modelReady) showError("YOLOv4 model is unavailable. Please check models/yolov4.weights.");
    updateButtons();
  } catch {
    if (version !== selectionVersion) return;
    resetSelection();
    showError("This image could not be opened. Please choose a valid JPG, JPEG, or PNG image.");
  }
}

function drawDetections(detections) {
  context.drawImage(previewImage, 0, 0);
  // Scale label styling with the source resolution so it remains legible
  // when a multi-megapixel image is displayed in a smaller preview.
  const fontSize = Math.max(13, Math.round(canvas.width / 65));
  const padding = Math.max(4, Math.round(fontSize / 3));
  context.font = `600 ${fontSize}px "Segoe UI", sans-serif`;
  context.lineWidth = Math.max(2, canvas.width / 450);
  context.textBaseline = "top";
  for (const detection of detections) {
    const { x, y, width, height } = detection.bounding_box;
    context.strokeStyle = "#00e5a0";
    context.strokeRect(x, y, width, height);
    const label = `${detection.class_name} ${(detection.confidence * 100).toFixed(1)}%`;
    const labelWidth = context.measureText(label).width + padding * 2;
    const labelHeight = fontSize + padding * 2;
    const labelX = Math.max(0, Math.min(x, canvas.width - labelWidth));
    const labelY = Math.max(0, Math.min(y - labelHeight, canvas.height - labelHeight));
    context.fillStyle = "#00e5a0";
    context.fillRect(labelX, labelY, labelWidth, labelHeight);
    context.fillStyle = "#083d32";
    context.fillText(label, labelX + padding, labelY + padding);
  }
}

function displayResults(result) {
  drawDetections(result.detections);
  $("image-heading").textContent = "Detection result";
  $("image-caption").textContent = "Bounding boxes and confidence values from YOLOv4.";
  $("image-badge").textContent = "COMPLETED";
  $("image-badge").className = "tag";
  $("total-detections").textContent = result.total_detections;
  $("inference-time").replaceChildren(document.createTextNode(result.inference_time_ms.toFixed(2) + " "));
  const unit = document.createElement("small");
  unit.textContent = "ms";
  $("inference-time").append(unit);
  $("category-counts").replaceChildren();
  for (const [name, count] of Object.entries(result.counts)) {
    const category = document.createElement("span");
    category.className = "category";
    category.textContent = name;
    const amount = document.createElement("strong");
    amount.textContent = count;
    category.append(amount);
    $("category-counts").append(category);
  }
  $("detection-rows").replaceChildren();
  for (const detection of result.detections) {
    const row = document.createElement("tr");
    const box = detection.bounding_box;
    // Use textContent for response data and filenames, never HTML interpolation.
    for (const value of [detection.class_name, `${(detection.confidence * 100).toFixed(1)}%`,
                         `${box.x}, ${box.y}, ${box.width}, ${box.height}`]) {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.append(cell);
    }
    $("detection-rows").append(row);
  }
  if (!result.detections.length) {
    $("category-counts").textContent = "No objects above the confidence threshold.";
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 3;
    cell.textContent = "No detections. Try another image.";
    row.append(cell);
    $("detection-rows").append(row);
  }
  $("results-section").hidden = false;
}

async function runDetection() {
  if (!selectedFile || !previewImage || busy || !modelReady) return;
  const version = selectionVersion;
  const controller = new AbortController();
  requestController = controller;
  busy = true;
  showError();
  $("results-section").hidden = true;
  $("image-heading").textContent = "Image preview";
  $("image-caption").textContent = "Running YOLOv4 detection...";
  $("image-badge").textContent = "PROCESSING";
  $("image-badge").className = "tag neutral";
  context.drawImage(previewImage, 0, 0);
  updateButtons();
  const form = new FormData();
  form.append("file", selectedFile);
  try {
    const response = await fetch("/detect", { method: "POST", body: form, signal: controller.signal });
    if (version !== selectionVersion) return;
    if (!response.ok) {
      const messages = {
        503: "YOLOv4 model is unavailable. Please check models/yolov4.weights.",
        415: "Please upload a JPG, JPEG, or PNG image.",
        413: "Image must be 10 MiB or smaller.",
        400: "This image could not be processed. Please choose a valid JPG, JPEG, or PNG image."
      };
      const error = new Error("Detection request failed");
      error.userMessage = messages[response.status];
      throw error;
    }
    const result = await response.json();
    if (version !== selectionVersion) return;
    displayResults(result);
  } catch (error) {
    if (version !== selectionVersion || error.name === "AbortError") return;
    $("image-caption").textContent = "Image loaded. You can try detection again.";
    $("image-badge").textContent = "PREVIEW";
    showError(error.userMessage || "Detection failed. Please check the server and try again.");
  } finally {
    if (version === selectionVersion) {
      busy = false;
      requestController = null;
      updateButtons();
      void checkHealth();
    }
  }
}

$("choose-image").addEventListener("click", () => $("file-input").click());
$("file-input").addEventListener("change", (event) => void selectFile(event.target.files[0]));
$("detect-button").addEventListener("click", () => void runDetection());
$("clear-button").addEventListener("click", resetSelection);
$("refresh-status").addEventListener("click", () => { showError(); void checkHealth(); });
const dropZone = $("drop-zone");
// Prevent the browser from navigating away when a file misses the drop target.
for (const name of ["dragover", "drop"]) document.addEventListener(name, (event) => event.preventDefault());
dropZone.addEventListener("dragover", () => { if (!busy) dropZone.classList.add("dragging"); });
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragging"));
dropZone.addEventListener("drop", (event) => {
  dropZone.classList.remove("dragging");
  if (busy) return;
  if (event.dataTransfer.files.length !== 1) {
    showError("Please choose one image at a time.");
    return;
  }
  void selectFile(event.dataTransfer.files[0]);
});
void checkHealth();
