// Mouse/keyboard interaction, AGV/origin controls, CAD layer list.
// Classic script: shares the global scope with the other js/*.js files (see index.html).

function canvasPointer(e) {
  const rect = canvas.getBoundingClientRect();
  return [e.clientX - rect.left, e.clientY - rect.top];
}

let rightDown = null;  // where the right button went down: a click without a drag opens the model menu

function stopDrag() {
  if (dragMode === "ruler" && rulerP2) rulerAwaitingSecond = false;
  modelMove = null;
  dragMode = null;
  canvas.classList.remove("dragging");
}

// --- Mouse & Keyboard ---
canvas.addEventListener("mousedown", (e) => {
  const wp = screenToWorld(...canvasPointer(e));
  if (mapData && rulerActive && e.button === 0) {
    if (rulerAwaitingSecond) {
      rulerP2 = wp;
      rulerAwaitingSecond = false;
    } else {
      rulerP1 = wp;
      rulerP2 = null;
      rulerAwaitingSecond = true;
      dragMode = "ruler";
      lastMouse = { x: e.clientX, y: e.clientY };
    }
    render();
    return;
  }

  if (editor.open && editor.tool === "model" && e.button === 0) {
    if (startModelMove(wp)) {
      dragMode = "model";
      canvas.classList.add("dragging");
      return;
    }
  } else if (editor.open && e.button === 0) {
    startEditStroke(wp);
    return;
  }
  if (e.button === 2) rightDown = { x: e.clientX, y: e.clientY };
  if (e.button === 1 || (e.button === 2 && editor.open)) e.preventDefault();

  if (mapData && e.button === 0 && document.getElementById("layerAgv").checked) {
    if (pointInPolygon(wp.x, wp.y, getAgvCorners())) {
      dragMode = "agv";
      agvDragOffset = { x: agv.x - wp.x, y: agv.y - wp.y };
      canvas.classList.add("dragging");
      return;
    }
  }
  dragMode = "pan";
  lastMouse = { x: e.clientX, y: e.clientY };
  canvas.classList.add("dragging");
});

window.addEventListener("mousemove", (e) => {
  const wp = screenToWorld(...canvasPointer(e));
  document.getElementById("cursorCoord").textContent = `(${wp.x.toFixed(2)}, ${wp.y.toFixed(2)}) m`;

  if (dragMode === "ruler") {
    if (Math.hypot(e.clientX - lastMouse.x, e.clientY - lastMouse.y) >= 3 || rulerP2) {
      rulerP2 = wp;
      render();
    }
  } else if (dragMode === "edit") {
    moveEditStroke(wp);
  } else if (dragMode === "model") {
    moveModel(wp);
  } else if (dragMode === "agv") {
    agv.x = wp.x + agvDragOffset.x; agv.y = wp.y + agvDragOffset.y; render();
  } else if (dragMode === "pan") {
    const dx = e.clientX - lastMouse.x, dy = e.clientY - lastMouse.y;
    lastMouse = { x: e.clientX, y: e.clientY };
    cam.cx -= dx / cam.scale; cam.cy += dy / cam.scale;
    render();
  }
});

window.addEventListener("mouseup", (e) => {
  if (dragMode === "edit") {
    if (e.button !== 0) return;
    moveEditStroke(screenToWorld(...canvasPointer(e)));
    finishEditStroke();
  }
  stopDrag();
});

window.addEventListener("blur", () => {
  finishEditStroke();
  stopDrag();
});
canvas.addEventListener("contextmenu", (e) => {
  if (!editor.open) return;
  e.preventDefault();
  const clicked = rightDown && Math.hypot(e.clientX - rightDown.x, e.clientY - rightDown.y) <= 4;
  rightDown = null;
  if (editor.tool !== "model" || !clicked) return;
  const hit = placementAt(screenToWorld(...canvasPointer(e)));
  if (hit) openModelMenu(hit, e.clientX, e.clientY);
});

canvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  if (e.shiftKey) { rotateAgv(e.deltaY > 0 ? -5 : 5); return; }
  const [sx, sy] = canvasPointer(e);
  const before = screenToWorld(sx, sy);
  cam.scale = Math.max(3, Math.min(600, cam.scale * (e.deltaY < 0 ? 1.15 : 1 / 1.15)));
  const after = screenToWorld(sx, sy);
  cam.cx += before.x - after.x; cam.cy += before.y - after.y;
  render();
}, { passive: false });

window.addEventListener("keydown", (e) => {
  if (e.target.closest?.("dialog[open], input, select, textarea, [contenteditable]")) return;
  const key = e.key.toLowerCase();
  if (e.ctrlKey || e.metaKey) {
    if (editor.open && (key === "z" || key === "y")) {
      e.preventDefault();
      finishEditStroke();
      if (key === "y" || e.shiftKey) redoEdit();
      else undoEdit();
    }
    return;
  }
  if (e.repeat && key === "r") return;
  if ((key === "delete" || key === "backspace") && editor.open && editor.tool === "model" && selectedPlacement) {
    deleteSelectedModel();
    return;
  }
  if (key === "escape" && editor.stroke) {
    editor.stroke = null;
    dragMode = null;
    render();
    return;
  }
  if (key === "q") rotateAgv(5);
  else if (key === "e") rotateAgv(-5);
  else if (key === "r") toggleRuler();
  else if (key === "f") fitToView();
  else if (key === "b") toggleSidebar();
});

function toggleRuler() {
  finishEditStroke();
  stopDrag();
  setRulerActive(!rulerActive);
  updateEditorControls();
  render();
}

function setRulerActive(active) {
  rulerActive = active;
  const button = document.getElementById("rulerBtn");
  button.classList.toggle("active", rulerActive);
  button.setAttribute("aria-pressed", String(rulerActive));
  canvas.classList.toggle("ruler-mode", rulerActive);
  if (!rulerActive) {
    rulerP1 = rulerP2 = null;
    rulerAwaitingSecond = false;
  }
}

function onAgvInput() {
  agv.length = Math.max(0.1, parseFloat(document.getElementById("agvL").value) || 1.2);
  agv.width = Math.max(0.1, parseFloat(document.getElementById("agvW").value) || 0.8);
  agv.height = Math.max(0.1, parseFloat(document.getElementById("agvH").value) || 0.6);
  updateIsaacPullRange();
  agv.yawDeg = parseFloat(document.getElementById("agvYaw").value) || 0;
  agv.margin = Math.max(0, parseFloat(document.getElementById("agvMargin").value) || 0);
  document.getElementById("yawLabel").textContent = `${Math.round(agv.yawDeg)}°`;
  render();
}

function setOrigin(mode) {
  let x = 0, y = 0, yaw = 0;
  if (mode === "center" && mapData) {
    const b = mapData.bounds;
    x = 0.5 * (b.min_x + b.max_x); y = 0.5 * (b.min_y + b.max_y);
  } else if (mode === "agv") {
    x = agv.x; y = agv.y; yaw = agv.yawDeg;
  }
  document.getElementById("originX").value = x.toFixed(3);
  document.getElementById("originY").value = y.toFixed(3);
  document.getElementById("originYaw").value = Math.round(yaw);
  showToast(`USD origin set to (${x.toFixed(2)}, ${y.toFixed(2)}, ${Math.round(yaw)}°)`);
}

let _layerKey = "";
function renderLayerList(layers) {
  const key = (layers || []).join("\u0000");
  if (key === _layerKey) return;  // keep user choices across rebuilds of the same map
  _layerKey = key;
  const field = document.getElementById("layerField");
  const list = document.getElementById("layerList");
  list.innerHTML = "";
  field.hidden = !layers?.length;
  document.getElementById("layerCount").textContent = (layers || []).length;
  for (const name of (layers || [])) {
    const row = document.createElement("div");
    row.className = "layer-row";

    const sw = document.createElement("label");
    sw.className = "switch";
    sw.title = "Include this layer";
    const cb = Object.assign(document.createElement("input"), { type: "checkbox", checked: true });
    cb.setAttribute("aria-label", `Include layer ${name}`);
    cb.dataset.layer = name;
    cb.addEventListener("change", () => row.classList.toggle("off", !cb.checked));
    sw.append(cb, Object.assign(document.createElement("span"), { className: "slider" }));

    const nm = Object.assign(document.createElement("span"), { className: "layer-name", textContent: name, title: name });

    const stepper = document.createElement("div");
    stepper.className = "stepper";
    const h = Object.assign(document.createElement("input"), {
      type: "number", step: "0.1", min: "0.05", max: "20", placeholder: "wall"
    });
    h.setAttribute("aria-label", `Wall height for ${name} in meters`);
    h.dataset.layer = name;
    stepper.append(h, Object.assign(document.createElement("span"), { className: "unit", textContent: "m" }));

    row.append(sw, nm, stepper);
    list.appendChild(row);
  }
}

function collectLayerConfig() {
  const excluded = [], heights = {};
  document.querySelectorAll("#layerList input[type=checkbox]").forEach((cb) => {
    if (!cb.checked) excluded.push(cb.dataset.layer);
  });
  document.querySelectorAll("#layerList input[type=number]").forEach((inp) => {
    const v = parseFloat(inp.value);
    if (Number.isFinite(v) && v > 0) heights[inp.dataset.layer] = v;
  });
  return { excluded_layers: excluded, layer_heights: heights };
}

function setAgvYaw(deg) {
  agv.yawDeg = deg;
  document.getElementById("agvYaw").value = deg;
  document.getElementById("yawLabel").textContent = `${deg}°`;
  render();
}

function rotateAgv(deltaDeg) {
  const y = ((agv.yawDeg + deltaDeg + 180) % 360 + 360) % 360 - 180;
  setAgvYaw(Math.round(y));
}

function resetAgvToCenter() {
  if (!mapData) return;
  const b = mapData.bounds;
  agv.x = 0.5 * (b.min_x + b.max_x);
  agv.y = 0.5 * (b.min_y + b.max_y);
  setAgvYaw(0);
}
