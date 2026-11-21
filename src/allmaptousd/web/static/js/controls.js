// Mouse/keyboard interaction, AGV/origin controls, CAD layer list.
// Classic script: shares the global scope with the other js/*.js files (see index.html).

// --- Mouse & Keyboard ---
canvas.addEventListener("mousedown", (e) => {
  const rect = canvas.getBoundingClientRect();
  const sx = e.clientX - rect.left, sy = e.clientY - rect.top;
  const wp = screenToWorld(sx, sy);

  if (rulerActive && e.button === 0) {
    if (!rulerP1 || (rulerP1 && rulerP2)) { rulerP1 = wp; rulerP2 = wp; }
    else { rulerP2 = wp; }
    render();
    return;
  }

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
  const rect = canvas.getBoundingClientRect();
  const sx = e.clientX - rect.left, sy = e.clientY - rect.top;
  const wp = screenToWorld(sx, sy);
  document.getElementById("cursorCoord").textContent = `(${wp.x.toFixed(2)}, ${wp.y.toFixed(2)}) m`;

  if (rulerActive && rulerP1 && e.buttons === 1) {
    rulerP2 = wp; render(); return;
  }
  if (dragMode === "agv") {
    agv.x = wp.x + agvDragOffset.x; agv.y = wp.y + agvDragOffset.y; render();
  } else if (dragMode === "pan") {
    const dx = e.clientX - lastMouse.x, dy = e.clientY - lastMouse.y;
    lastMouse = { x: e.clientX, y: e.clientY };
    cam.cx -= dx / cam.scale; cam.cy += dy / cam.scale;
    render();
  }
});

window.addEventListener("mouseup", () => { dragMode = null; canvas.classList.remove("dragging"); });

canvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  if (e.shiftKey) { rotateAgv(e.deltaY > 0 ? -5 : 5); return; }
  const rect = canvas.getBoundingClientRect();
  const sx = e.clientX - rect.left, sy = e.clientY - rect.top;
  const before = screenToWorld(sx, sy);
  cam.scale = Math.max(3, Math.min(600, cam.scale * (e.deltaY < 0 ? 1.15 : 1 / 1.15)));
  const after = screenToWorld(sx, sy);
  cam.cx += before.x - after.x; cam.cy += before.y - after.y;
  render();
}, { passive: false });

window.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  if (e.key === "q" || e.key === "Q") rotateAgv(5);
  else if (e.key === "e" || e.key === "E") rotateAgv(-5);
  else if (e.key === "r" || e.key === "R") toggleRuler();
  else if (e.key === "f" || e.key === "F") fitToView();
  else if (e.key === "b" || e.key === "B") toggleSidebar();
});

function toggleRuler() {
  rulerActive = !rulerActive;
  document.getElementById("rulerBtn").classList.toggle("active", rulerActive);
  canvas.classList.toggle("ruler-mode", rulerActive);
  if (!rulerActive) { rulerP1 = null; rulerP2 = null; }
  render();
}

function onAgvInput() {
  agv.length = Math.max(0.1, parseFloat(document.getElementById("agvL").value) || 1.2);
  agv.width = Math.max(0.1, parseFloat(document.getElementById("agvW").value) || 0.8);
  agv.height = Math.max(0.1, parseFloat(document.getElementById("agvH").value) || 0.6);
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
  field.style.display = layers && layers.length ? "flex" : "none";
  document.getElementById("layerCount").textContent = (layers || []).length;
  for (const name of (layers || [])) {
    const row = document.createElement("div");
    row.className = "layer-row";

    const sw = document.createElement("label");
    sw.className = "switch";
    sw.title = "Include this layer";
    const cb = Object.assign(document.createElement("input"), { type: "checkbox", checked: true });
    cb.dataset.layer = name;
    cb.addEventListener("change", () => row.classList.toggle("off", !cb.checked));
    sw.append(cb, Object.assign(document.createElement("span"), { className: "slider" }));

    const nm = Object.assign(document.createElement("span"), { className: "layer-name", textContent: name, title: name });

    const stepper = document.createElement("div");
    stepper.className = "stepper";
    const h = Object.assign(document.createElement("input"), {
      type: "number", step: "0.1", min: "0.05", max: "20", placeholder: "wall"
    });
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
