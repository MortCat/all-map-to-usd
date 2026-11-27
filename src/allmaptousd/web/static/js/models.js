// 3D equipment models: import dialog, model tray (3D thumbnails), placement on the map.
// Classic script: shares the global scope with the other js/*.js files (see index.html).

// Imported models (the server's library) and where they stand on the map.
const modelLibrary = new Map();  // id -> { summary, view: {yaw, pitch}, top: canvas|null, card }
let placements = [];             // { uid, model_id, name, x, y, yaw_deg }
let placementSeq = 0;
let selectedPlacement = null;
let modelGhost = null;           // { id, world } while a model is dragged from the tray onto the map
let modelMove = null;            // { placement, dx, dy } while a placed model is dragged

const modelTray = document.getElementById("modelTray");
const modelTrayList = document.getElementById("modelTrayList");
const modelDialog = document.getElementById("modelDialog");
const modelForm = document.getElementById("modelForm");
const modelMenu = document.getElementById("modelMenu");
const DRAG_THRESHOLD_PX = 4;  // a smaller move is a click, not a drag

// --- Library ---------------------------------------------------------------------------------

function syncModelLibrary(summaries) {
  const ids = new Set(summaries.map((s) => s.id));
  for (const id of [...modelLibrary.keys()]) if (!ids.has(id)) forgetModel(id);
  for (const summary of summaries) {
    if (!modelLibrary.has(summary.id)) addModelToLibrary(summary);
  }
  updateModelTray();
}

function addModelToLibrary(summary) {
  const entry = { summary, view: { yaw: -Math.PI / 3, pitch: Math.PI / 6 }, top: null, card: null };
  modelLibrary.set(summary.id, entry);
  entry.card = buildModelCard(entry);
  modelTrayList.appendChild(entry.card);
  fetch(`/api/models/${summary.id}/preview`).then((r) => (r.ok ? r.json() : null)).then(async (data) => {
    if (!data || !viewer3d || !modelLibrary.has(summary.id)) return;
    await viewer3d.upload(summary.id, {
      positions: decodeBase64(data.positions, Float32Array),
      indices: decodeBase64(data.indices, Uint32Array),
      colors: decodeBase64(data.colors, Uint8Array),
      uvs: decodeBase64(data.uvs, Float32Array),
      groups: data.groups,
      textures: data.textures,
    }, (i) => `/api/models/${summary.id}/textures/${i}`);
    if (!modelLibrary.has(summary.id)) return;
    const [dx, dy] = summary.size;
    entry.top = viewer3d.renderTop(summary.id, summary.size, Math.min(512 / Math.max(dx, dy, 1e-3), 400));
    drawModelThumb(entry);
    render();
  }).catch(() => {});
}

function forgetModel(id) {
  modelLibrary.get(id)?.card.remove();
  modelLibrary.delete(id);
  placements = placements.filter((p) => p.model_id !== id);
  if (selectedPlacement && !placements.includes(selectedPlacement)) selectedPlacement = null;
}

function decodeBase64(text, ArrayType) {
  const bytes = Uint8Array.from(atob(text), (c) => c.charCodeAt(0));
  return new ArrayType(bytes.buffer);
}

async function removeModel(id) {
  const entry = modelLibrary.get(id);
  const used = placements.filter((p) => p.model_id === id).length;
  if (used && !window.confirm(`Remove “${entry.summary.name}” and its ${used} placement${used > 1 ? "s" : ""} from the map?`)) return;
  await fetch(`/api/models/${id}`, { method: "DELETE" }).catch(() => {});
  forgetModel(id);
  updateModelTray();
  render();
}

// --- Tray ------------------------------------------------------------------------------------

function updateModelTray() {
  const show = editor.open && editor.tool === "model";
  modelTray.hidden = !show;
  document.getElementById("modelTrayEmpty").hidden = modelLibrary.size > 0;
  if (!show) closeModelMenu();
}

function buildModelCard(entry) {
  const { summary } = entry;
  const card = document.createElement("div");
  card.className = "model-card";
  card.title = `${summary.name}\nDrag to rotate · hold, then drag onto the map to place`;

  const thumb = document.createElement("canvas");
  thumb.className = "model-thumb";
  thumb.width = 360;
  thumb.height = 240;
  thumb.setAttribute("aria-label", `${summary.name} 3D preview`);

  const meta = document.createElement("div");
  meta.className = "model-meta";
  const [dx, dy, dz] = summary.size;
  meta.append(
    Object.assign(document.createElement("span"), { className: "model-name", textContent: summary.name }),
    Object.assign(document.createElement("span"), {
      className: "model-size", textContent: `${dx.toFixed(2)} × ${dy.toFixed(2)} × ${dz.toFixed(2)} m`,
    }),
  );
  if (summary.warning) {
    meta.append(Object.assign(document.createElement("span"), { className: "model-warning", textContent: summary.warning }));
  }

  const remove = Object.assign(document.createElement("button"), {
    type: "button", className: "model-remove", textContent: "×", title: "Remove from the library",
  });
  remove.setAttribute("aria-label", `Remove ${summary.name}`);
  remove.addEventListener("click", () => removeModel(summary.id));

  card.append(thumb, meta, remove);
  attachThumbGestures(entry, thumb, card);
  entry.thumb = thumb;
  drawModelThumb(entry);
  return card;
}

function drawModelThumb(entry) {
  const { thumb, summary, view } = entry;
  if (!thumb) return;
  if (viewer3d?.has(summary.id) && viewer3d.drawOrbit(summary.id, summary.size, thumb, view.yaw, view.pitch)) return;
  // No 3D preview (yet): show the top-view outline instead.
  const c = thumb.getContext("2d");
  c.clearRect(0, 0, thumb.width, thumb.height);
  const [dx, dy] = summary.size;
  const scale = 0.8 * Math.min(thumb.width / Math.max(dx, 1e-3), thumb.height / Math.max(dy, 1e-3));
  c.strokeStyle = "#a78bfa";
  c.lineWidth = 2;
  for (const poly of summary.footprint) {
    c.beginPath();
    poly.forEach(([x, y], i) => {
      const sx = thumb.width / 2 + x * scale, sy = thumb.height / 2 - y * scale;
      i ? c.lineTo(sx, sy) : c.moveTo(sx, sy);
    });
    c.closePath();
    c.stroke();
  }
}

// Left-drag a thumbnail onto the map to place the model; right-drag (or Shift + left-drag, for
// trackpads) turns the 3D picture.
function attachThumbGestures(entry, thumb, card) {
  let gesture = null;
  const end = () => {
    if (!gesture) return;
    card.classList.remove("lifted");
    document.body.classList.remove("placing-model");
    gesture = null;
    modelGhost = null;
    render();
  };
  thumb.addEventListener("contextmenu", (e) => e.preventDefault());  // right button turns the picture
  thumb.addEventListener("pointerdown", (e) => {
    const turn = e.button === 2 || (e.button === 0 && e.shiftKey);
    if (!turn && (e.button !== 0 || !mapData)) return;
    e.preventDefault();
    try { thumb.setPointerCapture(e.pointerId); } catch { /* pointer already gone */ }
    gesture = { mode: turn ? "rotate" : "pending", x: e.clientX, y: e.clientY };
  });
  thumb.addEventListener("pointermove", (e) => {
    if (!gesture) return;
    const dx = e.clientX - gesture.x, dy = e.clientY - gesture.y;
    if (gesture.mode === "pending" && Math.hypot(dx, dy) > DRAG_THRESHOLD_PX) {
      gesture.mode = "place";
      card.classList.add("lifted");
      document.body.classList.add("placing-model");
    }
    if (gesture.mode === "rotate") {
      entry.view.yaw -= dx * 0.012;
      entry.view.pitch = Math.max(-1.45, Math.min(1.45, entry.view.pitch + dy * 0.012));
      gesture.x = e.clientX;
      gesture.y = e.clientY;
      drawModelThumb(entry);
    } else if (gesture.mode === "place") {
      const world = canvasWorldAt(e.clientX, e.clientY);
      modelGhost = world ? { id: entry.summary.id, world } : null;
      render();
    }
  });
  thumb.addEventListener("pointerup", (e) => {
    if (gesture?.mode === "place") {
      const world = canvasWorldAt(e.clientX, e.clientY);
      if (world) addPlacement(entry.summary, world);
    }
    end();
  });
  thumb.addEventListener("pointercancel", end);
}

function canvasWorldAt(clientX, clientY) {
  const rect = canvas.getBoundingClientRect();
  if (clientX < rect.left || clientX > rect.right || clientY < rect.top || clientY > rect.bottom) return null;
  const under = document.elementFromPoint(clientX, clientY);
  if (under !== canvas) return null;  // over the tray, a menu or the HUD
  return screenToWorld(clientX - rect.left, clientY - rect.top);
}

function addPlacement(summary, world) {
  const base = summary.name.replace(/\s+/g, "_");
  const taken = new Set(placements.map((p) => p.name));
  let n = 1;
  while (taken.has(`${base}_${n}`)) n += 1;
  const placement = { uid: ++placementSeq, model_id: summary.id, name: `${base}_${n}`, x: world.x, y: world.y, yaw_deg: 0 };
  placements.push(placement);
  selectedPlacement = placement;
  render();
}

function setPlacements(equipment) {
  placements = (equipment || []).filter((e) => modelLibrary.has(e.model_id)).map((e) => ({ ...e, uid: ++placementSeq }));
  selectedPlacement = null;
  closeModelMenu();
}

function collectPlacements() {
  return placements.map(({ model_id, name, x, y, yaw_deg }) => ({ model_id, name, x, y, yaw_deg }));
}

// --- Geometry on the map ---------------------------------------------------------------------

function placementPolygons(p) {
  const summary = modelLibrary.get(p.model_id)?.summary;
  if (!summary) return [];
  const rad = (p.yaw_deg * Math.PI) / 180, c = Math.cos(rad), s = Math.sin(rad);
  return summary.footprint.map((poly) => poly.map(([x, y]) => [p.x + x * c - y * s, p.y + x * s + y * c]));
}

function placementAt(world) {
  for (let i = placements.length - 1; i >= 0; i--) {
    if (placementPolygons(placements[i]).some((poly) => pointInPolygon(world.x, world.y, poly))) return placements[i];
  }
  return null;
}

function drawPlacement(p, alpha = 1) {
  const entry = modelLibrary.get(p.model_id);
  if (!entry) return;
  const [dx, dy] = entry.summary.size;
  const centre = worldToScreen(p.x, p.y);
  const selected = p === selectedPlacement;
  ctx.save();
  ctx.globalAlpha = alpha;
  if (entry.top) {
    ctx.save();
    ctx.translate(centre.x, centre.y);
    ctx.rotate((-p.yaw_deg * Math.PI) / 180);
    ctx.drawImage(entry.top, -dx / 2 * cam.scale, -dy / 2 * cam.scale, dx * cam.scale, dy * cam.scale);
    ctx.restore();
  }
  ctx.strokeStyle = selected ? "#ddd6fe" : "#a78bfa";
  ctx.fillStyle = "rgba(167, 139, 250, 0.22)";
  ctx.lineWidth = selected ? 2 : 1.4;
  if (selected) ctx.setLineDash([5, 3]);
  for (const poly of placementPolygons(p)) {
    ctx.beginPath();
    poly.forEach(([x, y], i) => {
      const s = worldToScreen(x, y);
      i ? ctx.lineTo(s.x, s.y) : ctx.moveTo(s.x, s.y);
    });
    ctx.closePath();
    if (!entry.top) ctx.fill();
    ctx.stroke();
  }
  ctx.setLineDash([]);
  if (cam.scale > 25 || selected) {
    ctx.font = "600 11px sans-serif";
    ctx.textAlign = "center";
    ctx.fillStyle = "#ddd6fe";
    ctx.fillText(p.name, centre.x, centre.y - (Math.max(dx, dy) / 2) * cam.scale - 6);
  }
  ctx.restore();
}

function drawPlacements() {
  if (!document.getElementById("layerModels").checked) return;
  for (const p of placements) drawPlacement(p);
  if (modelGhost) {
    drawPlacement({ model_id: modelGhost.id, name: "", x: modelGhost.world.x, y: modelGhost.world.y, yaw_deg: 0 }, 0.6);
  }
}

// --- Moving, turning and deleting placed models ----------------------------------------------

function startModelMove(world) {
  const hit = placementAt(world);
  selectedPlacement = hit;
  closeModelMenu();
  render();
  if (!hit) return false;
  modelMove = { placement: hit, dx: hit.x - world.x, dy: hit.y - world.y };
  return true;
}

function moveModel(world) {
  if (!modelMove) return;
  modelMove.placement.x = world.x + modelMove.dx;
  modelMove.placement.y = world.y + modelMove.dy;
  render();
}

function openModelMenu(placement, clientX, clientY) {
  selectedPlacement = placement;
  const wrap = canvas.parentElement.getBoundingClientRect();
  document.getElementById("modelMenuTitle").textContent = placement.name;
  document.getElementById("modelMenuYaw").value = Math.round(placement.yaw_deg);
  modelMenu.hidden = false;
  const x = Math.min(clientX - wrap.left, wrap.width - modelMenu.offsetWidth - 8);
  const y = Math.min(clientY - wrap.top, wrap.height - modelMenu.offsetHeight - 8);
  modelMenu.style.left = `${Math.max(8, x)}px`;
  modelMenu.style.top = `${Math.max(8, y)}px`;
  document.getElementById("modelMenuYaw").focus();
  render();
}

function closeModelMenu() {
  modelMenu.hidden = true;
}

function setSelectedYaw(deg) {
  if (!selectedPlacement || !Number.isFinite(deg)) return;
  selectedPlacement.yaw_deg = ((deg + 180) % 360 + 360) % 360 - 180;
  document.getElementById("modelMenuYaw").value = Math.round(selectedPlacement.yaw_deg);
  render();
}

function turnSelectedModel(delta) {
  if (selectedPlacement) setSelectedYaw(selectedPlacement.yaw_deg + delta);
}

function deleteSelectedModel() {
  if (!selectedPlacement) return;
  placements = placements.filter((p) => p !== selectedPlacement);
  selectedPlacement = null;
  closeModelMenu();
  render();
}

document.getElementById("modelMenuYaw").addEventListener("input", (e) => setSelectedYaw(e.target.valueAsNumber));
modelMenu.addEventListener("keydown", (e) => {
  if (e.key === "Escape") { closeModelMenu(); canvas.focus?.(); }
});
window.addEventListener("pointerdown", (e) => {
  if (!modelMenu.hidden && !modelMenu.contains(e.target)) closeModelMenu();
}, true);

// --- Import dialog ---------------------------------------------------------------------------

let modelImport = null;  // { token, name } of the inspected file

function openModelDialog() {
  if (!mapData || modelDialog.open) return;
  modelForm.reset();
  modelImport = null;
  document.getElementById("modelPick").hidden = true;
  document.getElementById("modelError").hidden = true;
  document.getElementById("modelMissing").hidden = true;
  document.getElementById("modelImportBtn").disabled = true;
  document.getElementById("modelFileLabel").textContent = "Choose a USD, STEP or STL file";
  modelDialog.showModal();
}

function showModelError(message) {
  const error = document.getElementById("modelError");
  error.textContent = message;
  error.hidden = !message;
}

// One model file, a .zip, or every file of a folder (named by its path in the folder, so the model's
// textures and materials keep their places).
async function inspectModelUpload(files, label) {
  if (!files.length) return;
  showModelError("");
  document.getElementById("modelPick").hidden = true;
  document.getElementById("modelMissing").hidden = true;
  document.getElementById("modelImportBtn").disabled = true;
  document.getElementById("modelFileLabel").textContent = `Reading ${label}…`;
  try {
    const form = new FormData();
    for (const file of files) form.append("files", file, file.webkitRelativePath || file.name);
    const info = await postJson("/api/models/inspect", form, "These files could not be read.");
    document.getElementById("modelFileLabel").textContent = label;
    showModelInfo(info);
  } catch (err) {
    document.getElementById("modelFileLabel").textContent = "Choose a USD, STEP or STL file";
    showModelError(err.message);
  }
}

function showModelInfo(info) {
  modelImport = { token: info.token, name: info.name };
  const main = document.getElementById("modelMain");
  main.replaceChildren(...info.candidates.map((path) => new Option(path, path, false, path === info.main)));
  document.getElementById("modelMainField").hidden = info.candidates.length < 2;
  const missing = document.getElementById("modelMissing");
  const listed = info.missing.slice(0, 4).join(", ") + (info.missing.length > 4 ? ` and ${info.missing.length - 4} more` : "");
  missing.textContent = info.missing.length
    ? `This model uses files that were not uploaded: ${listed}. Without them it imports without those textures `
      + "or parts. To include them, choose the model's folder or a .zip of it."
    : "";
  missing.hidden = !info.missing.length;
  renderModelTree(info);
  document.getElementById("modelPick").hidden = false;
  document.getElementById("modelImportBtn").disabled = false;
}

document.getElementById("modelFile").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (file) await inspectModelUpload([file], file.name);
  e.target.value = "";
});

document.getElementById("modelFolder").addEventListener("change", async (e) => {
  const files = [...e.target.files];
  const folder = files[0]?.webkitRelativePath.split("/")[0] || "folder";
  await inspectModelUpload(files, `${folder} (${files.length} files)`);
  e.target.value = "";
});

document.getElementById("modelMain").addEventListener("change", async (e) => {
  if (!modelImport) return;
  showModelError("");
  document.getElementById("modelImportBtn").disabled = true;
  try {
    showModelInfo(await postJson("/api/models/select", { token: modelImport.token, main: e.target.value },
      "This file could not be read."));
  } catch (err) {
    showModelError(err.message);
  }
});

function renderModelTree(info) {
  const unitLabels = { m: "m", cm: "cm", mm: "mm", in: "inch" };
  document.getElementById("modelUnitAuto").textContent = `Auto (${unitLabels[info.units]})`;
  document.getElementById("modelUpAuto").textContent = `Auto (${info.up_axis.toUpperCase()} up)`;
  document.getElementById("modelUnitNote").textContent =
    `Units: ${unitLabels[info.units]}, ${info.units_note}. ${info.up_axis.toUpperCase()} axis up.`;
  const whole = document.getElementById("modelWhole");
  whole.checked = true;
  const tree = document.getElementById("modelTree");
  tree.innerHTML = "";
  tree.append(buildTreeList(info.tree, 0));
  if (info.truncated) {
    tree.append(Object.assign(document.createElement("p"), {
      className: "model-tree-note", textContent: "Only the first parts are listed.",
    }));
  }
  const single = info.format === "stl";
  document.getElementById("modelWholeRow").hidden = single;
  tree.hidden = single;
  syncTreeChecks();
}

function buildTreeList(nodes, depth) {
  const list = document.createElement("ul");
  list.className = "model-tree-list";
  for (const node of nodes) {
    const item = document.createElement("li");
    const row = document.createElement("div");
    row.className = "model-tree-row";
    const toggle = Object.assign(document.createElement("button"), { type: "button", className: "tree-toggle" });
    toggle.hidden = !node.children.length;
    toggle.setAttribute("aria-label", `Show parts of ${node.name}`);
    const box = Object.assign(document.createElement("input"), { type: "checkbox" });
    box.dataset.path = node.path;
    const label = document.createElement("label");
    label.append(box, Object.assign(document.createElement("span"), { textContent: node.name }));
    if (node.type) label.append(Object.assign(document.createElement("small"), { textContent: node.type }));
    row.append(toggle, label);
    item.append(row);
    if (node.children.length) {
      const children = buildTreeList(node.children, depth + 1);
      const open = depth < 1;
      children.hidden = !open;
      toggle.textContent = open ? "▾" : "▸";
      toggle.setAttribute("aria-expanded", String(open));
      toggle.addEventListener("click", () => {
        children.hidden = !children.hidden;
        toggle.textContent = children.hidden ? "▸" : "▾";
        toggle.setAttribute("aria-expanded", String(!children.hidden));
      });
      item.append(children);
    }
    list.append(item);
  }
  return list;
}

// A checked part brings everything below it; "Whole model" brings everything.
// Parts below a chosen one show as chosen and locked; they get their own choice back when it is cleared.
function syncTreeChecks(event) {
  const changed = event?.target;
  if (changed?.dataset.path) changed.dataset.own = changed.checked ? "1" : "";
  const whole = document.getElementById("modelWhole").checked;
  const tree = document.getElementById("modelTree");
  tree.classList.toggle("disabled", whole);
  const walk = (list, implied) => {
    for (const item of list.children) {
      const box = item.querySelector(":scope > .model-tree-row input");
      if (!box) continue;
      box.disabled = implied;
      box.checked = implied || box.dataset.own === "1";
      const children = item.querySelector(":scope > ul");
      if (children) walk(children, implied || box.checked);
    }
  };
  const root = tree.querySelector(":scope > ul");
  if (root) walk(root, whole);
}

function selectedModelPaths() {
  if (document.getElementById("modelWhole").checked) return [];
  const tree = document.getElementById("modelTree");
  return [...tree.querySelectorAll("input[type=checkbox]:checked:not(:disabled)")].map((b) => b.dataset.path);
}

document.getElementById("modelWhole").addEventListener("change", syncTreeChecks);
document.getElementById("modelTree").addEventListener("change", syncTreeChecks);

modelForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (e.submitter?.value !== "import") { modelDialog.close(); return; }
  if (!modelImport) return;
  const paths = selectedModelPaths();
  if (!document.getElementById("modelWhole").checked && !paths.length) {
    showModelError("Choose the parts to import, or tick Whole model.");
    return;
  }
  const button = document.getElementById("modelImportBtn");
  button.disabled = true;
  button.textContent = "Importing…";
  showModelError("");
  try {
    const summary = await postJson("/api/models/import", {
      token: modelImport.token,
      paths,
      unit: document.getElementById("modelUnit").value,
      up_axis: document.getElementById("modelUp").value,
    }, "The model could not be imported.");
    if (!modelLibrary.has(summary.id)) addModelToLibrary(summary);
    updateModelTray();
    modelDialog.close();
    const files = summary.files ? ` with ${summary.files} texture and material file${summary.files === 1 ? "" : "s"}` : "";
    const missing = summary.missing_files.length
      ? ` ${summary.missing_files.length} file${summary.missing_files.length === 1 ? " it uses was" : "s it uses were"} not uploaded.` : "";
    showToast(summary.warning || `Imported ${summary.name}${files}.${missing} Drag its picture onto the map.`);
  } catch (err) {
    showModelError(err.message);
  } finally {
    button.disabled = false;
    button.textContent = "Import";
  }
});
