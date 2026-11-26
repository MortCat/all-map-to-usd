// Keep drawn strokes visible until the session acknowledges them.
const editor = {
  open: false,
  tool: "brush",
  pending: [],
  saveTask: null,
  saveError: "",
  stroke: null,
  widths: {},
};
const editTools = document.getElementById("editTools");
const editWidth = document.getElementById("editWidth");
const editRectangleStyle = document.getElementById("editRectangleStyle");
const emptyMapDialog = document.getElementById("emptyMapDialog");
const emptyMapForm = document.getElementById("emptyMapForm");
const replaceMapDialog = document.getElementById("replaceMapDialog");
let replaceMapDecision = null;

function updateEditorControls() {
  const editButton = document.getElementById("editBtn");
  editButton.disabled = !mapData;
  editButton.classList.toggle("active", editor.open);
  editButton.setAttribute("aria-expanded", String(editor.open));
  editTools.hidden = !editor.open;
  canvas.classList.toggle("edit-mode", editor.open && !rulerActive);
  canvas.title = editor.open ? "Middle-drag or right-drag to pan" : "";
  for (const button of editTools.querySelectorAll("[data-edit-tool]")) {
    const selected = button.dataset.editTool === editor.tool;
    button.classList.toggle("active", selected);
    button.setAttribute("aria-pressed", String(selected));
  }
  const rectangle = editor.tool === "rectangle";
  document.getElementById("editRectangleField").hidden = !rectangle;
  document.getElementById("editWidthField").hidden = rectangle && editRectangleStyle.value === "filled";
  const busy = mapRequestActive || !!editor.saveTask;
  document.getElementById("editUndo").disabled = busy || !(mapData?.can_undo || editor.pending.length);
  document.getElementById("editRedo").disabled = busy || !!editor.pending.length || !mapData?.can_redo;
  const retryButton = document.getElementById("editRetry");
  retryButton.hidden = !editor.saveError;
  retryButton.disabled = busy;
  const status = document.getElementById("editStatus");
  status.textContent = [
    rulerActive ? "Ruler on" : "",
    editor.saveError ? "Unsaved strokes" : editor.pending.length ? "Updating…" : "",
  ].filter(Boolean).join(" · ");
  status.title = [
    rulerActive ? "Press R to resume drawing" : "",
    editor.saveError,
  ].filter(Boolean).join(" · ");
}

function toggleEditor(force) {
  if (!mapData || mapRequestActive) return;
  finishEditStroke();
  editor.open = force ?? !editor.open;
  updateEditorControls();
  render();
}

function resetEditor(open = false) {
  editor.open = open;
  editor.tool = "brush";
  editor.pending = [];
  editor.saveError = "";
  editor.stroke = null;
  const res = mapData.edit_resolution;
  editor.widths = { brush: res, line: res, rectangle: res, eraser: Math.max(0.25, res) };
  editWidth.min = res;
  editWidth.value = res;
  editRectangleStyle.value = "outline";
  setRulerActive(false);
  updateEditorControls();
}

async function canReplaceMap() {
  if (replaceMapDialog.open) return false;
  if (!(editor.pending.length || editor.stroke || mapData?.can_undo)) return true;
  replaceMapDialog.showModal();
  return new Promise(resolve => {
    replaceMapDecision = resolve;
  });
}

function resolveMapReplacement(accepted) {
  const resolve = replaceMapDecision;
  replaceMapDecision = null;
  replaceMapDialog.close();
  resolve?.(accepted);
}

replaceMapDialog.addEventListener("submit", event => {
  event.preventDefault();
  resolveMapReplacement(event.submitter?.value === "replace");
});
replaceMapDialog.addEventListener("cancel", event => {
  event.preventDefault();
  resolveMapReplacement(false);
});

editTools.addEventListener("click", (event) => {
  const button = event.target.closest("[data-edit-tool]");
  if (!button || mapRequestActive) return;
  editor.tool = button.dataset.editTool;
  editWidth.value = editor.widths[editor.tool];
  updateEditorControls();
});
function saveEditWidth() {
  const fallback = editor.widths[editor.tool];
  editor.widths[editor.tool] = Math.min(5, Math.max(mapData.edit_resolution, editWidth.valueAsNumber || fallback));
  editWidth.value = editor.widths[editor.tool];
}

editWidth.addEventListener("change", saveEditWidth);
editRectangleStyle.addEventListener("change", updateEditorControls);

function editPoint(world) {
  const b = mapData.drawing_bounds;
  if (!b) return [world.x, world.y];
  return [
    Math.max(b.min_x, Math.min(b.max_x, world.x)),
    Math.max(b.min_y, Math.min(b.max_y, world.y)),
  ];
}

function startEditStroke(world) {
  if (mapRequestActive) return;
  const point = editPoint(world);
  if (point[0] !== world.x || point[1] !== world.y) return;
  saveEditWidth();
  const tool = editor.tool;
  editor.stroke = {
    tool,
    width: editor.widths[tool],
    filled: tool === "rectangle" && editRectangleStyle.value === "filled",
    points: tool === "brush" || tool === "eraser" ? [point] : [point, point],
  };
  dragMode = "edit";
  render();
}

function moveEditStroke(world) {
  const stroke = editor.stroke;
  if (!stroke) return;
  const point = editPoint(world);
  if (stroke.tool === "line" || stroke.tool === "rectangle") {
    stroke.points[1] = point;
  } else {
    const previous = stroke.points.at(-1);
    if (Math.hypot(point[0] - previous[0], point[1] - previous[1]) < mapData.edit_resolution / 4) return;
    stroke.points.push(point);
    if (stroke.points.length >= 8192) {
      finishEditStroke();
      showToast("Stroke completed. Press again to continue drawing.");
    }
  }
  render();
}

function finishEditStroke() {
  if (!editor.stroke) return;
  editor.pending.push(editor.stroke);
  editor.stroke = null;
  if (dragMode === "edit") dragMode = null;
  savePendingEdits();
  render();
}

function savePendingEdits() {
  if (!editor.pending.length) return true;
  if (!editor.saveTask) {
    editor.saveError = "";
    editor.saveTask = flushEdits().finally(() => {
      editor.saveTask = null;
      updateEditorControls();
    });
    updateEditorControls();
  }
  return editor.saveTask;
}

async function flushEdits() {
  try {
    while (editor.pending.length) {
      const strokes = editor.pending.slice(0, 500);
      const data = await postJson("/api/map-edits", {
        revision: mapData.edit_revision,
        strokes,
        config: collectBuildConfig(),
      });
      editor.pending.splice(0, strokes.length);
      applyMapPayload(data, false, null, { preserveVehicle: true });
    }
    return true;
  } catch (err) {
    editor.saveError = err.message;
    showToast(err.message);
    return false;
  }
}

function undoEdit() {
  if (mapRequestActive) return;
  if (editor.saveError && editor.pending.length && !editor.stroke) {
    editor.pending.pop();
    if (!editor.pending.length) editor.saveError = "";
    updateEditorControls();
    render();
  } else if (editor.stroke || editor.pending.length || mapData?.can_undo) {
    return changeEditHistory("undo");
  }
}

function redoEdit() {
  if (!editor.stroke && !editor.pending.length && mapData?.can_redo) return changeEditHistory("redo");
}

function changeEditHistory(direction) {
  return runMapRequest(async () => {
    const data = await postJson(`/api/map-edits/${direction}`, {
      revision: mapData.edit_revision,
      config: collectBuildConfig(),
    });
    applyMapPayload(data, false, null, { preserveVehicle: true });
  });
}

function drawMapEdits() {
  const b = mapData.drawing_bounds;
  ctx.save();
  if (b) {
    const tl = worldToScreen(b.min_x, b.max_y);
    const br = worldToScreen(b.max_x, b.min_y);
    ctx.strokeStyle = "#64748b";
    ctx.lineWidth = 1;
    ctx.setLineDash([5, 4]);
    ctx.strokeRect(tl.x, tl.y, br.x - tl.x, br.y - tl.y);
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.rect(tl.x, tl.y, br.x - tl.x, br.y - tl.y);
    ctx.clip();
  }
  for (const stroke of editor.pending) drawEditStroke(stroke);
  if (editor.stroke) drawEditStroke(editor.stroke);
  ctx.restore();
  const stroke = editor.stroke;
  if (stroke?.tool === "line" || stroke?.tool === "rectangle") {
    const [[x1, y1], [x2, y2]] = stroke.points;
    const dx = Math.abs(x2 - x1), dy = Math.abs(y2 - y1);
    const label = stroke.tool === "line"
      ? `${Math.hypot(dx, dy).toFixed(2)} m` : `${dx.toFixed(2)} × ${dy.toFixed(2)} m`;
    const tip = worldToScreen(x2, y2);
    ctx.save();
    ctx.font = "600 12px monospace";
    const width = ctx.measureText(label).width + 12;
    const x = Math.max(4, Math.min(canvas.clientWidth - width - 4, tip.x + 10));
    const y = Math.max(4, tip.y - 28);
    ctx.fillStyle = "#172234";
    ctx.fillRect(x, y, width, 20);
    ctx.textAlign = "left";
    ctx.fillStyle = "#a5f3fc";
    ctx.fillText(label, x + 6, y + 14);
    ctx.restore();
  }
}

function drawEditStroke(stroke) {
  const points = stroke.points.map(([x, y]) => worldToScreen(x, y));
  const start = points[0];
  const end = points.at(-1);
  const width = Math.max(2, stroke.width * cam.scale);
  const erasing = stroke.tool === "eraser";
  ctx.strokeStyle = ctx.fillStyle = erasing ? "rgba(251, 113, 133, 0.65)" : "rgba(34, 211, 238, 0.75)";
  ctx.lineWidth = width;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  if (stroke.tool === "rectangle" && (start.x !== end.x || start.y !== end.y)) {
    const w = Math.max(mapData.edit_resolution * cam.scale, Math.abs(end.x - start.x));
    const h = Math.max(mapData.edit_resolution * cam.scale, Math.abs(end.y - start.y));
    const x = Math.min(start.x, end.x), y = Math.min(start.y, end.y);
    if (stroke.filled) ctx.fillRect(x, y, w, h);
    else ctx.strokeRect(x, y, w, h);
  } else if (points.length === 1 || (points.length === 2 && start.x === end.x && start.y === end.y)) {
    ctx.beginPath();
    ctx.arc(start.x, start.y, width / 2, 0, 2 * Math.PI);
    ctx.fill();
  } else {
    ctx.beginPath();
    ctx.moveTo(start.x, start.y);
    for (const point of points.slice(1)) ctx.lineTo(point.x, point.y);
    ctx.stroke();
  }
}

async function openEmptyMap() {
  if (mapRequestActive || emptyMapDialog.open || !(await canReplaceMap())) return;
  document.getElementById("emptyMapError").hidden = true;
  emptyMapDialog.showModal();
}

emptyMapDialog.addEventListener("cancel", (event) => {
  if (mapRequestActive) event.preventDefault();
});
emptyMapForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const fields = emptyMapForm.elements;
  const error = document.getElementById("emptyMapError");
  error.hidden = true;
  const request = {
    width: fields.width.valueAsNumber,
    height: fields.height.valueAsNumber,
    resolution: fields.resolution.valueAsNumber,
    config: collectBuildConfig(),
  };
  runMapRequest(async () => {
    const buttons = emptyMapForm.querySelectorAll("button");
    buttons.forEach((button) => { button.disabled = true; });
    try {
      const data = await postJson("/api/empty-map", request);
      document.getElementById("gridRes").value = data.config.grid_resolution;
      emptyMapDialog.close();
      applyMapPayload(data, true, "edit");
    } catch (err) {
      error.textContent = err.message;
      error.hidden = false;
    } finally {
      buttons.forEach((button) => { button.disabled = false; });
    }
  });
});
