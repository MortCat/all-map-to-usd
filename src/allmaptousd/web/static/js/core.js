// Shared state, toast, sidebar/accordions/steppers, camera & scale bar.
// Classic script: shares the global scope with the other js/*.js files (see index.html).

let mapData = null;

// Camera: world (x, y) -> screen (sx, sy), Y-up
const cam = { cx: 0, cy: 0, scale: 40 };
const agv = { x: 0, y: 0, length: 1.2, width: 0.8, height: 0.6, yawDeg: 0, margin: 0.1 };

let rulerActive = false, rulerP1 = null, rulerP2 = null;
let dragMode = null; // 'pan' | 'agv'
let lastMouse = { x: 0, y: 0 };
let agvDragOffset = { x: 0, y: 0 };
let lastCollision = { hit: false, minDist: Infinity, reason: "" };

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const sidebarShell = document.getElementById("sidebarShell");

function showToast(msg) {
  const el = document.getElementById("toast");
  el.textContent = msg;
  el.classList.add("show");
  clearTimeout(el._t);
  el._t = setTimeout(() => el.classList.remove("show"), 2600);
}

// --- Collapsible & Resizable Sidebar + Accordions ---
let _sidebarAnimRaf = null;
function _syncCanvasDuringTransition(durationMs = 320) {
  if (_sidebarAnimRaf) cancelAnimationFrame(_sidebarAnimRaf);
  const start = performance.now();
  function tick(now) {
    resizeCanvas();
    if (now - start < durationMs) {
      _sidebarAnimRaf = requestAnimationFrame(tick);
    } else {
      _sidebarAnimRaf = null;
    }
  }
  _sidebarAnimRaf = requestAnimationFrame(tick);
}

function toggleSidebar() {
  sidebarShell.classList.toggle("collapsed");
  _syncCanvasDuringTransition(340);
}

function toggleAcc(id) {
  document.getElementById(id)?.classList.toggle("open");
}

function stepInput(id, delta, decimals = 2, cb = null) {
  const el = document.getElementById(id);
  if (!el) return;
  const min = el.min !== "" ? parseFloat(el.min) : -Infinity;
  const max = el.max !== "" ? parseFloat(el.max) : Infinity;
  const cur = parseFloat(el.value) || 0;
  const next = Math.max(min, Math.min(max, cur + delta));
  el.value = decimals > 0 ? next.toFixed(decimals) : Math.round(next);
  if (cb) cb();
}

(function setupSidebarResize() {
  const handle = document.getElementById("resizeHandle");
  let resizing = false;
  handle.addEventListener("mousedown", (e) => {
    e.preventDefault();
    resizing = true;
    sidebarShell.classList.add("resizing");
    document.body.style.cursor = "col-resize";
  });
  window.addEventListener("mousemove", (e) => {
    if (!resizing) return;
    const w = Math.max(280, Math.min(480, e.clientX));
    document.documentElement.style.setProperty("--sidebar-w", `${w}px`);
    resizeCanvas();
  });
  window.addEventListener("mouseup", () => {
    if (!resizing) return;
    resizing = false;
    sidebarShell.classList.remove("resizing");
    document.body.style.cursor = "";
  });
})();

function resizeCanvas() {
  const rect = canvas.parentElement.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  render();
}
window.addEventListener("resize", resizeCanvas);

function worldToScreen(wx, wy) {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  return { x: w * 0.5 + (wx - cam.cx) * cam.scale, y: h * 0.5 - (wy - cam.cy) * cam.scale };
}
function screenToWorld(sx, sy) {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  return { x: cam.cx + (sx - w * 0.5) / cam.scale, y: cam.cy - (sy - h * 0.5) / cam.scale };
}

function zoomBy(factor) {
  cam.scale = Math.max(3, Math.min(600, cam.scale * factor));
  render();
}

function fitToView() {
  if (!mapData) return;
  const b = mapData.bounds;
  const bw = Math.max(1, b.max_x - b.min_x);
  const bh = Math.max(1, b.max_y - b.min_y);
  cam.cx = 0.5 * (b.min_x + b.max_x);
  cam.cy = 0.5 * (b.min_y + b.max_y);
  const pad = 72;
  const sx = (canvas.clientWidth - pad) / bw;
  const sy = (canvas.clientHeight - pad) / bh;
  cam.scale = Math.max(4, Math.min(300, Math.min(sx, sy)));
  render();
}

function updateScaleBar() {
  const targetPx = 90;
  const rawMeters = targetPx / cam.scale;
  const niceSteps = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50];
  let best = niceSteps[0];
  for (const s of niceSteps) {
    if (s <= rawMeters * 1.35) best = s;
  }
  const px = Math.max(24, Math.round(best * cam.scale));
  document.getElementById("scaleLine").style.width = `${px}px`;
  document.getElementById("scaleLabel").textContent = best >= 1 ? `${best} m` : `${Math.round(best * 100)} cm`;
}
