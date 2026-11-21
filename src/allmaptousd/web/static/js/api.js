// Backend calls: load/upload/rebuild, USD export, Isaac Sim push.
// Classic script: shares the global scope with the other js/*.js files (see index.html).

// --- Server API ---
function applyMapPayload(payload, shouldFit = true) {
  mapData = payload;
  document.getElementById("emptyOverlay").classList.add("hidden");
  const m = payload.metadata, b = payload.bounds;
  const w = (b.max_x - b.min_x).toFixed(2), h = (b.max_y - b.min_y).toFixed(2);
  const polyCount = (payload.obstacle_polygons || []).length;
  const boxCount = (payload.obstacle_boxes || []).length;
  const obsLabel = polyCount > 0 ? `${polyCount} polygons` : `${boxCount} boxes`;
  const navLabel = (payload.stations || []).length || (payload.paths || []).length
    ? ` · ${payload.stations.length} stations · ${payload.paths.length} paths` : "";
  const coarse = m.extra && m.extra.resolution_was_coarsened;
  const coarseLabel = coarse ? ` · ⚠ grid coarsened to ${m.extra.effective_grid_resolution} m` : "";
  document.getElementById("metaSummary").textContent =
    `${m.name} · ${w}×${h} m · ${obsLabel} · ${payload.forbidden_zones.length} zones${navLabel}${coarseLabel}`;
  if (coarse) {
    showToast(`Map too large: grid resolution coarsened to ${m.extra.effective_grid_resolution} m`);
  }
  renderLayerList(payload.layers);
  const frontier = (m.extra && m.extra.unknown_frontier_cells) || 0;
  document.getElementById("sealUnknownField").style.display = frontier > 0 ? "flex" : "none";
  document.getElementById("frontierCount").textContent = `${frontier} cells`;
  if (payload.config && "seal_unknown" in payload.config) {
    document.getElementById("sealUnknown").checked = !!payload.config.seal_unknown;
  }

  if (payload.config?.vehicle_proxy) {
    const vp = payload.config.vehicle_proxy;
    agv.length = vp.length; agv.width = vp.width; agv.height = vp.height;
    agv.x = vp.pos_x; agv.y = vp.pos_y; agv.yawDeg = vp.yaw_deg;
    document.getElementById("includeAgv").checked = vp.enabled !== false;
    document.getElementById("agvL").value = vp.length.toFixed(2);
    document.getElementById("agvW").value = vp.width.toFixed(2);
    document.getElementById("agvH").value = vp.height.toFixed(2);
    document.getElementById("agvYaw").value = vp.yaw_deg;
    document.getElementById("yawLabel").textContent = `${Math.round(vp.yaw_deg)}°`;
  }

  ["rebuildBtn", "exportUsdBtn", "exportUsdaBtn", "pushIsaacBtn"].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.disabled = false;
  });

  if (shouldFit) fitToView();
  else render();
}

function collectBuildConfig() {
  return {
    reconstruction_mode: document.getElementById("reconMode").value || "contour_polygon",
    grid_resolution: parseFloat(document.getElementById("gridRes").value) || 0.05,
    wall_height: parseFloat(document.getElementById("wallHeight").value) || 2.5,
    forbidden_height: parseFloat(document.getElementById("forbiddenHeight").value) || 1.2,
    morph_close_cells: parseInt(document.getElementById("morphClose").value, 10) || 0,
    dilate_cells: parseInt(document.getElementById("dilateCells").value, 10) || 0,
    min_cluster_cells: parseInt(document.getElementById("minCluster").value, 10) || 3,
    fill_solid_obstacles: document.getElementById("fillSolid").checked,
    solid_fill_max_area_m2: parseFloat(document.getElementById("solidMaxArea").value) || 30.0,
    contour_simplify_tol: parseFloat(document.getElementById("contourTol").value) || 0.04,
    include_forbidden_as_collision: document.getElementById("forbiddenCollision").checked,
    include_ground: true,
    ground_margin: 2.0,
    include_navigation: document.getElementById("includeNav").checked,
    seal_unknown: document.getElementById("sealUnknown").checked,
    ...collectLayerConfig(),
    origin_x: parseFloat(document.getElementById("originX").value) || 0,
    origin_y: parseFloat(document.getElementById("originY").value) || 0,
    origin_yaw_deg: parseFloat(document.getElementById("originYaw").value) || 0,
    vehicle_proxy: {
      enabled: document.getElementById("includeAgv").checked,
      length: agv.length, width: agv.width, height: agv.height,
      pos_x: agv.x, pos_y: agv.y, yaw_deg: agv.yawDeg
    }
  };
}

async function loadSample(sampleId) {
  const res = await fetch(`/api/load-sample/${sampleId}`, { method: "POST" });
  if (!res.ok) { showToast("Failed to load sample map"); return; }
  const data = await res.json();
  applyMapPayload(data, true);
  showToast(`Loaded sample: ${data.metadata.name}`);
}

function onMapFormatChange() {
  const fmt = document.getElementById("mapFormat").value;
  document.getElementById("dxfUnitsField").style.display = fmt === "dxf_cad" ? "flex" : "none";
  document.getElementById("imageResField").style.display = fmt === "raster_image" ? "flex" : "none";
}

async function uploadFiles(fileList) {
  if (!fileList || !fileList.length) return;
  const fd = new FormData();
  for (const f of fileList) fd.append("files", f);
  fd.append("format_hint", document.getElementById("mapFormat").value || "auto");
  fd.append("image_resolution", document.getElementById("imageRes").value || "0.05");
  fd.append("dxf_units", document.getElementById("dxfUnits").value || "auto");
  const res = await fetch("/api/upload", { method: "POST", body: fd });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Parse failed" }));
    showToast(`Error: ${err.detail}`);
    return;
  }
  const data = await res.json();
  applyMapPayload(data, true);
  showToast(`Loaded: ${data.metadata.name}`);
}

async function rebuildMap() {
  const res = await fetch("/api/rebuild", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(collectBuildConfig())
  });
  if (!res.ok) { showToast("Rebuild failed"); return; }
  const data = await res.json();
  applyMapPayload(data, false);
  const pCount = (data.obstacle_polygons || []).length;
  const bCount = (data.obstacle_boxes || []).length;
  showToast(pCount > 0 ? `Rebuilt (${pCount} contour polygons)` : `Rebuilt (${bCount} AABB boxes)`);
}

async function triggerDownload(url, bodyObj, fallbackName) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(bodyObj)
  });
  if (!res.ok) { showToast("Export failed"); return; }
  const blob = await res.blob();
  const cd = res.headers.get("Content-Disposition") || "";
  const m = cd.match(/filename\*=UTF-8''([^;]+)/) || cd.match(/filename="([^"]+)"/);
  const filename = m ? decodeURIComponent(m[1]) : fallbackName;
  const savedPath = decodeURIComponent(res.headers.get("X-Saved-Path") || "");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(a.href);
  const expEl = document.getElementById("exportStatus");
  if (expEl && savedPath) {
    expEl.style.color = "#4ade80";
    expEl.textContent = `Saved: ${savedPath}`;
  }
  showToast(savedPath ? `Exported to ${savedPath}` : `Exported: ${filename}`);
}

function exportUsd(ext) {
  triggerDownload(`/api/export/usd?ext=${ext}`, collectBuildConfig(), `map_layout.${ext}`);
}
// Full IsaacSimConfig as loaded from /api/isaac/config (see main.js). The panel only edits a few
// fields; the rest (ssh_user, remote_save_dir, transfer_mode, ...) must round-trip unchanged,
// because every Test / Push saves the config back to isaac_config.json.
let isaacConfigBase = {};

function collectIsaacConfig() {
  return {
    ...isaacConfigBase,
    host: document.getElementById("isaacHost").value.trim() || "127.0.0.1",
    port: parseInt(document.getElementById("isaacPort").value, 10) || 8226,
    target_prim_path: document.getElementById("isaacPrim").value.trim() || "/World/AGV_Map",
    push_mode: document.getElementById("isaacPushMode").value || "reference"
  };
}

async function testIsaac() {
  const cfg = collectIsaacConfig();
  const statusEl = document.getElementById("isaacStatus");
  statusEl.style.color = "var(--text-dim)";
  statusEl.textContent = `Testing ${cfg.host}:${cfg.port} ...`;
  const res = await fetch("/api/isaac/test", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(cfg)
  });
  const data = await res.json();
  statusEl.style.color = data.reachable ? "#4ade80" : "#f87171";
  statusEl.textContent = data.message;
}

async function pushToIsaac() {
  const btn = document.getElementById("pushIsaacBtn");
  const statusEl = document.getElementById("isaacStatus");
  const isaacCfg = collectIsaacConfig();
  btn.disabled = true;
  btn.textContent = "Pushing...";
  statusEl.style.color = "var(--accent)";
  statusEl.textContent = `Building USD and pushing to ${isaacCfg.host}:${isaacCfg.port} ...`;
  try {
    const res = await fetch("/api/isaac/push", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        build_config: collectBuildConfig(),
        isaac_config: isaacCfg
      })
    });
    const data = await res.json();
    if (!res.ok || !data.ok) {
      statusEl.style.color = "#f87171";
      statusEl.textContent = data.detail || data.message || "Push failed";
      showToast("Push to Isaac Sim failed");
    } else {
      statusEl.style.color = "#4ade80";
      statusEl.textContent = data.message;
      showToast(`Mounted in Isaac Sim (${data.mounted_prim})`);
    }
  } catch (err) {
    statusEl.style.color = "#f87171";
    statusEl.textContent = `Request error: ${err}`;
  } finally {
    btn.disabled = false;
    btn.textContent = "Push to Isaac Sim";
  }
}
