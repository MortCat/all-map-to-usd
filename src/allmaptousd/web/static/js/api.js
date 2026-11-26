// Backend calls: load/upload/rebuild, USD export, Isaac Sim sync.
// Classic script: shares the global scope with the other js/*.js files (see index.html).

// --- Server API ---
let mapRequestActive = false;

async function runMapRequest(action) {
  if (mapRequestActive) return;
  finishEditStroke();
  mapRequestActive = true;
  updateEditorControls();
  try {
    if (await savePendingEdits()) await action();
  } catch (err) {
    showToast(err.message);
  } finally {
    mapRequestActive = false;
    updateEditorControls();
  }
}

function mapRevisionHeader() {
  return mapData ? { "X-Map-Revision": mapData.edit_revision } : {};
}

async function postMapRequest(url, body, fallback = "Map request failed.") {
  const isJson = body !== undefined && !(body instanceof FormData);
  const res = await fetch(url, {
    method: "POST",
    headers: isJson ? { "Content-Type": "application/json", ...mapRevisionHeader() } : {},
    body: isJson ? JSON.stringify(body) : body,
  });
  if (!res.ok) {
    const error = await res.json().catch(() => null);
    throw new Error(typeof error?.detail === "string" ? error.detail : fallback);
  }
  return res;
}

async function postJson(url, body, fallback) {
  return (await postMapRequest(url, body, fallback)).json();
}

function applyMapPayload(payload, shouldFit = true, newSource = null, { preserveVehicle = false } = {}) {
  mapData = payload;
  if (newSource) resetEditor(newSource === "edit");
  document.getElementById("emptyOverlay").classList.add("hidden");
  const m = payload.metadata, b = payload.bounds;
  const w = (b.max_x - b.min_x).toFixed(2), h = (b.max_y - b.min_y).toFixed(2);
  const polyCount = (payload.obstacle_polygons || []).length;
  const boxCount = (payload.obstacle_boxes || []).length;
  const obsLabel = [
    polyCount ? `${polyCount} polygons` : "",
    boxCount ? `${boxCount} boxes` : "",
  ].filter(Boolean).join(" + ") || "0 obstacles";
  const navLabel = (payload.stations || []).length || (payload.paths || []).length
    ? ` · ${payload.stations.length} stations · ${payload.paths.length} paths` : "";
  const coarse = m.extra?.resolution_was_coarsened;
  const coarseLabel = coarse ? ` · ⚠ grid coarsened to ${m.extra.effective_grid_resolution} m` : "";
  document.getElementById("metaSummary").textContent =
    `${m.name} · ${w}×${h} m · ${obsLabel} · ${payload.forbidden_zones.length} zones${navLabel}${coarseLabel}`;
  if (coarse) {
    showToast(`Map too large: grid resolution coarsened to ${m.extra.effective_grid_resolution} m`);
  }
  renderLayerList(payload.layers);
  const frontier = m.extra?.unknown_frontier_cells || 0;
  document.getElementById("sealUnknownField").hidden = !frontier;
  document.getElementById("frontierCount").textContent = `${frontier} cells`;
  if (payload.config && "seal_unknown" in payload.config) {
    document.getElementById("sealUnknown").checked = !!payload.config.seal_unknown;
  }

  if (!preserveVehicle && payload.config?.vehicle_proxy) {
    const vp = payload.config.vehicle_proxy;
    agv.length = vp.length; agv.width = vp.width; agv.height = vp.height;
    agv.x = vp.pos_x; agv.y = vp.pos_y; agv.yawDeg = vp.yaw_deg;
    document.getElementById("includeAgv").checked = vp.enabled !== false;
    document.getElementById("agvL").value = vp.length.toFixed(2);
    document.getElementById("agvW").value = vp.width.toFixed(2);
    document.getElementById("agvH").value = vp.height.toFixed(2);
    document.getElementById("agvYaw").value = vp.yaw_deg;
    document.getElementById("yawLabel").textContent = `${Math.round(vp.yaw_deg)}°`;
    updateIsaacPullRange();
  }

  ["rebuildBtn", "exportUsdBtn", "exportUsdaBtn", "pushIsaacBtn"].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.disabled = false;
  });

  updateEditorControls();
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
  if (mapRequestActive || !(await canReplaceMap())) return;
  return runMapRequest(async () => {
    const data = await postJson(`/api/load-sample/${sampleId}`, undefined, "Failed to load sample map.");
    applyMapPayload(data, true, "loaded");
    showToast(`Loaded sample: ${data.metadata.name}`);
  });
}

function onMapFormatChange() {
  const fmt = document.getElementById("mapFormat").value;
  document.getElementById("dxfUnitsField").hidden = fmt !== "dxf_cad";
  document.getElementById("imageResField").hidden = fmt !== "raster_image";
}

async function uploadFiles(fileList) {
  const files = Array.from(fileList || []);
  if (!files.length || mapRequestActive || !(await canReplaceMap())) return;
  const fd = new FormData();
  for (const f of files) fd.append("files", f);
  fd.append("format_hint", document.getElementById("mapFormat").value || "auto");
  fd.append("image_resolution", document.getElementById("imageRes").value || "0.05");
  fd.append("dxf_units", document.getElementById("dxfUnits").value || "auto");
  return runMapRequest(async () => {
    const data = await postJson("/api/upload", fd, "Map import failed.");
    applyMapPayload(data, true, "loaded");
    showToast(`Loaded: ${data.metadata.name}`);
  });
}

function rebuildMap() {
  return runMapRequest(async () => {
    applyMapPayload(await postJson("/api/rebuild", collectBuildConfig()), false);
    showToast("Map rebuilt");
  });
}

async function downloadUsd(ext) {
  const res = await postMapRequest(`/api/export/usd?ext=${ext}`, collectBuildConfig(), "Export failed.");
  const blob = await res.blob();
  const cd = res.headers.get("Content-Disposition") || "";
  const m = cd.match(/filename\*=UTF-8''([^;]+)/) || cd.match(/filename="([^"]+)"/);
  const filename = m ? decodeURIComponent(m[1]) : `map_layout.${ext}`;
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
    expEl.dataset.state = "success";
    expEl.textContent = `Saved: ${savedPath}`;
  }
  showToast(savedPath ? `Exported to ${savedPath}` : `Exported: ${filename}`);
}

function exportUsd(ext) {
  return runMapRequest(() => downloadUsd(ext));
}
// Full IsaacSimConfig as loaded from /api/isaac/config (see main.js). The panel only edits a few
// fields; the rest (ssh_user, remote_save_dir, transfer_mode, ...) must round-trip unchanged,
// because sync operations save the config back to isaac_config.json.
let isaacConfigBase = {};

function readIsaacPullSettings() {
  return {
    root_prim_path: document.getElementById("isaacPullRoot").value.trim() || "/",
    exclude_prim_path: document.getElementById("isaacPullExclude").value.trim(),
    floor_elevation_m: document.getElementById("isaacPullFloor").valueAsNumber,
    floor_clearance_m: document.getElementById("isaacPullClearance").valueAsNumber,
    geometry: document.getElementById("isaacPullGeometry").value,
  };
}

function collectIsaacConfig() {
  const pull = readIsaacPullSettings();
  return {
    ...isaacConfigBase,
    host: document.getElementById("isaacHost").value.trim() || "127.0.0.1",
    port: parseInt(document.getElementById("isaacPort").value, 10) || 8226,
    target_prim_path: document.getElementById("isaacPrim").value.trim() || "/World/AGV_Map",
    push_mode: document.getElementById("isaacPushMode").value || "reference",
    pull: isaacPullRange(pull) ? pull : isaacConfigBase.pull,
  };
}

function isaacPullRange(pull) {
  const floor = pull.floor_elevation_m;
  const clearance = pull.floor_clearance_m;
  if (!Number.isFinite(floor) || Math.abs(floor) > 1e6 ||
      !Number.isFinite(clearance) || clearance < 0 || clearance > 20 || clearance >= agv.height) return null;
  return [floor + clearance, floor + agv.height];
}

const INVALID_PULL_RANGE = "Enter a valid floor elevation and a clearance below the AGV height.";

function updateIsaacPullRange() {
  const range = isaacPullRange(readIsaacPullSettings());
  document.getElementById("isaacPullRange").textContent = range
    ? `Pull range: ${range[0].toFixed(3)}–${range[1].toFixed(3)} m (AGV height ${agv.height.toFixed(2)} m)`
    : INVALID_PULL_RANGE;
}

async function runIsaacAction(buttonId, busyLabel, message, action, failureNotice) {
  const button = document.getElementById(buttonId);
  if (button.disabled) return;
  const status = document.getElementById("isaacStatus");
  const label = button.textContent;
  button.disabled = true;
  button.textContent = busyLabel;
  status.dataset.state = "pending";
  status.textContent = message;
  try {
    status.textContent = await action();
    status.dataset.state = "success";
  } catch (err) {
    status.textContent = err.message;
    status.dataset.state = "error";
    if (failureNotice) showToast(failureNotice);
  } finally {
    button.disabled = false;
    button.textContent = label;
  }
}

function isaacMapRequest(direction, cfg) {
  return postJson(`/api/isaac/${direction}`, {
    build_config: collectBuildConfig(),
    isaac_config: cfg,
  }, `Isaac Sim ${direction} failed.`);
}

async function pullFromIsaac() {
  if (mapRequestActive) return;
  if (!isaacPullRange(readIsaacPullSettings())) {
    showToast(INVALID_PULL_RANGE);
    return;
  }
  if (!(await canReplaceMap())) return;
  const cfg = collectIsaacConfig();
  return runMapRequest(() => runIsaacAction(
    "pullIsaacBtn", "Pulling…", `Reading ${cfg.pull.root_prim_path} from ${cfg.host}:${cfg.port}…`,
    async () => {
      const data = await isaacMapRequest("pull", cfg);
      isaacConfigBase = cfg;
      const build = data.config;
      document.getElementById("gridRes").value = build.grid_resolution;
      document.getElementById("reconMode").value = build.reconstruction_mode;
      document.getElementById("morphClose").value = build.morph_close_cells;
      document.getElementById("dilateCells").value = build.dilate_cells;
      document.getElementById("minCluster").value = build.min_cluster_cells;
      document.getElementById("fillSolid").checked = build.fill_solid_obstacles;
      document.getElementById("originX").value = build.origin_x;
      document.getElementById("originY").value = build.origin_y;
      document.getElementById("originYaw").value = build.origin_yaw_deg;
      applyMapPayload(data, true, "loaded");
      const meta = data.metadata.extra;
      const skipped = meta.isaac_unsupported_prims ? ` · ${meta.isaac_unsupported_prims} unsupported geometry skipped` : "";
      const coarsened = data.metadata.resolution > meta.isaac_requested_resolution_m
        ? ` · grid coarsened to ${data.metadata.resolution} m` : "";
      showToast("Isaac Sim map loaded");
      return `Pulled ${meta.isaac_projected_prims} obstacle prims from ${data.metadata.name}${skipped}${coarsened}`;
    }, "Pull from Isaac Sim failed",
  ));
}

function testIsaac() {
  const cfg = collectIsaacConfig();
  return runIsaacAction("testIsaacBtn", "Testing…", `Testing ${cfg.host}:${cfg.port}…`, async () => {
    const data = await postJson("/api/isaac/test", cfg, "Connection test failed.");
    if (!data.reachable) throw new Error(data.message);
    return data.message;
  });
}

function pushToIsaac() {
  const cfg = collectIsaacConfig();
  return runMapRequest(() => runIsaacAction(
    "pushIsaacBtn", "Pushing…", `Building USD and pushing to ${cfg.host}:${cfg.port}…`,
    async () => {
      const data = await isaacMapRequest("push", cfg);
      if (!data.ok) throw new Error(data.message || "Push failed.");
      showToast(`Mounted in Isaac Sim (${data.mounted_prim})`);
      return data.message;
    }, "Push to Isaac Sim failed",
  ));
}
