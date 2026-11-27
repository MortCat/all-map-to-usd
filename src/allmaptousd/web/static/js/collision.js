// AGV footprint vs obstacle/zone collision and clearance.
// Classic script: shares the global scope with the other js/*.js files (see index.html).

// --- 2D Collision Engine ---
function getAgvCorners(inflate = 0) {
  const rad = (agv.yawDeg * Math.PI) / 180;
  const c = Math.cos(rad), s = Math.sin(rad);
  const hl = agv.length * 0.5 + inflate, hw = agv.width * 0.5 + inflate;
  return [[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]].map(
    ([lx, ly]) => [agv.x + lx * c - ly * s, agv.y + lx * s + ly * c]
  );
}

function pointInPolygon(px, py, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const xi = poly[i][0], yi = poly[i][1];
    const xj = poly[j][0], yj = poly[j][1];
    if (((yi > py) !== (yj > py)) && (px < ((xj - xi) * (py - yi)) / (yj - yi + 1e-12) + xi)) {
      inside = !inside;
    }
  }
  return inside;
}

function segIntersect(a, b, c, d) {
  const ccw = (p1, p2, p3) => (p3[1] - p1[1]) * (p2[0] - p1[0]) > (p2[1] - p1[1]) * (p3[0] - p1[0]);
  return (ccw(a, c, d) !== ccw(b, c, d)) && (ccw(a, b, c) !== ccw(a, b, d));
}

function distPointToSeg(px, py, ax, ay, bx, by) {
  const dx = bx - ax, dy = by - ay;
  const len2 = dx * dx + dy * dy;
  let t = len2 > 1e-12 ? ((px - ax) * dx + (py - ay) * dy) / len2 : 0;
  t = Math.max(0, Math.min(1, t));
  return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
}

function polygonsOverlapAndDist(polyA, polyB) {
  for (let i = 0; i < polyA.length; i++) {
    if (pointInPolygon(polyA[i][0], polyA[i][1], polyB)) return { hit: true, dist: 0 };
  }
  for (let i = 0; i < polyB.length; i++) {
    if (pointInPolygon(polyB[i][0], polyB[i][1], polyA)) return { hit: true, dist: 0 };
  }
  let minDist = Infinity;
  for (let i = 0; i < polyA.length; i++) {
    const a1 = polyA[i], a2 = polyA[(i + 1) % polyA.length];
    for (let j = 0; j < polyB.length; j++) {
      const b1 = polyB[j], b2 = polyB[(j + 1) % polyB.length];
      if (segIntersect(a1, a2, b1, b2)) return { hit: true, dist: 0 };
      minDist = Math.min(
        minDist,
        distPointToSeg(a1[0], a1[1], b1[0], b1[1], b2[0], b2[1]),
        distPointToSeg(b1[0], b1[1], a1[0], a1[1], a2[0], a2[1])
      );
    }
  }
  return { hit: false, dist: minDist };
}

function polygonBounds(poly) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const [x, y] of poly) {
    if (x < x0) x0 = x;
    if (y < y0) y0 = y;
    if (x > x1) x1 = x;
    if (y > y1) y1 = y;
  }
  return { x0, y0, x1, y1 };
}

// Obstacle outlines with their bounding boxes, built once per map payload rather than on every mouse move.
let obstacleCache = { source: null, items: [] };
function obstacleShapes() {
  if (obstacleCache.source !== mapData) {
    const items = [];
    for (const poly of (mapData.obstacle_polygons || [])) {
      if (poly && poly.length >= 3) items.push({ poly, reason: "Obstacle", ...polygonBounds(poly) });
    }
    for (const [x0, y0, x1, y1] of (mapData.obstacle_boxes || [])) {
      items.push({ poly: [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], reason: "Wall", x0, y0, x1, y1 });
    }
    obstacleCache = { source: mapData, items };
  }
  return obstacleCache.items;
}

function evaluateCollision() {
  if (!mapData) return;
  const agvPoly = getAgvCorners();
  const agvBox = polygonBounds(agvPoly);

  let hit = false, minDist = Infinity, reason = "";
  for (const item of obstacleShapes()) {
    const dx = Math.max(0, item.x0 - agvBox.x1, agvBox.x0 - item.x1);
    const dy = Math.max(0, item.y0 - agvBox.y1, agvBox.y0 - item.y1);
    if (Math.hypot(dx, dy) > minDist) continue;
    const res = polygonsOverlapAndDist(agvPoly, item.poly);
    if (res.hit) { hit = true; minDist = 0; reason = item.reason; break; }
    if (res.dist < minDist) minDist = res.dist;
  }

  // Placed 3D models always count as obstacles.
  for (const p of hit ? [] : placements) {
    for (const poly of placementPolygons(p)) {
      const box = polygonBounds(poly);
      const dx = Math.max(0, box.x0 - agvBox.x1, agvBox.x0 - box.x1);
      const dy = Math.max(0, box.y0 - agvBox.y1, agvBox.y0 - box.y1);
      if (Math.hypot(dx, dy) > minDist) continue;
      const res = polygonsOverlapAndDist(agvPoly, poly);
      if (res.hit) { hit = true; minDist = 0; reason = `Model (${p.name})`; break; }
      if (res.dist < minDist) minDist = res.dist;
    }
    if (hit) break;
  }

  const checkForbidden = document.getElementById("forbiddenCollision")?.checked ?? true;
  if (!hit && checkForbidden) {
    for (const z of mapData.forbidden_zones) {
      if (!z.vertices || z.vertices.length < 3) continue;
      const res = polygonsOverlapAndDist(agvPoly, z.vertices);
      if (res.hit) { hit = true; minDist = 0; reason = `Zone (${z.name})`; break; }
      if (res.dist < minDist) minDist = res.dist;
    }
  }

  lastCollision = { hit, minDist, reason };
  let state, text, clearance;
  if (hit) {
    [state, text, clearance] = ["hit", `COLLISION · ${reason}`, "0.00 m"];
  } else if (minDist < agv.margin) {
    [state, text, clearance] = ["warn", `TOO CLOSE · < ${agv.margin.toFixed(2)} m`, `Min ${minDist.toFixed(2)} m`];
  } else {
    [state, text, clearance] = ["ok", "CLEAR", Number.isFinite(minDist) ? `Min ${minDist.toFixed(2)} m` : "--"];
  }
  document.getElementById("collisionStatus").className = `status-banner ${state}`;
  document.getElementById("collisionLabel").textContent = text;
  document.getElementById("clearanceText").textContent = clearance;
  // Mirror the result in the always-visible bottom HUD (the banner lives in a collapsible panel).
  const hud = document.getElementById("hudCollision");
  hud.hidden = false;
  hud.className = `hud-status ${state}`;
  document.getElementById("hudCollisionText").textContent =
    state === "hit" ? text : `${text.split(" · ")[0]} · ${clearance.replace("Min ", "")}`;
  document.getElementById("agvCoord").textContent = `(${agv.x.toFixed(2)}, ${agv.y.toFixed(2)}, ${Math.round(agv.yawDeg)}°)`;
}
