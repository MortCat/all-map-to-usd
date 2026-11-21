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

function getBoxCornersAndAabb(box) {
  const [x0, y0, x1, y1] = box;
  return { poly: [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], x0, y0, x1, y1 };
}

function evaluateCollision() {
  if (!mapData) return;
  const agvPoly = getAgvCorners();
  let agvMinX = Infinity, agvMinY = Infinity, agvMaxX = -Infinity, agvMaxY = -Infinity;
  for (const [x, y] of agvPoly) {
    if (x < agvMinX) agvMinX = x;
    if (y < agvMinY) agvMinY = y;
    if (x > agvMaxX) agvMaxX = x;
    if (y > agvMaxY) agvMaxY = y;
  }

  let hit = false, minDist = Infinity, reason = "";
  for (const poly of (mapData.obstacle_polygons || [])) {
    if (!poly || poly.length < 3) continue;
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const [x, y] of poly) {
      if (x < x0) x0 = x;
      if (y < y0) y0 = y;
      if (x > x1) x1 = x;
      if (y > y1) y1 = y;
    }
    const dx = Math.max(0, x0 - agvMaxX, agvMinX - x1);
    const dy = Math.max(0, y0 - agvMaxY, agvMinY - y1);
    if (Math.hypot(dx, dy) > minDist && minDist < Infinity) continue;
    const res = polygonsOverlapAndDist(agvPoly, poly);
    if (res.hit) { hit = true; minDist = 0; reason = "Obstacle"; break; }
    if (res.dist < minDist) minDist = res.dist;
  }

  if (!hit) {
    for (const box of (mapData.obstacle_boxes || [])) {
      const { poly: boxPoly, x0, y0, x1, y1 } = getBoxCornersAndAabb(box);
      const dx = Math.max(0, x0 - agvMaxX, agvMinX - x1);
      const dy = Math.max(0, y0 - agvMaxY, agvMinY - y1);
      if (Math.hypot(dx, dy) > minDist && minDist < Infinity) continue;
      const res = polygonsOverlapAndDist(agvPoly, boxPoly);
      if (res.hit) { hit = true; minDist = 0; reason = "Wall"; break; }
      if (res.dist < minDist) minDist = res.dist;
    }
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
  const banner = document.getElementById("collisionStatus");
  const label = document.getElementById("collisionLabel");
  const ct = document.getElementById("clearanceText");
  if (hit) {
    banner.className = "status-banner hit";
    label.textContent = `COLLISION · ${reason}`;
    ct.textContent = "0.00 m";
  } else if (minDist < agv.margin) {
    banner.className = "status-banner warn";
    label.textContent = `TOO CLOSE · < ${agv.margin.toFixed(2)} m`;
    ct.textContent = `Min ${minDist.toFixed(2)} m`;
  } else {
    banner.className = "status-banner ok";
    label.textContent = "CLEAR";
    ct.textContent = Number.isFinite(minDist) ? `Min ${minDist.toFixed(2)} m` : "--";
  }
  document.getElementById("agvCoord").textContent = `(${agv.x.toFixed(2)}, ${agv.y.toFixed(2)}, ${Math.round(agv.yawDeg)}°)`;
}
