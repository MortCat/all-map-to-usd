// Canvas rendering of the map, AGV, ruler and overlays.
// Classic script: shares the global scope with the other js/*.js files (see index.html).

// --- Rendering ---
function render() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#0b0f17";
  ctx.fillRect(0, 0, w, h);
  updateScaleBar();

  if (document.getElementById("layerGrid").checked) {
    const tl = screenToWorld(0, 0), br = screenToWorld(w, h);
    const step = cam.scale < 15 ? 5 : (cam.scale < 35 ? 2 : 1);
    ctx.strokeStyle = "rgba(148, 163, 184, 0.11)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let gx = Math.floor(tl.x / step) * step; gx <= br.x; gx += step) {
      const sx = worldToScreen(gx, 0).x;
      ctx.moveTo(sx, 0); ctx.lineTo(sx, h);
    }
    for (let gy = Math.floor(br.y / step) * step; gy <= tl.y; gy += step) {
      const sy = worldToScreen(0, gy).y;
      ctx.moveTo(0, sy); ctx.lineTo(w, sy);
    }
    ctx.stroke();

    const o = worldToScreen(0, 0);
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = "rgba(239, 68, 68, 0.55)";
    ctx.beginPath(); ctx.moveTo(o.x, o.y); ctx.lineTo(o.x + 28, o.y); ctx.stroke();
    ctx.strokeStyle = "rgba(34, 197, 94, 0.55)";
    ctx.beginPath(); ctx.moveTo(o.x, o.y); ctx.lineTo(o.x, o.y - 28); ctx.stroke();
  }

  if (!mapData) return;

  if (document.getElementById("layerBoxes").checked) {
    ctx.fillStyle = "rgba(148, 163, 184, 0.62)";
    ctx.strokeStyle = "rgba(226, 232, 240, 0.65)";
    ctx.lineWidth = 1.1;
    for (const poly of (mapData.obstacle_polygons || [])) {
      if (!poly || poly.length < 3) continue;
      ctx.beginPath();
      poly.forEach(([wx, wy], idx) => {
        const s = worldToScreen(wx, wy);
        idx === 0 ? ctx.moveTo(s.x, s.y) : ctx.lineTo(s.x, s.y);
      });
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
    }
    for (const [x0, y0, x1, y1] of (mapData.obstacle_boxes || [])) {
      const poly = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
      ctx.beginPath();
      poly.forEach(([wx, wy], idx) => {
        const s = worldToScreen(wx, wy);
        idx === 0 ? ctx.moveTo(s.x, s.y) : ctx.lineTo(s.x, s.y);
      });
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
    }
  }

  if (document.getElementById("layerPoints").checked) {
    ctx.fillStyle = "rgba(56, 189, 248, 0.75)";
    const ptSize = cam.scale > 60 ? 2.5 : 1.6;
    for (const [px, py] of mapData.raw_points.concat(mapData.drawn_points || [])) {
      const s = worldToScreen(px, py);
      ctx.fillRect(s.x - ptSize * 0.5, s.y - ptSize * 0.5, ptSize, ptSize);
    }
  }

  if (document.getElementById("layerForbidden").checked) {
    for (const z of mapData.forbidden_zones) {
      if (!z.vertices || z.vertices.length < 3) continue;
      ctx.beginPath();
      z.vertices.forEach(([vx, vy], idx) => {
        const s = worldToScreen(vx, vy);
        idx === 0 ? ctx.moveTo(s.x, s.y) : ctx.lineTo(s.x, s.y);
      });
      ctx.closePath();
      ctx.fillStyle = "rgba(239, 68, 68, 0.25)";
      ctx.strokeStyle = "rgba(248, 113, 113, 0.9)";
      ctx.lineWidth = 1.5;
      ctx.fill(); ctx.stroke();

      let cx = 0, cy = 0;
      z.vertices.forEach(([vx, vy]) => { cx += vx; cy += vy; });
      const cs = worldToScreen(cx / z.vertices.length, cy / z.vertices.length);
      ctx.fillStyle = "#fca5a5";
      ctx.font = "600 11px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(z.name, cs.x, cs.y);
    }
  }

  if (document.getElementById("layerRoutes").checked) {
    ctx.strokeStyle = "rgba(34, 211, 238, 0.8)";
    ctx.lineWidth = 1.6;
    for (const p of (mapData.paths || [])) {
      ctx.beginPath();
      p.points.forEach(([px, py], i) => {
        const s = worldToScreen(px, py);
        i === 0 ? ctx.moveTo(s.x, s.y) : ctx.lineTo(s.x, s.y);
      });
      ctx.stroke();
    }
    const r = Math.max(3, Math.min(9, 0.15 * cam.scale));
    ctx.font = "600 10px sans-serif";
    ctx.textAlign = "left";
    for (const st of (mapData.stations || [])) {
      const s = worldToScreen(st.x, st.y);
      const tip = worldToScreen(st.x + Math.cos(st.yaw) * 0.35, st.y + Math.sin(st.yaw) * 0.35);
      ctx.fillStyle = "#22d3ee";
      ctx.beginPath(); ctx.arc(s.x, s.y, r, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = "#a5f3fc"; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(s.x, s.y); ctx.lineTo(tip.x, tip.y); ctx.stroke();
      if (cam.scale > 25) { ctx.fillStyle = "#a5f3fc"; ctx.fillText(st.name, s.x + r + 3, s.y - r); }
    }
  }

  drawPlacements();

  if (document.getElementById("layerAgv").checked) {
    evaluateCollision();
    const corners = getAgvCorners().map(([x, y]) => worldToScreen(x, y));
    const status = lastCollision.hit ? "hit" : (lastCollision.minDist < agv.margin ? "warn" : "ok");
    const fill = { hit: "rgba(239, 68, 68, 0.45)", warn: "rgba(245, 158, 11, 0.38)", ok: "rgba(34, 197, 94, 0.32)" };
    const stroke = { hit: "#f87171", warn: "#fbbf24", ok: "#4ade80" };
    if (agv.margin > 0) {
      const m = agv.margin;
      const outer = getAgvCorners(m).map(([x, y]) => worldToScreen(x, y));
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = stroke[status];
      ctx.lineWidth = 1;
      ctx.beginPath();
      outer.forEach((p, i) => (i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y)));
      ctx.closePath(); ctx.stroke();
      ctx.setLineDash([]);
    }
    ctx.beginPath();
    corners.forEach((p, i) => (i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y)));
    ctx.closePath();
    ctx.fillStyle = fill[status];
    ctx.strokeStyle = stroke[status];
    ctx.lineWidth = 2;
    ctx.fill(); ctx.stroke();

    const centerS = worldToScreen(agv.x, agv.y);
    const rad = (agv.yawDeg * Math.PI) / 180;
    const frontS = worldToScreen(
      agv.x + Math.cos(rad) * (agv.length * 0.42),
      agv.y + Math.sin(rad) * (agv.length * 0.42)
    );
    ctx.strokeStyle = "#fbbf24";
    ctx.lineWidth = 2.5;
    ctx.beginPath(); ctx.moveTo(centerS.x, centerS.y); ctx.lineTo(frontS.x, frontS.y); ctx.stroke();
    ctx.fillStyle = "#fbbf24";
    ctx.beginPath(); ctx.arc(frontS.x, frontS.y, 4, 0, Math.PI * 2); ctx.fill();
  }

  drawMapEdits();

  if (rulerP1 && !rulerP2) {
    const start = worldToScreen(rulerP1.x, rulerP1.y);
    ctx.fillStyle = "#f59e0b";
    ctx.beginPath(); ctx.arc(start.x, start.y, 4, 0, Math.PI * 2); ctx.fill();
  }
  if (rulerP1 && rulerP2) {
    const s1 = worldToScreen(rulerP1.x, rulerP1.y);
    const s2 = worldToScreen(rulerP2.x, rulerP2.y);
    ctx.strokeStyle = "#f59e0b";
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 4]);
    ctx.beginPath(); ctx.moveTo(s1.x, s1.y); ctx.lineTo(s2.x, s2.y); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#f59e0b";
    ctx.beginPath(); ctx.arc(s1.x, s1.y, 4, 0, Math.PI * 2); ctx.fill();
    ctx.beginPath(); ctx.arc(s2.x, s2.y, 4, 0, Math.PI * 2); ctx.fill();

    const dist = Math.hypot(rulerP2.x - rulerP1.x, rulerP2.y - rulerP1.y);
    const mx = 0.5 * (s1.x + s2.x), my = 0.5 * (s1.y + s2.y) - 12;
    const label = `${dist.toFixed(3)} m (${Math.round(dist * 1000)} mm)`;
    ctx.font = "bold 12px monospace";
    const tw = ctx.measureText(label).width + 14;
    ctx.fillStyle = "rgba(11, 15, 23, 0.92)";
    ctx.fillRect(mx - tw * 0.5, my - 13, tw, 20);
    ctx.strokeStyle = "rgba(245, 158, 11, 0.65)";
    ctx.lineWidth = 1;
    ctx.strokeRect(mx - tw * 0.5, my - 13, tw, 20);
    ctx.fillStyle = "#fbbf24";
    ctx.textAlign = "center";
    ctx.fillText(label, mx, my + 1);
  }
}
