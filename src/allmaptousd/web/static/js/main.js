// Startup wiring (runs last).
// Classic script: shares the global scope with the other js/*.js files (see index.html).

(function init() {
  resizeCanvas();
  new ResizeObserver(resizeCanvas).observe(canvas.parentElement);
  updateIsaacPullRange();

  const dz = document.getElementById("dropzone");
  const fi = document.getElementById("fileInput");
  dz.addEventListener("click", () => fi.click());
  fi.addEventListener("change", () => {
    uploadFiles(fi.files);
    fi.value = "";
  });
  dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("dragover"); });
  dz.addEventListener("dragleave", () => dz.classList.remove("dragover"));
  dz.addEventListener("drop", (e) => {
    e.preventDefault(); dz.classList.remove("dragover");
    uploadFiles(e.dataTransfer.files);
  });

  // Parameter edits only take effect on Rebuild: flag that until the next map payload arrives.
  // (Seal Unknown rebuilds by itself and Zone Collision only changes the 2D check.)
  const markRebuildNeeded = (e) => {
    if (!mapData || ["sealUnknown", "forbiddenCollision"].includes(e.target.id)) return;
    setRebuildPending(true);
  };
  const buildPanel = document.querySelector("#accBuild .acc-body");
  buildPanel.addEventListener("input", markRebuildNeeded);
  buildPanel.addEventListener("change", markRebuildNeeded);

  fetch("/api/isaac/config").then((r) => r.json()).then((cfg) => {
    isaacConfigBase = cfg;
    document.getElementById("isaacHost").value = cfg.host || "127.0.0.1";
    document.getElementById("isaacPort").value = cfg.port || 8226;
    document.getElementById("isaacPushMode").value = cfg.push_mode || "reference";
    document.getElementById("isaacPrim").value = cfg.target_prim_path || "/World/AGV_Map";
    if (cfg.pull) {
      document.getElementById("isaacPullRoot").value = cfg.pull.root_prim_path;
      document.getElementById("isaacPullExclude").value = cfg.pull.exclude_prim_path;
      document.getElementById("isaacPullFloor").value = cfg.pull.floor_elevation_m;
      document.getElementById("isaacPullClearance").value = cfg.pull.floor_clearance_m;
      document.getElementById("isaacPullGeometry").value = cfg.pull.geometry;
      updateIsaacPullRange();
    }
  }).catch(() => {});
})();
