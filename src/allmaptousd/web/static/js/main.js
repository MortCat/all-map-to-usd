// Startup wiring (runs last).
// Classic script: shares the global scope with the other js/*.js files (see index.html).

(function init() {
  resizeCanvas();

  const dz = document.getElementById("dropzone");
  const fi = document.getElementById("fileInput");
  dz.addEventListener("click", () => fi.click());
  fi.addEventListener("change", () => uploadFiles(fi.files));
  dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("dragover"); });
  dz.addEventListener("dragleave", () => dz.classList.remove("dragover"));
  dz.addEventListener("drop", (e) => {
    e.preventDefault(); dz.classList.remove("dragover");
    uploadFiles(e.dataTransfer.files);
  });

  fetch("/api/isaac/config").then((r) => r.json()).then((cfg) => {
    isaacConfigBase = cfg;
    document.getElementById("isaacHost").value = cfg.host || "127.0.0.1";
    document.getElementById("isaacPort").value = cfg.port || 8226;
    document.getElementById("isaacPushMode").value = cfg.push_mode || "reference";
    document.getElementById("isaacPrim").value = cfg.target_prim_path || "/World/AGV_Map";
  }).catch(() => {});
})();
