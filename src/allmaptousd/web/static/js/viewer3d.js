// Shared WebGL2 renderer for 3D model thumbnails and top views.
// One context draws every model and the picture is copied into ordinary 2D canvases, so any number
// of thumbnails can exist (browsers allow only a handful of live WebGL contexts).
// Classic script: shares the global scope with the other js/*.js files (see index.html).

const viewer3d = (() => {
  const glCanvas = document.createElement("canvas");
  const gl = glCanvas.getContext("webgl2", { antialias: true, premultipliedAlpha: false, preserveDrawingBuffer: true });
  if (!gl) return null;

  const compile = (type, source) => {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
    return shader;
  };
  const program = gl.createProgram();
  gl.attachShader(program, compile(gl.VERTEX_SHADER, `#version 300 es
    in vec3 aPos;
    in vec3 aColor;
    in vec2 aUv;
    uniform mat4 uMvp;
    out vec3 vPos;
    out vec3 vColor;
    out vec2 vUv;
    void main() { vPos = aPos; vColor = aColor; vUv = aUv; gl_Position = uMvp * vec4(aPos, 1.0); }`));
  // Flat shading from screen-space derivatives: no normals needed, and both sides of a face light up.
  // The colour is the part's material colour, times its diffuse texture when it has one.
  gl.attachShader(program, compile(gl.FRAGMENT_SHADER, `#version 300 es
    precision highp float;
    in vec3 vPos;
    in vec3 vColor;
    in vec2 vUv;
    uniform vec3 uLight;
    uniform bool uTextured;
    uniform sampler2D uTexture;
    out vec4 outColor;
    void main() {
      vec3 n = normalize(cross(dFdx(vPos), dFdy(vPos)));
      vec3 base = uTextured ? vColor * texture(uTexture, vUv).rgb : vColor;
      outColor = vec4(base * (0.38 + 0.62 * abs(dot(n, uLight))), 1.0);
    }`));
  gl.linkProgram(program);
  const loc = {
    pos: gl.getAttribLocation(program, "aPos"),
    color: gl.getAttribLocation(program, "aColor"),
    uv: gl.getAttribLocation(program, "aUv"),
    mvp: gl.getUniformLocation(program, "uMvp"),
    light: gl.getUniformLocation(program, "uLight"),
    textured: gl.getUniformLocation(program, "uTextured"),
    texture: gl.getUniformLocation(program, "uTexture"),
  };

  const meshes = new Map();  // model id -> { vao, groups: [{ first, count, texture }] }

  function attribute(location, data, size, type, normalized) {
    gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(location);
    gl.vertexAttribPointer(location, size, type, normalized, 0, 0);
  }

  function loadTexture(url) {
    return new Promise((resolve) => {
      const image = new Image();
      image.onload = () => {
        const texture = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, texture);
        gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);  // USD texture coordinates start at the bottom
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);
        gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
        gl.generateMipmap(gl.TEXTURE_2D);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
        resolve(texture);
      };
      image.onerror = () => resolve(null);
      image.src = url;
    });
  }

  // `data`: the /api/models/{id}/preview payload, arrays already decoded. Resolves once the
  // textures are loaded, so the first picture drawn already shows them.
  async function upload(id, data, textureUrl) {
    const textures = await Promise.all(Array.from({ length: data.textures }, (_, i) => loadTexture(textureUrl(i))));
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    attribute(loc.pos, data.positions, 3, gl.FLOAT, false);
    attribute(loc.color, data.colors, 3, gl.UNSIGNED_BYTE, true);
    attribute(loc.uv, data.uvs, 2, gl.FLOAT, false);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, gl.createBuffer());
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, data.indices, gl.STATIC_DRAW);
    gl.bindVertexArray(null);
    const groups = data.groups.map(([first, count, texture]) => ({ first, count, texture: textures[texture] || null }));
    meshes.set(id, { vao, groups });
  }

  // --- column-major 4x4 matrices ---
  const mul = (a, b) => {
    const o = new Float32Array(16);
    for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
      o[c * 4 + r] = s;
    }
    return o;
  };
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const norm = (a) => { const l = Math.hypot(...a) || 1; return a.map((v) => v / l); };
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  function lookAt(eye, target, up) {
    const z = norm(sub(eye, target)), x = norm(cross(up, z)), y = cross(z, x);
    return new Float32Array([x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0,
      -dot(x, eye), -dot(y, eye), -dot(z, eye), 1]);
  }
  function perspective(fovy, aspect, near, far) {
    const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
    return new Float32Array([f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) * nf, -1, 0, 0, 2 * far * near * nf, 0]);
  }
  function ortho(l, r, b, t, n, f) {
    return new Float32Array([2 / (r - l), 0, 0, 0, 0, 2 / (t - b), 0, 0, 0, 0, -2 / (f - n), 0,
      -(r + l) / (r - l), -(t + b) / (t - b), -(f + n) / (f - n), 1]);
  }

  function draw(id, mvp, light, width, height) {
    const mesh = meshes.get(id);
    if (!mesh) return false;
    glCanvas.width = width;
    glCanvas.height = height;
    gl.viewport(0, 0, width, height);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);
    gl.useProgram(program);
    gl.uniformMatrix4fv(loc.mvp, false, mvp);
    gl.uniform3fv(loc.light, norm(light));
    gl.uniform1i(loc.texture, 0);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindVertexArray(mesh.vao);
    for (const group of mesh.groups) {
      gl.uniform1i(loc.textured, group.texture ? 1 : 0);
      gl.bindTexture(gl.TEXTURE_2D, group.texture);
      gl.drawElements(gl.TRIANGLES, group.count * 3, gl.UNSIGNED_INT, group.first * 12);
    }
    gl.bindVertexArray(null);
    return true;
  }

  // Orbit view of a model (size [dx, dy, dz], centred on x/y, base on z = 0) into a 2D canvas.
  function drawOrbit(id, size, target, yaw, pitch) {
    const [dx, dy, dz] = size;
    const centre = [0, 0, dz / 2];
    const radius = Math.max(0.5 * Math.hypot(dx, dy, dz), 1e-3);
    const fov = (35 * Math.PI) / 180;
    const distance = (radius / Math.sin(fov / 2)) * 1.05;
    const eye = [
      centre[0] + distance * Math.cos(pitch) * Math.cos(yaw),
      centre[1] + distance * Math.cos(pitch) * Math.sin(yaw),
      centre[2] + distance * Math.sin(pitch),
    ];
    const view = lookAt(eye, centre, [0, 0, 1]);
    const proj = perspective(fov, target.width / target.height, distance * 0.05, distance * 4);
    const light = [eye[0] - centre[0] + 0.4 * distance, eye[1] - centre[1] - 0.3 * distance, eye[2] + distance];
    if (!draw(id, mul(proj, view), light, target.width, target.height)) return false;
    const c = target.getContext("2d");
    c.clearRect(0, 0, target.width, target.height);
    c.drawImage(glCanvas, 0, 0);
    return true;
  }

  // Top view (looking down -Z, +Y up in the image) at `pxPerMeter`, returned as a canvas.
  function renderTop(id, size, pxPerMeter) {
    const [dx, dy, dz] = size;
    const width = Math.max(2, Math.round(dx * pxPerMeter)), height = Math.max(2, Math.round(dy * pxPerMeter));
    const view = lookAt([0, 0, dz + 1], [0, 0, 0], [0, 1, 0]);
    const proj = ortho(-dx / 2, dx / 2, -dy / 2, dy / 2, 0.5, dz + 1.5);
    if (!draw(id, mul(proj, view), [0.35, 0.45, 1.0], width, height)) return null;
    const out = document.createElement("canvas");
    out.width = width;
    out.height = height;
    out.getContext("2d").drawImage(glCanvas, 0, 0);
    return out;
  }

  return { upload, drawOrbit, renderTop, has: (id) => meshes.has(id) };
})();
