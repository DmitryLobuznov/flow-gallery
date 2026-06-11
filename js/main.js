/*
 * Cahn–Hilliard phase separation, solved on the GPU with WebGL2.
 *
 *   ∂c/∂t = M ∇²μ,      μ = c³ − c − κ ∇²c
 *
 * The order parameter c (∈ ≈[−1, 1]) lives in a single-channel float
 * texture. Each simulation sub-step is two fragment-shader passes —
 * one computes the chemical potential μ, the next applies the
 * conservative update — ping-ponging between two textures.
 *
 * Space discretisation: isotropic 9-point Laplacian, periodic BCs
 * (GL_REPEAT wrap). Time: explicit Euler with several sub-steps per
 * animation frame. The matching Python reference in /python uses a
 * semi-implicit spectral scheme and is the source of truth for the
 * physics; this file trades a little accuracy for real-time speed.
 */

const N = 256; // simulation grid is N×N

const canvas = document.getElementById("sim");
const gl = canvas.getContext("webgl2", { antialias: false, preserveDrawingBuffer: false });
if (!gl) {
  document.body.innerHTML =
    '<p style="color:#e7e9ee;font-family:sans-serif;padding:40px">WebGL2 is not available in this browser.</p>';
  throw new Error("WebGL2 unavailable");
}
const floatLinear = gl.getExtension("EXT_color_buffer_float");
if (!floatLinear) {
  document.body.innerHTML =
    '<p style="color:#e7e9ee;font-family:sans-serif;padding:40px">This GPU/browser cannot render to float textures (EXT_color_buffer_float).</p>';
  throw new Error("EXT_color_buffer_float unavailable");
}
// Linear filtering of float textures needs this extension; fall back to
// NEAREST when it's missing (the sim samples at texel centres, so the only
// cost is a slightly blockier upscale on display).
const canLinear = !!gl.getExtension("OES_texture_float_linear");

/* ---------- shader helpers ---------- */

function compile(type, src) {
  const sh = gl.createShader(type);
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    throw new Error(gl.getShaderInfoLog(sh) + "\n" + src);
  }
  return sh;
}

function program(fragSrc, vertSrc = VERT) {
  const p = gl.createProgram();
  gl.attachShader(p, compile(gl.VERTEX_SHADER, vertSrc));
  gl.attachShader(p, compile(gl.FRAGMENT_SHADER, fragSrc));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
    throw new Error(gl.getProgramInfoLog(p));
  }
  return p;
}

// Full-screen triangle; vUV spans [0,1]².
const VERT = `#version 300 es
out vec2 vUV;
void main() {
  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
  vUV = p;
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

// Shared GLSL: isotropic 9-point Laplacian with periodic sampling.
const LAP = `
uniform sampler2D uField;
uniform vec2 uTexel;        // 1/N
float S(vec2 uv){ return texture(uField, uv).r; }
float lap(vec2 uv){
  float c  = S(uv);
  float n  = S(uv + vec2(0.0,  uTexel.y));
  float s  = S(uv + vec2(0.0, -uTexel.y));
  float e  = S(uv + vec2( uTexel.x, 0.0));
  float w  = S(uv + vec2(-uTexel.x, 0.0));
  float ne = S(uv + vec2( uTexel.x,  uTexel.y));
  float nw = S(uv + vec2(-uTexel.x,  uTexel.y));
  float se = S(uv + vec2( uTexel.x, -uTexel.y));
  float sw = S(uv + vec2(-uTexel.x, -uTexel.y));
  // (1/6)[4(n+s+e+w) + (diagonals) - 20c],  h = 1
  return (4.0*(n+s+e+w) + (ne+nw+se+sw) - 20.0*c) / 6.0;
}`;

/* Pass 1 — chemical potential μ = c³ − c − κ ∇²c */
const MU_FS = `#version 300 es
precision highp float;
in vec2 vUV;
out vec4 frag;
${LAP}
uniform float uKappa;
void main(){
  float c = S(vUV);
  float mu = c*c*c - c - uKappa * lap(vUV);
  frag = vec4(mu, 0.0, 0.0, 1.0);
}`;

/* Pass 2 — conservative update c ← c + dt · M · ∇²μ */
const UPDATE_FS = `#version 300 es
precision highp float;
in vec2 vUV;
out vec4 frag;
${LAP}
uniform sampler2D uC;
uniform float uDt;
uniform float uM;
void main(){
  float c = texture(uC, vUV).r;
  float c2 = c + uDt * uM * lap(vUV);   // uField bound to μ here
  frag = vec4(c2, 0.0, 0.0, 1.0);
}`;

/* Brush — add a soft Gaussian blob of phase under the pointer */
const SPLAT_FS = `#version 300 es
precision highp float;
in vec2 vUV;
out vec4 frag;
uniform sampler2D uField;
uniform vec2 uPos;       // in [0,1]²
uniform float uRadius;   // in uv units
uniform float uAmp;
void main(){
  float c = texture(uField, vUV).r;
  vec2 d = vUV - uPos;
  float g = exp(-dot(d, d) / (uRadius * uRadius));
  frag = vec4(clamp(c + uAmp * g, -1.2, 1.2), 0.0, 0.0, 1.0);
}`;

/* Display — map c∈[−1,1] to a colormap */
const SHOW_FS = `#version 300 es
precision highp float;
in vec2 vUV;
out vec4 frag;
uniform sampler2D uField;
uniform int uCmap;

// Polynomial colormap fits (Matt DesLauriers, MIT).
vec3 viridis(float t){
  const vec3 c0=vec3(0.2777,0.0054,0.3341),c1=vec3(0.1050,1.4046,1.3845);
  const vec3 c2=vec3(-0.3308,0.2148,0.0952),c3=vec3(-4.6342,-5.7991,-19.3324);
  const vec3 c4=vec3(6.2282,14.1799,56.6905),c5=vec3(4.7763,-13.7451,-65.3530);
  const vec3 c6=vec3(-5.4354,4.6456,26.3124);
  return c0+t*(c1+t*(c2+t*(c3+t*(c4+t*(c5+t*c6)))));
}
vec3 magma(float t){
  const vec3 c0=vec3(-0.0023,-0.0009,-0.0184),c1=vec3(0.2516,0.6775,2.4944);
  const vec3 c2=vec3(8.3537,-3.5777,0.3144),c3=vec3(-27.6687,14.2647,-13.6492);
  const vec3 c4=vec3(52.1761,-27.9436,12.9441),c5=vec3(-50.7685,29.0405,4.2341);
  const vec3 c6=vec3(18.6552,-11.4894,-5.6010);
  return c0+t*(c1+t*(c2+t*(c3+t*(c4+t*(c5+t*c6)))));
}
vec3 inferno(float t){
  const vec3 c0=vec3(0.0002,0.0016,-0.0194),c1=vec3(0.1065,0.5639,3.9327);
  const vec3 c2=vec3(11.6024,-3.9728,-15.9423),c3=vec3(-41.7039,17.4363,44.3541);
  const vec3 c4=vec3(77.1629,-33.4023,-81.8073),c5=vec3(-71.3194,32.6263,73.2095);
  const vec3 c6=vec3(25.1311,-12.2426,-23.0703);
  return c0+t*(c1+t*(c2+t*(c3+t*(c4+t*(c5+t*c6)))));
}

void main(){
  float c = texture(uField, vUV).r;
  float t = clamp(0.5 * (c + 1.0), 0.0, 1.0);
  vec3 col;
  if      (uCmap == 0) col = viridis(t);
  else if (uCmap == 1) col = magma(t);
  else if (uCmap == 2) col = inferno(t);
  else if (uCmap == 3) col = mix(vec3(0.04,0.05,0.07), vec3(0.93,0.95,0.98), smoothstep(0.35,0.65,t));
  else                 col = mix(vec3(0.02,0.18,0.20), vec3(0.99,0.80,0.34), smoothstep(0.30,0.70,t));
  frag = vec4(col, 1.0);
}`;

const progMu = program(MU_FS);
const progUpdate = program(UPDATE_FS);
const progSplat = program(SPLAT_FS);
const progShow = program(SHOW_FS);

// Cache uniform locations.
const loc = (p, name) => gl.getUniformLocation(p, name);
const U = {
  muField: loc(progMu, "uField"), muTexel: loc(progMu, "uTexel"), muKappa: loc(progMu, "uKappa"),
  upField: loc(progUpdate, "uField"), upC: loc(progUpdate, "uC"), upTexel: loc(progUpdate, "uTexel"),
  upDt: loc(progUpdate, "uDt"), upM: loc(progUpdate, "uM"),
  spField: loc(progSplat, "uField"), spPos: loc(progSplat, "uPos"),
  spRadius: loc(progSplat, "uRadius"), spAmp: loc(progSplat, "uAmp"),
  shField: loc(progShow, "uField"), shCmap: loc(progShow, "uCmap"),
};

/* ---------- ping-pong textures ---------- */

function makeTex() {
  const t = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, t);
  gl.texStorage2D(gl.TEXTURE_2D, 1, gl.R32F, N, N);
  const filter = canLinear ? gl.LINEAR : gl.NEAREST;
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
  return t;
}

let texC0 = makeTex(); // c (ping)
let texC1 = makeTex(); // c (pong)
const texMu = makeTex(); // μ

const fbo = gl.createFramebuffer();
function renderTo(tex) {
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
  gl.viewport(0, 0, N, N);
}

const vao = gl.createVertexArray(); // empty; vertices come from gl_VertexID
function drawTri() { gl.bindVertexArray(vao); gl.drawArrays(gl.TRIANGLES, 0, 3); }

/* ---------- state & controls ---------- */

const params = { mean: 0.0, kappa: 1.0, M: 1.0, dt: 0.025, sub: 25, cmap: 0 };
let running = true;

function seed() {
  // Random initial condition: mean composition + small noise.
  const data = new Float32Array(N * N);
  const mean = params.mean;
  // Deterministic-ish but varied noise (no external RNG dependency).
  for (let i = 0; i < data.length; i++) {
    data[i] = mean + 0.18 * (Math.random() - 0.5) * 2.0;
  }
  gl.bindTexture(gl.TEXTURE_2D, texC0);
  gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, N, N, gl.RED, gl.FLOAT, data);
}

const texel = [1 / N, 1 / N];

function step() {
  for (let s = 0; s < params.sub; s++) {
    // Pass 1: μ from current c (texC0)
    gl.useProgram(progMu);
    renderTo(texMu);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, texC0);
    gl.uniform1i(U.muField, 0);
    gl.uniform2fv(U.muTexel, texel);
    gl.uniform1f(U.muKappa, params.kappa);
    drawTri();

    // Pass 2: c1 = c0 + dt·M·∇²μ
    gl.useProgram(progUpdate);
    renderTo(texC1);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, texMu);
    gl.uniform1i(U.upField, 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, texC0);
    gl.uniform1i(U.upC, 1);
    gl.uniform2fv(U.upTexel, texel);
    gl.uniform1f(U.upDt, params.dt);
    gl.uniform1f(U.upM, params.M);
    drawTri();

    [texC0, texC1] = [texC1, texC0]; // swap
  }
}

function splat(uv, amp) {
  gl.useProgram(progSplat);
  renderTo(texC1);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, texC0);
  gl.uniform1i(U.spField, 0);
  gl.uniform2f(U.spPos, uv[0], uv[1]);
  gl.uniform1f(U.spRadius, 0.04);
  gl.uniform1f(U.spAmp, amp);
  drawTri();
  [texC0, texC1] = [texC1, texC0];
}

function show() {
  gl.useProgram(progShow);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, texC0);
  gl.uniform1i(U.shField, 0);
  gl.uniform1i(U.shCmap, params.cmap);
  drawTri();
}

/* ---------- render loop ---------- */

let last = performance.now ? performance.now() : 0;
let frames = 0, acc = 0;
const fpsEl = document.getElementById("fps");

function frame(now) {
  if (running) step();
  show();

  frames++;
  acc += now - last;
  last = now;
  if (acc >= 500) {
    fpsEl.textContent = Math.round((frames * 1000) / acc) + " fps";
    frames = 0; acc = 0;
  }
  requestAnimationFrame(frame);
}

/* ---------- canvas sizing ---------- */

function resize() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.floor(window.innerWidth * dpr);
  canvas.height = Math.floor(window.innerHeight * dpr);
}
window.addEventListener("resize", resize);
resize();

/* ---------- UI wiring ---------- */

const bind = (id, key, fmt, after) => {
  const el = document.getElementById(id);
  const out = document.getElementById(id + "Val");
  const apply = () => {
    const v = key === "sub" || key === "cmap" ? parseInt(el.value, 10) : parseFloat(el.value);
    params[key] = v;
    if (out) out.textContent = fmt ? fmt(v) : v;
    if (after) after();
  };
  el.addEventListener("input", apply);
  apply();
};

bind("mean", "mean", (v) => v.toFixed(2));
bind("kappa", "kappa", (v) => v.toFixed(2));
bind("mob", "M", (v) => v.toFixed(2));
bind("dt", "dt", (v) => v.toFixed(3));
bind("sub", "sub", (v) => v);
document.getElementById("cmap").addEventListener("change", (e) => {
  params.cmap = parseInt(e.target.value, 10);
});

const btnPlay = document.getElementById("playPause");
btnPlay.addEventListener("click", () => {
  running = !running;
  btnPlay.textContent = running ? "⏸ Pause" : "▶ Play";
  btnPlay.classList.toggle("primary", running);
});
document.getElementById("reset").addEventListener("click", seed);

window.addEventListener("keydown", (e) => {
  if (e.code === "Space") { e.preventDefault(); btnPlay.click(); }
  if (e.key === "r" || e.key === "R") seed();
});

/* ---------- pointer painting ---------- */

let painting = false, sign = 1;
function pointerUV(e) {
  const r = canvas.getBoundingClientRect();
  const x = (e.clientX - r.left) / r.width;
  const y = 1.0 - (e.clientY - r.top) / r.height; // flip Y for GL
  return [x, y];
}
canvas.addEventListener("pointerdown", (e) => {
  painting = true;
  sign = e.button === 2 ? -1 : 1;
  splat(pointerUV(e), 0.9 * sign);
});
canvas.addEventListener("pointermove", (e) => {
  if (painting) splat(pointerUV(e), 0.5 * sign);
});
window.addEventListener("pointerup", () => { painting = false; });
canvas.addEventListener("contextmenu", (e) => e.preventDefault());

/* ---------- go ---------- */

seed();
requestAnimationFrame(frame);
