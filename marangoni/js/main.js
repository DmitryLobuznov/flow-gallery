/*
 * Marangoni–Bénard convection cells, solved on the GPU with WebGL2.
 *
 * Right at onset, the convection pattern is governed by the Swift–Hohenberg
 * equation — the canonical amplitude model for Rayleigh–Bénard / Marangoni
 * convection. For the convection amplitude u(x,y):
 *
 *   ∂ₜu = r·u − (1 + ∇²)² u + g·u² − u³
 *       = (r−1)u − 2∇²u − ∇⁴u + g·u² − u³
 *
 *   • (1+∇²)² selects a finite cell size (wavelength 2π): a band of modes grows.
 *   • the cubic −u³ saturates the amplitude (always bounded — very robust).
 *   • the quadratic g·u² breaks symmetry and favours HEXAGONS (Marangoni cells)
 *     over rolls (g = 0).
 *
 * One fragment-shader pass per sub-step evaluates the RHS with a 5-point ∇² and
 * a 13-point ∇⁴ stencil and takes an explicit Euler step; dt is auto-capped to
 * the stability limit. The Python reference (/python) uses an exact spectral
 * semi-implicit step and is the source of truth.
 */

const N = 128;             // grid
const CELLS = 10.0;        // convection cells across the box
const LEN = CELLS * 2.0 * Math.PI;
const DX = LEN / N;

const canvas = document.getElementById("sim");
const gl = canvas.getContext("webgl2", { antialias: false });
if (!gl) {
  document.body.innerHTML =
    '<p style="color:#e7e9ee;font-family:sans-serif;padding:40px">WebGL2 is not available in this browser.</p>';
  throw new Error("WebGL2 unavailable");
}
if (!gl.getExtension("EXT_color_buffer_float")) {
  document.body.innerHTML =
    '<p style="color:#e7e9ee;font-family:sans-serif;padding:40px">This GPU/browser cannot render to float textures.</p>';
  throw new Error("EXT_color_buffer_float unavailable");
}
const canLinear = !!gl.getExtension("OES_texture_float_linear");

function compile(type, src) {
  const sh = gl.createShader(type);
  gl.shaderSource(sh, src); gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(sh) + "\n" + src);
  return sh;
}
function program(fragSrc) {
  const p = gl.createProgram();
  gl.attachShader(p, compile(gl.VERTEX_SHADER, VERT));
  gl.attachShader(p, compile(gl.FRAGMENT_SHADER, fragSrc));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  return p;
}
const VERT = `#version 300 es
out vec2 vUV;
void main(){ vec2 p = vec2((gl_VertexID<<1)&2, gl_VertexID&2); vUV = p; gl_Position = vec4(p*2.0-1.0,0.0,1.0); }`;

/* Swift–Hohenberg explicit step. */
const STEP_FS = `#version 300 es
precision highp float;
in vec2 vUV; out vec4 frag;
uniform sampler2D uU;
uniform vec2 uTexel;
uniform float uR, uG, uDt, uDx;
float S(vec2 uv){ return texture(uU, uv).r; }
void main(){
  vec2 t = uTexel;
  float c  = S(vUV);
  float n  = S(vUV+vec2(0.0,t.y)), s = S(vUV+vec2(0.0,-t.y));
  float e  = S(vUV+vec2(t.x,0.0)), w = S(vUV+vec2(-t.x,0.0));
  float ne = S(vUV+vec2(t.x,t.y)), nw = S(vUV+vec2(-t.x,t.y));
  float se = S(vUV+vec2(t.x,-t.y)), sw = S(vUV+vec2(-t.x,-t.y));
  float nn = S(vUV+vec2(0.0,2.0*t.y)), ss = S(vUV+vec2(0.0,-2.0*t.y));
  float ee = S(vUV+vec2(2.0*t.x,0.0)), ww = S(vUV+vec2(-2.0*t.x,0.0));
  float dx2 = uDx*uDx;
  float lap = ((n+s+e+w) - 4.0*c)/dx2;                                  // ∇²
  float bih = (20.0*c - 8.0*(n+s+e+w) + 2.0*(ne+nw+se+sw) + (nn+ss+ee+ww))/(dx2*dx2); // ∇⁴
  float rhs = (uR-1.0)*c - 2.0*lap - bih + uG*c*c - c*c*c;
  frag = vec4(clamp(c + uDt*rhs, -4.0, 4.0), 0.0, 0.0, 1.0);
}`;

/* Paint a blob to perturb the field. */
const SPLAT_FS = `#version 300 es
precision highp float;
in vec2 vUV; out vec4 frag;
uniform sampler2D uU; uniform vec2 uPos; uniform float uR, uA;
void main(){
  float c = texture(uU, vUV).r;
  vec2 d = vUV - uPos; d -= round(d);
  frag = vec4(c + uA*exp(-dot(d,d)/(uR*uR)), 0.0, 0.0, 1.0);
}`;

const SHOW_FS = `#version 300 es
precision highp float;
in vec2 vUV; out vec4 frag;
uniform sampler2D uU; uniform float uScale; uniform int uCmap;
vec3 inferno(float t){
  const vec3 c0=vec3(0.0002,0.0016,-0.0194),c1=vec3(0.1065,0.5639,3.9327);
  const vec3 c2=vec3(11.6024,-3.9728,-15.9423),c3=vec3(-41.7039,17.4363,44.3541);
  const vec3 c4=vec3(77.1629,-33.4023,-81.8073),c5=vec3(-71.3194,32.6263,73.2095);
  const vec3 c6=vec3(25.1311,-12.2426,-23.0703);
  return c0+t*(c1+t*(c2+t*(c3+t*(c4+t*(c5+t*c6)))));
}
vec3 magma(float t){
  const vec3 c0=vec3(-0.0023,-0.0009,-0.0184),c1=vec3(0.2516,0.6775,2.4944);
  const vec3 c2=vec3(8.3537,-3.5777,0.3144),c3=vec3(-27.6687,14.2647,-13.6492);
  const vec3 c4=vec3(52.1761,-27.9436,12.9441),c5=vec3(-50.7685,29.0405,4.2341);
  const vec3 c6=vec3(18.6552,-11.4894,-5.6010);
  return c0+t*(c1+t*(c2+t*(c3+t*(c4+t*(c5+t*c6)))));
}
vec3 viridis(float t){
  const vec3 c0=vec3(0.2777,0.0054,0.3341),c1=vec3(0.1050,1.4046,1.3845);
  const vec3 c2=vec3(-0.3308,0.2148,0.0952),c3=vec3(-4.6342,-5.7991,-19.3324);
  const vec3 c4=vec3(6.2282,14.1799,56.6905),c5=vec3(4.7763,-13.7451,-65.3530);
  const vec3 c6=vec3(-5.4354,4.6456,26.3124);
  return c0+t*(c1+t*(c2+t*(c3+t*(c4+t*(c5+t*c6)))));
}
void main(){
  float u = texture(uU, vUV).r;
  float t = clamp(0.5 + 0.5*u/uScale, 0.0, 1.0);
  vec3 col;
  if (uCmap==0) col = inferno(t);
  else if (uCmap==1) col = magma(t);
  else if (uCmap==2) col = viridis(t);
  else col = mix(vec3(0.02,0.10,0.30), vec3(0.98,0.86,0.30), t); // thermal
  frag = vec4(col, 1.0);
}`;

const progStep = program(STEP_FS);
const progSplat = program(SPLAT_FS);
const progShow = program(SHOW_FS);
const loc = (p, n) => gl.getUniformLocation(p, n);
const U = {
  stU: loc(progStep, "uU"), stTexel: loc(progStep, "uTexel"), stR: loc(progStep, "uR"),
  stG: loc(progStep, "uG"), stDt: loc(progStep, "uDt"), stDx: loc(progStep, "uDx"),
  spU: loc(progSplat, "uU"), spPos: loc(progSplat, "uPos"), spR: loc(progSplat, "uR"), spA: loc(progSplat, "uA"),
  shU: loc(progShow, "uU"), shScale: loc(progShow, "uScale"), shCmap: loc(progShow, "uCmap"),
};

function makeTex() {
  const tx = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tx);
  gl.texStorage2D(gl.TEXTURE_2D, 1, gl.R32F, N, N);
  const f = canLinear ? gl.LINEAR : gl.NEAREST;
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, f);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, f);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
  return tx;
}
let texA = makeTex();
let texB = makeTex();
const fbo = gl.createFramebuffer();
function renderTo(tex) {
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
  gl.viewport(0, 0, N, N);
}
const vao = gl.createVertexArray();
function drawTri() { gl.bindVertexArray(vao); gl.drawArrays(gl.TRIANGLES, 0, 3); }
const texel = [1 / N, 1 / N];

const params = { r: 0.3, g: 1.0, sub: 30, cmap: 0, scale: 1.0 };

function seed() {
  const data = new Float32Array(N * N);
  for (let i = 0; i < data.length; i++) data[i] = 0.05 * (Math.random() - 0.5) * 2.0;
  gl.bindTexture(gl.TEXTURE_2D, texA);
  gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, N, N, gl.RED, gl.FLOAT, data);
}

// Explicit-stability cap for the biharmonic (k⁴) + Laplacian (k²) terms.
function stableDt() {
  const s = 64.0 / DX ** 4 + 16.0 / DX ** 2 + Math.abs(params.r - 1.0) + 4.0;
  return 0.8 * 2.0 / s;
}

function step() {
  const dt = stableDt();
  for (let i = 0; i < params.sub; i++) {
    gl.useProgram(progStep);
    renderTo(texB);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, texA); gl.uniform1i(U.stU, 0);
    gl.uniform2fv(U.stTexel, texel);
    gl.uniform1f(U.stR, params.r); gl.uniform1f(U.stG, params.g);
    gl.uniform1f(U.stDt, dt); gl.uniform1f(U.stDx, DX);
    drawTri();
    [texA, texB] = [texB, texA];
  }
}

function splat(uv, amp) {
  gl.useProgram(progSplat);
  renderTo(texB);
  gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, texA); gl.uniform1i(U.spU, 0);
  gl.uniform2f(U.spPos, uv[0], uv[1]); gl.uniform1f(U.spR, 0.04); gl.uniform1f(U.spA, amp);
  drawTri();
  [texA, texB] = [texB, texA];
}

const readBuf = new Float32Array(N * N * 4);
function autoScale() {
  // periodic, cheap: estimate amplitude from a readback every so often
  renderTo(texA);
  gl.readPixels(0, 0, N, N, gl.RGBA, gl.FLOAT, readBuf);
  let s2 = 0, m = 0, cnt = 0;
  for (let i = 0; i < N * N; i++) { const v = readBuf[i * 4]; if (Number.isFinite(v)) { m += v; cnt++; } }
  m /= cnt;
  for (let i = 0; i < N * N; i++) { const v = readBuf[i * 4]; if (Number.isFinite(v)) s2 += (v - m) ** 2; }
  const std = Math.sqrt(s2 / cnt);
  params.scale = Math.max(0.15, 2.2 * std);
}

function show() {
  gl.useProgram(progShow);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, texA); gl.uniform1i(U.shU, 0);
  gl.uniform1f(U.shScale, params.scale); gl.uniform1i(U.shCmap, params.cmap);
  drawTri();
}

let running = true, last = 0, frames = 0, acc = 0, scaleTick = 0;
const fpsEl = document.getElementById("fps");
function frame(now) {
  if (running) step();
  if (++scaleTick % 20 === 0) autoScale();
  show();
  frames++; acc += now - last; last = now;
  if (acc >= 500) { fpsEl.textContent = Math.round((frames * 1000) / acc) + " fps"; frames = 0; acc = 0; }
  requestAnimationFrame(frame);
}

function resize() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.floor(window.innerWidth * dpr);
  canvas.height = Math.floor(window.innerHeight * dpr);
}
window.addEventListener("resize", resize);
resize();

const bind = (id, key, fmt) => {
  const el = document.getElementById(id);
  const out = document.getElementById(id + "Val");
  const apply = () => {
    const v = key === "sub" || key === "cmap" ? parseInt(el.value, 10) : parseFloat(el.value);
    params[key] = v;
    if (out) out.textContent = fmt ? fmt(v) : v;
  };
  el.addEventListener("input", apply); apply();
};
bind("rr", "r", (v) => v.toFixed(2));
bind("gg", "g", (v) => v.toFixed(1));
bind("sub", "sub", (v) => v);
document.getElementById("cmap").addEventListener("change", (e) => { params.cmap = parseInt(e.target.value, 10); });

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

let painting = false, sign = 1;
function pointerUV(e) {
  const r = canvas.getBoundingClientRect();
  return [(e.clientX - r.left) / r.width, 1.0 - (e.clientY - r.top) / r.height];
}
canvas.addEventListener("pointerdown", (e) => { painting = true; sign = e.button === 2 ? -1 : 1; splat(pointerUV(e), 0.8 * sign); });
canvas.addEventListener("pointermove", (e) => { if (painting) splat(pointerUV(e), 0.4 * sign); });
window.addEventListener("pointerup", () => { painting = false; });
canvas.addEventListener("contextmenu", (e) => e.preventDefault());

seed();
requestAnimationFrame(frame);

/* test hook for headless verification */
window.__mar = {
  run(steps) { const w = running; running = false; for (let i = 0; i < steps; i++) step(); running = w; },
  stats() {
    renderTo(texA);
    gl.readPixels(0, 0, N, N, gl.RGBA, gl.FLOAT, readBuf);
    let mn = Infinity, mx = -Infinity, sum = 0, sum2 = 0, bad = 0;
    for (let i = 0; i < N * N; i++) { const v = readBuf[i * 4]; if (!Number.isFinite(v)) { bad++; continue; } mn = Math.min(mn, v); mx = Math.max(mx, v); sum += v; sum2 += v * v; }
    const c = N * N - bad, mean = sum / c;
    return { min: mn, max: mx, std: Math.sqrt(sum2 / c - mean * mean), nonFinite: bad };
  },
};
