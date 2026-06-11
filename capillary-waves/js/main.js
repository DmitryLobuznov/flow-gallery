/*
 * Gravity–capillary surface waves, solved on the GPU with WebGL2.
 *
 * A height field h(x,y) on a periodic pond obeys the linearised
 * gravity–capillary wave equation with damping:
 *
 *   ∂²h/∂t² = g ∇²h − σ ∇⁴h − 2ζ ∂h/∂t
 *
 *   • g ∇²h    — gravity / long-wave restoring force (non-dispersive, ω≈√g·k)
 *   • σ ∇⁴h    — surface tension: the capillary term. It makes SHORT ripples
 *                travel faster (ω grows with k) — the hallmark of capillarity.
 *   • 2ζ ∂h/∂t — viscous damping, so energy bleeds away like a real pond.
 *
 * Time stepping is an explicit leapfrog; the stiff biharmonic term sets the
 * stability limit, so the internal dt is auto-capped from g and σ — the demo
 * can't be driven unstable. This is a fast LOCAL proxy: the exact
 * gravity–capillary dispersion ω² = (g k + σ k³)·tanh(kH) is non-local and
 * lives in the Python reference (/python), which is the source of truth.
 */

const N = 512; // simulation grid is N×N

const canvas = document.getElementById("sim");
const gl = canvas.getContext("webgl2", { antialias: false, preserveDrawingBuffer: false });
if (!gl) {
  document.body.innerHTML =
    '<p style="color:#e7e9ee;font-family:sans-serif;padding:40px">WebGL2 is not available in this browser.</p>';
  throw new Error("WebGL2 unavailable");
}
if (!gl.getExtension("EXT_color_buffer_float")) {
  document.body.innerHTML =
    '<p style="color:#e7e9ee;font-family:sans-serif;padding:40px">This GPU/browser cannot render to float textures (EXT_color_buffer_float).</p>';
  throw new Error("EXT_color_buffer_float unavailable");
}
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
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  return p;
}

const VERT = `#version 300 es
out vec2 vUV;
void main() {
  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
  vUV = p;
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

// Shared sampling helpers with periodic wrap (texture set to REPEAT).
const TAPS = `
uniform sampler2D uCurr;   // h at time n
uniform sampler2D uPrev;   // h at time n-1
uniform vec2 uTexel;       // 1/N
float H(sampler2D s, vec2 uv){ return texture(s, uv).r; }
`;

/* Leapfrog update: one explicit step of the damped wave equation. */
const STEP_FS = `#version 300 es
precision highp float;
in vec2 vUV;
out vec4 frag;
${TAPS}
uniform float uG;     // gravity coefficient (Laplacian)
uniform float uSigma; // surface tension (biharmonic)
uniform float uZeta;  // damping
uniform float uDt;
void main(){
  vec2 t = uTexel;
  // 13-point stencil samples of the current height.
  float c  = H(uCurr, vUV);
  float n  = H(uCurr, vUV + vec2(0.0,  t.y));
  float s  = H(uCurr, vUV + vec2(0.0, -t.y));
  float e  = H(uCurr, vUV + vec2( t.x, 0.0));
  float w  = H(uCurr, vUV + vec2(-t.x, 0.0));
  float ne = H(uCurr, vUV + vec2( t.x,  t.y));
  float nw = H(uCurr, vUV + vec2(-t.x,  t.y));
  float se = H(uCurr, vUV + vec2( t.x, -t.y));
  float sw = H(uCurr, vUV + vec2(-t.x, -t.y));
  float nn = H(uCurr, vUV + vec2(0.0,  2.0*t.y));
  float ss = H(uCurr, vUV + vec2(0.0, -2.0*t.y));
  float ee = H(uCurr, vUV + vec2( 2.0*t.x, 0.0));
  float ww = H(uCurr, vUV + vec2(-2.0*t.x, 0.0));

  float lap  = (n + s + e + w) - 4.0*c;                       // ∇²h
  float bih  = 20.0*c - 8.0*(n+s+e+w) + 2.0*(ne+nw+se+sw)     // ∇⁴h
             + (nn+ss+ee+ww);

  float hp = H(uPrev, vUV);
  float accel = uG * lap - uSigma * bih;
  // Damped leapfrog: hn = [2c − (1−ζdt)hp + dt²·accel] / (1+ζdt)
  float zd = uZeta * uDt;
  float hn = (2.0*c - (1.0 - zd)*hp + uDt*uDt*accel) / (1.0 + zd);
  frag = vec4(clamp(hn, -4.0, 4.0), 0.0, 0.0, 1.0); // safety clamp vs blow-up
}`;

/* Brush — add a smooth Gaussian impulse (a "drop"). */
const SPLAT_FS = `#version 300 es
precision highp float;
in vec2 vUV;
out vec4 frag;
uniform sampler2D uCurr;
uniform vec2 uPos;
uniform float uRadius;
uniform float uAmp;
void main(){
  float h = texture(uCurr, vUV).r;
  vec2 d = vUV - uPos;
  // wrap the shortest distance on the periodic domain
  d -= round(d);
  float g = exp(-dot(d, d) / (uRadius * uRadius));
  frag = vec4(h + uAmp * g, 0.0, 0.0, 1.0);
}`;

/* Display — shade the height field like a lit water surface. */
const SHOW_FS = `#version 300 es
precision highp float;
in vec2 vUV;
out vec4 frag;
uniform sampler2D uCurr;
uniform vec2 uTexel;
uniform int uCmap;
void main(){
  vec2 t = uTexel;
  float h  = texture(uCurr, vUV).r;
  float e  = texture(uCurr, vUV + vec2( t.x, 0.0)).r;
  float w  = texture(uCurr, vUV + vec2(-t.x, 0.0)).r;
  float n  = texture(uCurr, vUV + vec2(0.0,  t.y)).r;
  float s  = texture(uCurr, vUV + vec2(0.0, -t.y)).r;
  // Surface normal from the height gradient.
  vec3 nrm = normalize(vec3(-(e - w), -(n - s), 0.25));
  vec3 L = normalize(vec3(0.5, 0.6, 0.8));
  float diff = clamp(dot(nrm, L), 0.0, 1.0);
  float spec = pow(diff, 40.0);
  float slope = clamp((e - w) * (e - w) + (n - s) * (n - s), 0.0, 1.0);

  vec3 col;
  if (uCmap == 0) {              // caustic water
    vec3 deep = vec3(0.03, 0.18, 0.32);
    vec3 shallow = vec3(0.25, 0.65, 0.85);
    col = mix(deep, shallow, 0.35 + 0.65 * diff) + spec * vec3(0.9, 0.97, 1.0);
    col += slope * 6.0 * vec3(0.10, 0.18, 0.22);     // caustic glints
  } else if (uCmap == 1) {       // ocean
    col = mix(vec3(0.01, 0.06, 0.13), vec3(0.10, 0.45, 0.62), 0.4 + 0.6*diff);
    col += spec * vec3(0.7, 0.85, 0.95);
  } else if (uCmap == 2) {       // mercury
    float m = 0.45 + 0.55 * diff;
    col = vec3(m) + spec * vec3(1.0);
    col *= vec3(0.92, 0.95, 1.0);
  } else {                       // schlieren (signed height)
    float v = clamp(0.5 + 1.2 * h, 0.0, 1.0);
    col = mix(vec3(0.05,0.05,0.08), vec3(0.98,0.98,1.0), v);
  }
  frag = vec4(col, 1.0);
}`;

const progStep = program(STEP_FS);
const progSplat = program(SPLAT_FS);
const progShow = program(SHOW_FS);

const loc = (p, n) => gl.getUniformLocation(p, n);
const U = {
  stCurr: loc(progStep, "uCurr"), stPrev: loc(progStep, "uPrev"), stTexel: loc(progStep, "uTexel"),
  stG: loc(progStep, "uG"), stSigma: loc(progStep, "uSigma"), stZeta: loc(progStep, "uZeta"), stDt: loc(progStep, "uDt"),
  spCurr: loc(progSplat, "uCurr"), spPos: loc(progSplat, "uPos"), spRadius: loc(progSplat, "uRadius"), spAmp: loc(progSplat, "uAmp"),
  shCurr: loc(progShow, "uCurr"), shTexel: loc(progShow, "uTexel"), shCmap: loc(progShow, "uCmap"),
};

/* ---------- triple-buffer height textures ---------- */

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
// Three height buffers cycled as prev → curr → next.
let texPrev = makeTex();
let texCurr = makeTex();
let texNext = makeTex();

const fbo = gl.createFramebuffer();
function renderTo(tex) {
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
  gl.viewport(0, 0, N, N);
}
const vao = gl.createVertexArray();
function drawTri() { gl.bindVertexArray(vao); gl.drawArrays(gl.TRIANGLES, 0, 3); }

const texel = [1 / N, 1 / N];

function calm() {
  const zero = new Float32Array(N * N);
  for (const tx of [texPrev, texCurr, texNext]) {
    gl.bindTexture(gl.TEXTURE_2D, tx);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, N, N, gl.RED, gl.FLOAT, zero);
  }
}

/* ---------- parameters & stability ---------- */

const params = { g: 0.30, sigma: 0.40, zeta: 0.6, sub: 3, cmap: 0 };
let raining = true;

// Leapfrog stability: dt² (g·8 + σ·64) < 4. Stay at 80% of the bound.
function stableDt() {
  return 0.8 * 2.0 / Math.sqrt(params.g * 8.0 + params.sigma * 64.0 + 1e-3);
}

function step() {
  const dt = stableDt();
  for (let i = 0; i < params.sub; i++) {
    gl.useProgram(progStep);
    renderTo(texNext);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, texCurr);
    gl.uniform1i(U.stCurr, 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, texPrev);
    gl.uniform1i(U.stPrev, 1);
    gl.uniform2fv(U.stTexel, texel);
    gl.uniform1f(U.stG, params.g);
    gl.uniform1f(U.stSigma, params.sigma);
    gl.uniform1f(U.stZeta, params.zeta);
    gl.uniform1f(U.stDt, dt);
    drawTri();
    // cycle: prev ← curr, curr ← next, next ← old prev (reused buffer)
    const tmp = texPrev;
    texPrev = texCurr;
    texCurr = texNext;
    texNext = tmp;
  }
}

function splat(uv, amp, radius) {
  gl.useProgram(progSplat);
  renderTo(texNext);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, texCurr);
  gl.uniform1i(U.spCurr, 0);
  gl.uniform2f(U.spPos, uv[0], uv[1]);
  gl.uniform1f(U.spRadius, radius);
  gl.uniform1f(U.spAmp, amp);
  drawTri();
  // texNext now holds the splatted current; swap it in as curr.
  const tmp = texCurr;
  texCurr = texNext;
  texNext = tmp;
}

function show() {
  gl.useProgram(progShow);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, texCurr);
  gl.uniform1i(U.shCurr, 0);
  gl.uniform2fv(U.shTexel, texel);
  gl.uniform1i(U.shCmap, params.cmap);
  drawTri();
}

/* ---------- render loop ---------- */

let last = 0, frames = 0, acc = 0, rainAcc = 0;
const fpsEl = document.getElementById("fps");

function frame(now) {
  const dtMs = now - last;
  last = now;

  if (raining) {
    rainAcc += dtMs;
    if (rainAcc > 280) {
      rainAcc = 0;
      splat([Math.random(), Math.random()], 0.9, 0.012);
    }
  }
  step();
  show();

  frames++; acc += dtMs;
  if (acc >= 500) { fpsEl.textContent = Math.round((frames * 1000) / acc) + " fps"; frames = 0; acc = 0; }
  requestAnimationFrame(frame);
}

/* ---------- sizing ---------- */

function resize() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.floor(window.innerWidth * dpr);
  canvas.height = Math.floor(window.innerHeight * dpr);
}
window.addEventListener("resize", resize);
resize();

/* ---------- UI ---------- */

const bind = (id, key, fmt) => {
  const el = document.getElementById(id);
  const out = document.getElementById(id + "Val");
  const apply = () => {
    const v = key === "sub" || key === "cmap" ? parseInt(el.value, 10) : parseFloat(el.value);
    params[key] = v;
    if (out) out.textContent = fmt ? fmt(v) : v;
  };
  el.addEventListener("input", apply);
  apply();
};
bind("tens", "sigma", (v) => v.toFixed(2));
bind("grav", "g", (v) => v.toFixed(2));
bind("damp", "zeta", (v) => v.toFixed(1));
bind("sub", "sub", (v) => v);
document.getElementById("cmap").addEventListener("change", (e) => { params.cmap = parseInt(e.target.value, 10); });

const btnRain = document.getElementById("rain");
btnRain.addEventListener("click", () => {
  raining = !raining;
  btnRain.textContent = raining ? "🌧 Rain: on" : "🌤 Rain: off";
  btnRain.classList.toggle("primary", raining);
});
document.getElementById("reset").addEventListener("click", calm);

window.addEventListener("keydown", (e) => {
  if (e.code === "Space") { e.preventDefault(); btnRain.click(); }
  if (e.key === "r" || e.key === "R") calm();
});

/* ---------- pointer ---------- */

let painting = false;
function pointerUV(e) {
  const r = canvas.getBoundingClientRect();
  return [(e.clientX - r.left) / r.width, 1.0 - (e.clientY - r.top) / r.height];
}
canvas.addEventListener("pointerdown", (e) => { painting = true; splat(pointerUV(e), 1.1, 0.014); });
canvas.addEventListener("pointermove", (e) => { if (painting) splat(pointerUV(e), 0.35, 0.010); });
window.addEventListener("pointerup", () => { painting = false; });
canvas.addEventListener("contextmenu", (e) => e.preventDefault());

/* ---------- go ---------- */

calm();
// a few seed drops so the gallery thumbnail is lively from frame one
for (let i = 0; i < 6; i++) splat([Math.random(), Math.random()], 1.0, 0.012);
requestAnimationFrame(frame);
