/*
 * Thin-film dewetting, solved on the GPU with WebGL2.
 *
 * A liquid film of thickness h(x,y) on a substrate obeys the lubrication
 * (thin-film) equation
 *
 *   ∂h/∂t = −∇·[ h³ ∇( γ∇²h + Π(h) ) ]
 *
 *   • γ∇²h — surface tension smoothing the film's curvature.
 *   • Π(h) — disjoining pressure from a bounded double-well wetting energy
 *            W(h) = κ(h−h_p)²(h−h_d)², Π = −W'. A film whose mean thickness sits
 *            on the central hump is spinodally unstable: it ruptures and retracts
 *            into droplets (height ≈ h_d) on an ultrathin precursor (≈ h_p).
 *   • h³   — the lubrication mobility. It is degenerate (→0 as h→0), which is
 *            what makes dewetting look different from plain phase separation:
 *            drops connected by thin necks, retracting rims.
 *
 * Two fragment-shader passes per sub-step: pass 1 builds the pressure p, pass 2
 * applies the conservative flux update h ← h − dt·∇·(h³∇p) using face-averaged
 * mobilities so total volume ∫h is conserved. The explicit step is auto-capped
 * to the stability limit. The Python reference (/python) uses a semi-implicit
 * Fourier scheme and is the source of truth.
 */

const N = 256;

const canvas = document.getElementById("sim");
const gl = canvas.getContext("webgl2", { antialias: false });
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

function compile(type, src) {
  const sh = gl.createShader(type);
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(sh) + "\n" + src);
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
void main(){ vec2 p = vec2((gl_VertexID<<1)&2, gl_VertexID&2); vUV = p; gl_Position = vec4(p*2.0-1.0,0.0,1.0); }`;

const SAMP = `
uniform sampler2D uH;
uniform vec2 uTexel;
float Hs(vec2 uv){ return texture(uH, uv).r; }
`;

// Disjoining pressure Π(h) = −W'(h), W = κ(h−hp)²(h−hd)².
const PI_GLSL = `
uniform float uHp, uHd, uKap;
float Pi(float h){
  return -2.0*uKap*(h-uHp)*(h-uHd)*(2.0*h-uHp-uHd);
}`;

/* Pass 1 — pressure p = γ∇²h + Π(h). */
const PRES_FS = `#version 300 es
precision highp float;
in vec2 vUV; out vec4 frag;
${SAMP}
${PI_GLSL}
uniform float uGamma;
void main(){
  vec2 t = uTexel;
  float c = Hs(vUV);
  float n = Hs(vUV+vec2(0.0,t.y)), s = Hs(vUV+vec2(0.0,-t.y));
  float e = Hs(vUV+vec2(t.x,0.0)), w = Hs(vUV+vec2(-t.x,0.0));
  float lap = (n+s+e+w) - 4.0*c;
  frag = vec4(uGamma*lap + Pi(c), 0.0, 0.0, 1.0);
}`;

/* Pass 2 — conservative update h ← h − dt·∇·(h³∇p), face-averaged mobility. */
const UPD_FS = `#version 300 es
precision highp float;
in vec2 vUV; out vec4 frag;
uniform sampler2D uH;     // thickness
uniform sampler2D uP;     // pressure (from pass 1)
uniform vec2 uTexel;
uniform float uDt;
float mob(float h){ float hp = max(h,0.0); return hp*hp*hp; } // h³, clamped ≥0
void main(){
  vec2 t = uTexel;
  float hc = texture(uH, vUV).r;
  float hn = texture(uH, vUV+vec2(0.0,t.y)).r, hs = texture(uH, vUV+vec2(0.0,-t.y)).r;
  float he = texture(uH, vUV+vec2(t.x,0.0)).r, hw = texture(uH, vUV+vec2(-t.x,0.0)).r;
  float pc = texture(uP, vUV).r;
  float pn = texture(uP, vUV+vec2(0.0,t.y)).r, ps = texture(uP, vUV+vec2(0.0,-t.y)).r;
  float pe = texture(uP, vUV+vec2(t.x,0.0)).r, pw = texture(uP, vUV+vec2(-t.x,0.0)).r;
  // Face mobilities = average of the two adjacent cell mobilities.
  float me = 0.5*(mob(hc)+mob(he)), mw = 0.5*(mob(hc)+mob(hw));
  float mn = 0.5*(mob(hc)+mob(hn)), ms = 0.5*(mob(hc)+mob(hs));
  // Conservative divergence of the flux m·∇p across the four faces.
  float div = me*(pe-pc) - mw*(pc-pw) + mn*(pn-pc) - ms*(pc-ps);
  float h2 = hc - uDt*div;
  frag = vec4(clamp(h2, -0.5, 6.0), 0.0, 0.0, 1.0); // safety clamp
}`;

/* Brush — locally thin (dig a hole) or thicken the film. */
const SPLAT_FS = `#version 300 es
precision highp float;
in vec2 vUV; out vec4 frag;
uniform sampler2D uH;
uniform vec2 uPos; uniform float uRadius, uAmp;
void main(){
  float h = texture(uH, vUV).r;
  vec2 d = vUV - uPos; d -= round(d);
  float g = exp(-dot(d,d)/(uRadius*uRadius));
  frag = vec4(h + uAmp*g, 0.0, 0.0, 1.0);
}`;

/* Display — thickness as a lit metallic film. */
const SHOW_FS = `#version 300 es
precision highp float;
in vec2 vUV; out vec4 frag;
uniform sampler2D uH;
uniform vec2 uTexel;
uniform float uHp, uHd;
uniform int uCmap;
vec3 ramp(float t, int m){
  if (m==0){ // copper
    return clamp(vec3(1.25,0.65,0.35)*t + vec3(0.05,0.02,0.0), 0.0, 1.0);
  } else if (m==1){ // gold
    return clamp(vec3(1.3,0.95,0.35)*pow(t,0.8)+vec3(0.04,0.03,0.02),0.0,1.0);
  } else if (m==2){ // ice
    return mix(vec3(0.02,0.05,0.10), vec3(0.75,0.92,1.0), t);
  } else { // viridis-ish
    return mix(vec3(0.27,0.0,0.33), vec3(0.99,0.91,0.14), t);
  }
}
void main(){
  vec2 t = uTexel;
  float h  = texture(uH, vUV).r;
  float e  = texture(uH, vUV+vec2(t.x,0.0)).r, w = texture(uH, vUV+vec2(-t.x,0.0)).r;
  float n  = texture(uH, vUV+vec2(0.0,t.y)).r, s = texture(uH, vUV+vec2(0.0,-t.y)).r;
  vec3 nrm = normalize(vec3(-(e-w)*4.0, -(n-s)*4.0, 1.0));
  vec3 L = normalize(vec3(0.5,0.6,0.8));
  float diff = clamp(dot(nrm,L),0.0,1.0);
  float spec = pow(diff, 32.0);
  float tt = clamp((h-uHp)/(uHd-uHp), 0.0, 1.0);
  vec3 col = ramp(tt, uCmap)*(0.5+0.5*diff) + spec*0.5;
  frag = vec4(col, 1.0);
}`;

const progPres = program(PRES_FS);
const progUpd = program(UPD_FS);
const progSplat = program(SPLAT_FS);
const progShow = program(SHOW_FS);
const loc = (p, n) => gl.getUniformLocation(p, n);
const U = {
  prH: loc(progPres, "uH"), prTexel: loc(progPres, "uTexel"), prGamma: loc(progPres, "uGamma"),
  prHp: loc(progPres, "uHp"), prHd: loc(progPres, "uHd"), prKap: loc(progPres, "uKap"),
  upH: loc(progUpd, "uH"), upP: loc(progUpd, "uP"), upTexel: loc(progUpd, "uTexel"), upDt: loc(progUpd, "uDt"),
  spH: loc(progSplat, "uH"), spPos: loc(progSplat, "uPos"), spRadius: loc(progSplat, "uRadius"), spAmp: loc(progSplat, "uAmp"),
  shH: loc(progShow, "uH"), shTexel: loc(progShow, "uTexel"), shHp: loc(progShow, "uHp"), shHd: loc(progShow, "uHd"), shCmap: loc(progShow, "uCmap"),
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
let texH = makeTex();
let texH2 = makeTex();
const texP = makeTex();

const fbo = gl.createFramebuffer();
function renderTo(tex) {
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
  gl.viewport(0, 0, N, N);
}
const vao = gl.createVertexArray();
function drawTri() { gl.bindVertexArray(vao); gl.drawArrays(gl.TRIANGLES, 0, 3); }
const texel = [1 / N, 1 / N];

const params = { mean: 0.8, hp: 0.5, hd: 1.8, kappa: 0.45, gamma: 2.5, sub: 200, cmap: 0 };

function seed() {
  const data = new Float32Array(N * N);
  for (let i = 0; i < data.length; i++) data[i] = params.mean + 0.04 * (Math.random() - 0.5) * 2.0;
  gl.bindTexture(gl.TEXTURE_2D, texH);
  gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, N, N, gl.RED, gl.FLOAT, data);
}

// Explicit-stability cap: dt·M0·(γ·k⁴max + c·k²max) < 2, with 5-point stencil
// k²max≈8, k⁴max≈64, M0=hd³ worst-case mobility, c≈max|Π'|.
function stableDt() {
  const M0 = params.hd ** 3;
  const k = params.kappa;
  // max |Π'| = max|W''| over [hp,hd] ≈ 2κ(hd−hp)² at the wells.
  const c = 2 * k * (params.hd - params.hp) ** 2 + 1e-3;
  return 0.7 * 2.0 / (M0 * (params.gamma * 64.0 + c * 8.0));
}

function step() {
  const dt = stableDt();
  for (let i = 0; i < params.sub; i++) {
    // Pass 1: pressure into texP.
    gl.useProgram(progPres);
    renderTo(texP);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, texH);
    gl.uniform1i(U.prH, 0); gl.uniform2fv(U.prTexel, texel);
    gl.uniform1f(U.prGamma, params.gamma);
    gl.uniform1f(U.prHp, params.hp); gl.uniform1f(U.prHd, params.hd); gl.uniform1f(U.prKap, params.kappa);
    drawTri();
    // Pass 2: conservative update into texH2.
    gl.useProgram(progUpd);
    renderTo(texH2);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, texH); gl.uniform1i(U.upH, 0);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, texP); gl.uniform1i(U.upP, 1);
    gl.uniform2fv(U.upTexel, texel); gl.uniform1f(U.upDt, dt);
    drawTri();
    [texH, texH2] = [texH2, texH];
  }
}

function splat(uv, amp) {
  gl.useProgram(progSplat);
  renderTo(texH2);
  gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, texH); gl.uniform1i(U.spH, 0);
  gl.uniform2f(U.spPos, uv[0], uv[1]); gl.uniform1f(U.spRadius, 0.03); gl.uniform1f(U.spAmp, amp);
  drawTri();
  [texH, texH2] = [texH2, texH];
}

function show() {
  gl.useProgram(progShow);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, texH); gl.uniform1i(U.shH, 0);
  gl.uniform2fv(U.shTexel, texel);
  gl.uniform1f(U.shHp, params.hp); gl.uniform1f(U.shHd, params.hd); gl.uniform1i(U.shCmap, params.cmap);
  drawTri();
}

let running = true, last = 0, frames = 0, acc = 0;
const fpsEl = document.getElementById("fps");
function frame(now) {
  if (running) step();
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
bind("mean", "mean", (v) => v.toFixed(2), seed);
bind("kappa", "kappa", (v) => v.toFixed(2));
bind("gamma", "gamma", (v) => v.toFixed(1));
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

let painting = false, sign = -1;
function pointerUV(e) {
  const r = canvas.getBoundingClientRect();
  return [(e.clientX - r.left) / r.width, 1.0 - (e.clientY - r.top) / r.height];
}
canvas.addEventListener("pointerdown", (e) => { painting = true; sign = e.button === 2 ? 1 : -1; splat(pointerUV(e), 0.6 * sign); });
canvas.addEventListener("pointermove", (e) => { if (painting) splat(pointerUV(e), 0.3 * sign); });
window.addEventListener("pointerup", () => { painting = false; });
canvas.addEventListener("contextmenu", (e) => e.preventDefault());

seed();
requestAnimationFrame(frame);

/* ---------- test hook (used by headless verification) ----------
 * Exposes a way to fast-forward the solver without rendering and to read back
 * the thickness field's statistics, so the GPU scheme can be validated
 * numerically (mass conservation, instability growth) even where software
 * rendering is too slow to show dewetting visually. Harmless in normal use. */
window.__dewet = {
  run(steps) { const was = running; running = false; for (let i = 0; i < steps; i++) step(); running = was; },
  stats() {
    renderTo(texH); // bind texH as the FBO read source
    const buf = new Float32Array(N * N * 4);
    gl.readPixels(0, 0, N, N, gl.RGBA, gl.FLOAT, buf);
    let mn = Infinity, mx = -Infinity, sum = 0, sum2 = 0, bad = 0;
    for (let i = 0; i < N * N; i++) {
      const v = buf[i * 4];
      if (!Number.isFinite(v)) { bad++; continue; }
      mn = Math.min(mn, v); mx = Math.max(mx, v); sum += v; sum2 += v * v;
    }
    const n = N * N - bad, mean = sum / n;
    return { min: mn, max: mx, mean, std: Math.sqrt(sum2 / n - mean * mean), nonFinite: bad };
  },
};
