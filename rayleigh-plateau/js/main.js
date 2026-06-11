/*
 * Rayleigh–Plateau jet break-up, solved on the GPU with WebGL2.
 *
 * A liquid column of radius h(z) is unstable: surface tension pinches it into a
 * chain of droplets. A radius perturbation of wavelength λ grows when λ > 2πR₀,
 * fastest at λ ≈ 9.0 R₀ — that sets the number of drops.
 *
 * The state (radius h, axial velocity v) lives in one row of an RGBA32F texture
 * (R = h, G = v); the column runs along x and is periodic. Each sub-step is an
 * RK2 integration of the 1D slender-jet (lubrication) equations
 *
 *   ∂ₜ(h²) = −∂_z(h² v),   ∂ₜv = −v∂_z v − ∂_z κ + (3ν/h²)∂_z(h²∂_z v)
 *   κ = 1/(h√(1+h_z²)) − h_zz/(1+h_z²)^{3/2}.
 *
 * A display shader revolves h(z) into a lit tube. Because the 1D model can't
 * represent the neck actually snapping, the column auto-resets once it pinches,
 * giving a clean loop. The Python reference (/python) is the source of truth.
 */

const NZ = 512; // points along the jet

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
gl.getExtension("OES_texture_float_linear");

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

/* RK2 sub-step: out = base + factor·dt·rhs(eval). Both base and eval are state
 * textures (R=h, G=v); sampling is 1D along x with periodic wrap. */
const STEP_FS = `#version 300 es
precision highp float;
in vec2 vUV; out vec4 frag;
uniform sampler2D uBase;   // state to add onto
uniform sampler2D uEval;   // state at which to evaluate the RHS
uniform float uDx, uDt, uFactor, uVisc, uHfloor;
float H(sampler2D s, float x){ return texture(s, vec2(x, 0.5)).r; }
float Vv(sampler2D s, float x){ return texture(s, vec2(x, 0.5)).g; }
float texel; // set in main
float curv(float x){
  // central differences for h_z, h_zz (grid spacing uDx, texture step = 1/NZ)
  float t = texel;
  float hm = max(H(uEval, x - t), uHfloor);
  float hc = max(H(uEval, x),     uHfloor);
  float hp = max(H(uEval, x + t), uHfloor);
  float hz  = (hp - hm) / (2.0*uDx);
  float hzz = (hp - 2.0*hc + hm) / (uDx*uDx);
  float d = sqrt(1.0 + hz*hz);
  return 1.0/(hc*d) - hzz/(d*d*d);
}
void main(){
  texel = 1.0/float(${NZ});
  float x = vUV.x;
  float t = texel;
  float hc = max(H(uEval, x), uHfloor);
  float vc = Vv(uEval, x);
  // neighbours
  float hL = max(H(uEval, x - t), uHfloor), hR = max(H(uEval, x + t), uHfloor);
  float vL = Vv(uEval, x - t), vR = Vv(uEval, x + t);
  // ∂ₜh² = −∂_z(h² v)  → ∂ₜh = −∂_z(h² v)/(2h)
  float fluxR = hR*hR*vR, fluxL = hL*hL*vL;
  float dh2dz = (fluxR - fluxL)/(2.0*uDx);
  float dh = -dh2dz/(2.0*hc);
  // momentum
  float dvz = (vR - vL)/(2.0*uDx);
  float dkz = (curv(x + t) - curv(x - t))/(2.0*uDx);
  // viscous: (3ν/h²) ∂_z(h² ∂_z v)
  float gR = hR*hR*(Vv(uEval, x + 2.0*t) - vc)/(2.0*uDx);
  float gL = hL*hL*(vc - Vv(uEval, x - 2.0*t))/(2.0*uDx);
  float visc = (3.0*uVisc/(hc*hc)) * (gR - gL)/(2.0*uDx);
  float dv = -vc*dvz - dkz + visc;

  float hb = texture(uBase, vec2(x,0.5)).r;
  float vb = texture(uBase, vec2(x,0.5)).g;
  float hn = max(hb + uFactor*uDt*dh, uHfloor);
  float vn = vb + uFactor*uDt*dv;
  frag = vec4(hn, vn, 0.0, 1.0);
}`;

/* Display — revolve h(z) into a lit tube on the 2D canvas. */
const SHOW_FS = `#version 300 es
precision highp float;
in vec2 vUV; out vec4 frag;
uniform sampler2D uState;
uniform float uR0;
uniform int uCmap;
void main(){
  float h = texture(uState, vec2(vUV.x, 0.5)).r;
  float rmax = uR0*1.9;
  float rpix = h/rmax;                 // radius as fraction of half-height
  float y = vUV.y - 0.5;               // −0.5..0.5
  float a = abs(y)/max(rpix*0.5, 1e-4);
  if (a > 1.0) { frag = vec4(0.02,0.02,0.03,1.0); return; }  // background
  float cosang = sqrt(max(1.0 - a*a, 0.0));   // cylinder facing camera
  float sn = y/max(rpix*0.5,1e-4);
  float light = clamp(0.32 + 0.8*cosang - 0.22*sn, 0.0, 1.0);
  float spec = pow(cosang, 22.0);
  vec3 col;
  if (uCmap==0)      col = vec3(0.30+0.7*light, 0.55+0.45*light, 0.98);          // water
  else if (uCmap==1) col = vec3(0.95,0.78,0.35)*(0.4+0.7*light);                 // gold
  else if (uCmap==2) col = vec3(light)*vec3(0.9,0.95,1.0);                       // mercury
  else               col = mix(vec3(0.1,0.0,0.2), vec3(1.0,0.4,0.2), light);     // magma
  col += spec*0.8;
  frag = vec4(col, 1.0);
}`;

const progStep = program(STEP_FS);
const progShow = program(SHOW_FS);
const loc = (p, n) => gl.getUniformLocation(p, n);
const U = {
  stBase: loc(progStep, "uBase"), stEval: loc(progStep, "uEval"),
  stDx: loc(progStep, "uDx"), stDt: loc(progStep, "uDt"), stFac: loc(progStep, "uFactor"),
  stVisc: loc(progStep, "uVisc"), stHf: loc(progStep, "uHfloor"),
  shState: loc(progShow, "uState"), shR0: loc(progShow, "uR0"), shCmap: loc(progShow, "uCmap"),
};

function makeTex() {
  const tx = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tx);
  gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA32F, NZ, 1);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);   // periodic column
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  return tx;
}
let texA = makeTex();   // state
let texB = makeTex();   // scratch / next
let texMid = makeTex(); // RK2 midpoint

const fbo = gl.createFramebuffer();
function renderTo(tex) {
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
  gl.viewport(0, 0, NZ, 1);
}
const vao = gl.createVertexArray();
function drawTri() { gl.bindVertexArray(vao); gl.drawArrays(gl.TRIANGLES, 0, 3); }

const params = { r0: 1.0, drops: 4, visc: 0.15, sub: 4, cmap: 0 };
const HFLOOR = 0.06;
const DT = 1.5e-3;
const LENGTH = 40.0; // fixed column length → constant grid spacing (uniform stability)

// Each bead spans λ = LENGTH/drops. Rayleigh–Plateau is unstable for λ > 2πR₀
// (≈6.28), so drops is capped at 6 (λ ≈ 6.7) in the UI; fewer drops → longer,
// faster-growing waves.
function jetLength() { return LENGTH; }
function dx() { return LENGTH / NZ; }

function seed() {
  const data = new Float32Array(NZ * 4);
  const phase = Math.random() * Math.PI * 2;
  const L = jetLength();
  for (let i = 0; i < NZ; i++) {
    const z = (i / NZ) * L;
    const h = params.r0 * (1.0 + 0.04 * Math.cos(2 * Math.PI * params.drops * z / L + phase)
                               + 0.004 * (Math.random() - 0.5));
    data[i * 4] = h; data[i * 4 + 1] = 0; data[i * 4 + 2] = 0; data[i * 4 + 3] = 1;
  }
  gl.bindTexture(gl.TEXTURE_2D, texA);
  gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, NZ, 1, gl.RGBA, gl.FLOAT, data);
}

// One RK2 sub-step using two RHS evaluations (midpoint method).
function rk2() {
  const d = dx();
  // pass 1: mid = state + 0.5 dt rhs(state)
  gl.useProgram(progStep);
  renderTo(texMid);
  gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, texA); gl.uniform1i(U.stBase, 0);
  gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, texA); gl.uniform1i(U.stEval, 1);
  gl.uniform1f(U.stDx, d); gl.uniform1f(U.stDt, DT); gl.uniform1f(U.stFac, 0.5);
  gl.uniform1f(U.stVisc, params.visc); gl.uniform1f(U.stHf, HFLOOR);
  drawTri();
  // pass 2: next = state + dt rhs(mid)
  renderTo(texB);
  gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, texA); gl.uniform1i(U.stBase, 0);
  gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, texMid); gl.uniform1i(U.stEval, 1);
  gl.uniform1f(U.stFac, 1.0);
  drawTri();
  [texA, texB] = [texB, texA];
}

const readBuf = new Float32Array(NZ * 4);
function minRadius() {
  renderTo(texA);
  gl.readPixels(0, 0, NZ, 1, gl.RGBA, gl.FLOAT, readBuf);
  let m = Infinity;
  for (let i = 0; i < NZ; i++) { const v = readBuf[i * 4]; if (Number.isFinite(v)) m = Math.min(m, v); }
  return m;
}

function show() {
  gl.useProgram(progShow);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, texA); gl.uniform1i(U.shState, 0);
  gl.uniform1f(U.shR0, params.r0); gl.uniform1i(U.shCmap, params.cmap);
  drawTri();
}

let running = true, last = 0, frames = 0, acc = 0, holdFrames = 0;
const fpsEl = document.getElementById("fps");

function frame(now) {
  if (running) {
    if (holdFrames > 0) {
      holdFrames--;                 // linger on the pinched droplets
      if (holdFrames === 0) seed(); // then reset the column
    } else {
      for (let i = 0; i < params.sub; i++) rk2();
      if (minRadius() < 0.13) holdFrames = 40; // pinched → hold, then reset
    }
  }
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
    const v = key === "sub" || key === "cmap" || key === "drops" ? parseInt(el.value, 10) : parseFloat(el.value);
    params[key] = v;
    if (out) out.textContent = fmt ? fmt(v) : v;
    if (after) after();
  };
  el.addEventListener("input", apply);
  apply();
};
bind("drops", "drops", (v) => v, seed);
bind("visc", "visc", (v) => v.toFixed(2));
bind("sub", "sub", (v) => v);
document.getElementById("cmap").addEventListener("change", (e) => { params.cmap = parseInt(e.target.value, 10); });

const btnPlay = document.getElementById("playPause");
btnPlay.addEventListener("click", () => {
  running = !running;
  btnPlay.textContent = running ? "⏸ Pause" : "▶ Play";
  btnPlay.classList.toggle("primary", running);
});
document.getElementById("reset").addEventListener("click", () => { holdFrames = 0; seed(); });
window.addEventListener("keydown", (e) => {
  if (e.code === "Space") { e.preventDefault(); btnPlay.click(); }
  if (e.key === "r" || e.key === "R") { holdFrames = 0; seed(); }
});

seed();
requestAnimationFrame(frame);

/* test hook for headless verification */
window.__rp = {
  run(steps) { const w = running; running = false; for (let i = 0; i < steps; i++) rk2(); running = w; },
  minRadius, maxRadius() { renderTo(texA); gl.readPixels(0,0,NZ,1,gl.RGBA,gl.FLOAT,readBuf); let m=-1e9; for(let i=0;i<NZ;i++){const v=readBuf[i*4]; if(Number.isFinite(v))m=Math.max(m,v);} return m; },
  volume() { renderTo(texA); gl.readPixels(0,0,NZ,1,gl.RGBA,gl.FLOAT,readBuf); let s=0,bad=0; for(let i=0;i<NZ;i++){const v=readBuf[i*4]; if(Number.isFinite(v))s+=v*v; else bad++;} return {vol:s, nonFinite:bad}; },
};
