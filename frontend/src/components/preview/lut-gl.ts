/**
 * WebGL2 3D-LUT preview shader — the frontend half of the Fork A parity
 * contract (color_grading.plan.md SS4): samples the SAME baked `.cube` the
 * backend bakes for export (`grade.lut_bake.bake_cube_text`, served by
 * `GET /api/grade/cube`) through a real trilinear-filtered `TEXTURE_3D`, so
 * the math is whatever the GPU's native 3D-texture sampling does — not a
 * second hand-rolled interpolation implementation that could drift from
 * ffmpeg's `lut3d` filter.
 *
 * Scope note: this renders the CDL/creative-LUT color transform, plus
 * optional soft-local spatial finishing (SS9 -- see backend
 * `grade/softlocal.py` for why these are approximate-parity effects by
 * design, unlike the LUT itself): a vignette, a halation glow
 * (halation_grain.plan.md), and film grain (same plan). The geometric
 * framing (object-fit cover/contain + focus point) is emulated in the
 * fragment shader via `uFit`/`uFocus` (mirroring `applyFrameStyle`'s CSS
 * object-fit/object-position math, since a canvas has no native
 * object-fit); rotate/zoom stay a CSS transform on the canvas element
 * itself, identical to how the plain `<video>` path already does it.
 * Preview of an animated `motion` (zoompan) clip renders its static
 * midpoint frame rather than animating — the same simplification
 * `applyFrameStyle` already makes for `object-position` on a motion clip,
 * just not (yet) animated in WebGL. None of this affects export, which
 * always uses the full transform.
 *
 * Halation (halation_grain.plan.md) is the one real structural addition:
 * the renderer was single-pass (video+LUT+vignette straight to the
 * screen). A soft glow needs a blur, which needs to read NEIGHBORING
 * pixels of the ALREADY-GRADED picture -- impossible in one pass. When a
 * clip's `soft_local.halation` is active, `draw()` instead runs a small
 * FBO pipeline: (1) the EXISTING single-pass program renders video+LUT+
 * vignette into an offscreen texture instead of the screen, (2) a second
 * program isolates+tints+horizontally-blurs the highlights into another
 * offscreen texture, (3) a third program blurs that vertically (the
 * separable-Gaussian split -- two 1D passes, not one expensive 2D pass),
 * (4) a fourth program screen-blends the glow back over the graded
 * picture and applies grain, into the visible canvas. When no clip needs
 * halation, `draw()` skips straight to the original single-pass path
 * (which also gained inline grain support) -- zero extra cost.
 */

const VERTEX_SRC = `#version 300 es
in vec2 aPos;
out vec2 vUv;
void main() {
  vUv = aPos * 0.5 + 0.5;
  gl_Position = vec4(aPos, 0.0, 1.0);
}
`;

// halation_grain.plan.md: mirrors the fixed constants in backend
// grade/softlocal.py (HALATION_THRESHOLD/_SIGMA/_SIGMA_REF_H/_TINT) --
// approximate parity by design (independent implementations reading as
// the same effect), not required to be byte-identical, but kept at the
// same tuned values so preview and export don't visibly disagree.
const HALATION_THRESHOLD = 0.75;
const HALATION_SIGMA = 8.0;
const HALATION_SIGMA_REF_H = 1080;
const HALATION_TINT: [number, number, number] = [1.0, 0.35, 0.1];
const BLUR_TAPS = 12; // a (2*12+1)=25-tap 1D kernel per pass; two passes (H then V)

// uFit: 0 = cover (crops the SOURCE to fill the canvas), 1 = contain (pads
// the CANVAS with black to keep the whole source visible). uFocus:
// normalized (cx, cy) bias for where the crop/pad window sits -- mirrors
// applyFrameStyle's objectPosition (anchor point or motion-midpoint focus).
//
// uGrainStrength/uFrameSeed (halation_grain.plan.md): luma-ish hash noise,
// TEMPORAL (a new pseudo-random field each `uFrameSeed`, like ffmpeg's
// `noise=allf=t`) -- applied LAST, after everything else, matching the
// plan's "grain is the final texture on top of everything" ordering. This
// program is reused for the no-halation single-pass path (grain applied
// here, straight to the screen) AND as halation's pass 1 (grain forced to
// 0 there -- see the composite program, which applies grain instead so it
// lands after the glow, not baked into what feeds the blur).
const FRAGMENT_SRC = `#version 300 es
precision highp float;
precision highp sampler3D;
in vec2 vUv;
out vec4 outColor;
uniform sampler2D uVideo;
uniform sampler3D uLut;
uniform vec2 uVideoSize;
uniform vec2 uCanvasSize;
uniform vec2 uFocus;
uniform int uFit;
uniform vec2 uVignetteCenter;
uniform float uVignetteStrength;
uniform float uGrainStrength;
uniform float uFrameSeed;

void main() {
  float videoAspect = uVideoSize.x / max(uVideoSize.y, 1.0);
  float canvasAspect = uCanvasSize.x / max(uCanvasSize.y, 1.0);
  bool canvasWiderThanVideo = canvasAspect > videoAspect;

  vec2 uv;
  bool valid = true;
  if (uFit == 0) {
    // cover: crop the SOURCE. The un-cropped axis is whichever one canvas
    // and video already agree on the direction of; canvas UV maps 1:1 onto
    // that axis and the other axis is scaled + focus-biased into source UV.
    vec2 cropFrac = canvasWiderThanVideo
      ? vec2(1.0, videoAspect / canvasAspect)
      : vec2(canvasAspect / videoAspect, 1.0);
    vec2 origin = (vec2(1.0) - cropFrac) * uFocus;
    uv = origin + vUv * cropFrac;
  } else {
    // contain: pad the CANVAS. The video occupies a centered/focus-biased
    // sub-rect sized by the SMALLER scale factor; outside it is black.
    vec2 renderFrac = canvasWiderThanVideo
      ? vec2(videoAspect / canvasAspect, 1.0)
      : vec2(1.0, canvasAspect / videoAspect);
    vec2 origin = (vec2(1.0) - renderFrac) * uFocus;
    vec2 rel = vUv - origin;
    valid = all(greaterThanEqual(rel, vec2(0.0))) && all(lessThanEqual(rel, renderFrac));
    uv = rel / max(renderFrac, vec2(1e-6));
  }

  if (!valid) {
    outColor = vec4(0.0, 0.0, 0.0, 1.0);
    return;
  }
  vec4 src = texture(uVideo, clamp(uv, 0.0, 1.0));
  vec3 graded = texture(uLut, clamp(src.rgb, 0.0, 1.0)).rgb;

  // Soft-local vignette (SS9): a feathered radial darkening in FRAME space
  // (vUv, not source uv -- a vignette is a property of the delivered
  // picture, not the source crop), aspect-corrected so the falloff is
  // circular rather than stretched to the canvas's aspect ratio.
  if (uVignetteStrength > 0.001) {
    float canvasAspect = uCanvasSize.x / max(uCanvasSize.y, 1.0);
    vec2 d = vUv - uVignetteCenter;
    d.x *= canvasAspect;
    float dist = length(d);
    float falloff = 1.0 - uVignetteStrength * pow(clamp((dist - 0.3) / 0.6, 0.0, 1.0), 2.0);
    graded *= falloff;
  }

  if (uGrainStrength > 0.001) {
    float n = fract(sin(dot(vUv * uCanvasSize + uFrameSeed, vec2(12.9898, 78.233))) * 43758.5453);
    graded += (n - 0.5) * uGrainStrength;
  }

  outColor = vec4(clamp(graded, 0.0, 1.0), 1.0);
}
`;

// Halation pass 2/3 (halation_grain.plan.md): isolate highlights above
// uThreshold, tint red-orange, separable 1D Gaussian blur (this shader
// does ONE axis; uDirection picks which -- called twice, H then V, via
// two different FBOs). Tint commutes with a linear blur (both are just
// per-channel scale/weighted-sum), so it's applied once, in the H pass,
// rather than per-tap.
const EXTRACT_BLUR_FRAGMENT_SRC = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 outColor;
uniform sampler2D uSource;
uniform vec2 uTexelSize;
uniform vec2 uDirection;   // (1,0) horizontal pass, (0,1) vertical pass
uniform float uSigma;
uniform float uThreshold;  // <0 skips thresholding/tinting (the V pass -- already isolated by the H pass)
uniform vec3 uTint;

float gauss(float x, float sigma) {
  return exp(-0.5 * (x * x) / max(sigma * sigma, 1e-6));
}

void main() {
  vec3 sum = vec3(0.0);
  float wsum = 0.0;
  for (int i = -${BLUR_TAPS}; i <= ${BLUR_TAPS}; i++) {
    float off = float(i);
    vec2 uv = clamp(vUv + uDirection * uTexelSize * off, 0.0, 1.0);
    vec3 c = texture(uSource, uv).rgb;
    if (uThreshold >= 0.0) {
      float luma = dot(c, vec3(0.2126, 0.7152, 0.0722));
      c = luma > uThreshold ? c : vec3(0.0);
    }
    float w = gauss(off, uSigma);
    sum += c * w;
    wsum += w;
  }
  vec3 blurred = sum / max(wsum, 1e-6);
  if (uThreshold >= 0.0) {
    blurred *= uTint;
  }
  outColor = vec4(blurred, 1.0);
}
`;

// Halation pass 4 (composite): screen-blend the blurred glow back over the
// graded picture (LERPed by strength, matching ffmpeg's
// `blend=all_mode=screen:all_opacity=STRENGTH`), then grain LAST (film
// grain is in the emulsion -- the final texture on top of everything,
// including halation).
const COMPOSITE_FRAGMENT_SRC = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 outColor;
uniform sampler2D uGraded;
uniform sampler2D uGlow;
uniform float uHalationStrength;
uniform float uGrainStrength;
uniform float uFrameSeed;
uniform vec2 uCanvasSize;

void main() {
  vec3 base = texture(uGraded, vUv).rgb;
  vec3 glow = texture(uGlow, vUv).rgb;
  vec3 screened = vec3(1.0) - (vec3(1.0) - base) * (vec3(1.0) - glow);
  vec3 result = mix(base, screened, clamp(uHalationStrength, 0.0, 1.0));

  if (uGrainStrength > 0.001) {
    float n = fract(sin(dot(vUv * uCanvasSize + uFrameSeed, vec2(12.9898, 78.233))) * 43758.5453);
    result += (n - 0.5) * uGrainStrength;
  }

  outColor = vec4(clamp(result, 0.0, 1.0), 1.0);
}
`;

function compileShader(gl: WebGL2RenderingContext, type: number, src: string): WebGLShader {
  const shader = gl.createShader(type)!;
  gl.shaderSource(shader, src);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(`LUT shader compile failed: ${log}`);
  }
  return shader;
}

function linkProgram(gl: WebGL2RenderingContext, vs: WebGLShader, fragSrc: string): WebGLProgram {
  const fs = compileShader(gl, gl.FRAGMENT_SHADER, fragSrc);
  const program = gl.createProgram()!;
  gl.attachShader(program, vs);
  gl.attachShader(program, fs);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(`LUT program link failed: ${gl.getProgramInfoLog(program)}`);
  }
  gl.deleteShader(fs);
  return program;
}

export interface VignetteParams {
  cx: number;
  cy: number;
  strength: number;
}

/** halation_grain.plan.md: the full soft-local param set `draw()` accepts,
 * mirroring backend `resolve_clip_grade`'s `soft_local` descriptor shape. */
export interface SoftLocalParams {
  vignette?: VignetteParams | null;
  halation?: { strength: number } | null;
  grain?: { strength: number } | null;
}

interface FboTarget {
  fbo: WebGLFramebuffer;
  tex: WebGLTexture;
  width: number;
  height: number;
}

export interface LutRenderer {
  /** Upload a new baked `.cube`'s grid (parsed by `parseCubeText`). No-op
   * (skips re-upload) if the same `cacheKey` is already loaded. */
  setLut: (grid: Float32Array, size: number, cacheKey: string) => void;
  /** Draw one frame: sample `videoEl`'s current picture through the loaded
   * LUT, emulating `fit`/`focus` framing, into the renderer's canvas.
   * `softLocal` (SS9) is optional -- omitted/null draws with no vignette/
   * halation/grain, the identical single-pass path as before this plan. */
  draw: (
    videoEl: HTMLVideoElement,
    fit: "cover" | "contain",
    focus: { cx: number; cy: number },
    softLocal?: SoftLocalParams | null
  ) => void;
  dispose: () => void;
  readonly canvas: HTMLCanvasElement;
}

/** Parse `.cube` text (the exact format `grade.lut_bake.bake_cube_text`
 * emits) into a flat RGB grid + size, ready for `gl.texImage3D`. Row order
 * is R-fastest -- matches `LutRenderer.setLut`'s expected layout. */
export function parseCubeText(text: string): { grid: Float32Array; size: number } | null {
  let size: number | null = null;
  const values: number[] = [];
  for (const raw of text.split("\n")) {
    const line = raw.trim();
    if (!line || line.startsWith("#")) continue;
    const upper = line.toUpperCase();
    if (upper.startsWith("LUT_3D_SIZE")) {
      size = parseInt(line.split(/\s+/).pop() ?? "", 10);
      continue;
    }
    if (
      upper.startsWith("TITLE") ||
      upper.startsWith("DOMAIN_MIN") ||
      upper.startsWith("DOMAIN_MAX") ||
      upper.startsWith("LUT_1D_SIZE")
    ) {
      continue;
    }
    const parts = line.split(/\s+/);
    if (parts.length !== 3) continue;
    const r = Number(parts[0]);
    const g = Number(parts[1]);
    const b = Number(parts[2]);
    if (Number.isNaN(r) || Number.isNaN(g) || Number.isNaN(b)) continue;
    values.push(r, g, b);
  }
  if (!size || values.length !== size * size * size * 3) return null;
  return { grid: new Float32Array(values), size };
}

export function createLutRenderer(): LutRenderer | null {
  const canvas = document.createElement("canvas");
  const gl = canvas.getContext("webgl2", { premultipliedAlpha: false, antialias: false });
  if (!gl) return null;

  // RGB (3 bytes/pixel) rows aren't 4-byte aligned at most sizes (e.g. a
  // 33-wide LUT row is 99 bytes); WebGL's default UNPACK_ALIGNMENT=4 makes
  // texImage2D/3D reject an unpadded buffer as "not big enough". Every
  // upload in this module is RGB, so set this once, globally, for the
  // lifetime of this context.
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);

  const vs = compileShader(gl, gl.VERTEX_SHADER, VERTEX_SRC);
  const mainProgram = linkProgram(gl, vs, FRAGMENT_SRC);
  const extractBlurProgram = linkProgram(gl, vs, EXTRACT_BLUR_FRAGMENT_SRC);
  const compositeProgram = linkProgram(gl, vs, COMPOSITE_FRAGMENT_SRC);
  gl.deleteShader(vs);

  const quad = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, quad);
  gl.bufferData(
    gl.ARRAY_BUFFER,
    new Float32Array([-1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1]),
    gl.STATIC_DRAW
  );

  function bindQuad(program: WebGLProgram) {
    const aPos = gl!.getAttribLocation(program, "aPos");
    gl!.bindBuffer(gl!.ARRAY_BUFFER, quad);
    gl!.enableVertexAttribArray(aPos);
    gl!.vertexAttribPointer(aPos, 2, gl!.FLOAT, false, 0, 0);
  }

  const videoTex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, videoTex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

  const lutTex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_3D, lutTex);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);

  // halation_grain.plan.md: the 3 offscreen render targets the halation
  // pipeline ping-pongs through -- graded picture, horizontally-blurred
  // glow, fully-blurred glow. Created/resized lazily (only when halation
  // is actually active for some clip), so a project with no film-texture
  // look never allocates them.
  let gradedFbo: FboTarget | null = null;
  let glowHFbo: FboTarget | null = null;
  let glowFbo: FboTarget | null = null;

  function makeFbo(w: number, h: number): FboTarget {
    const tex = gl!.createTexture()!;
    gl!.bindTexture(gl!.TEXTURE_2D, tex);
    gl!.texParameteri(gl!.TEXTURE_2D, gl!.TEXTURE_MIN_FILTER, gl!.LINEAR);
    gl!.texParameteri(gl!.TEXTURE_2D, gl!.TEXTURE_MAG_FILTER, gl!.LINEAR);
    gl!.texParameteri(gl!.TEXTURE_2D, gl!.TEXTURE_WRAP_S, gl!.CLAMP_TO_EDGE);
    gl!.texParameteri(gl!.TEXTURE_2D, gl!.TEXTURE_WRAP_T, gl!.CLAMP_TO_EDGE);
    gl!.texImage2D(gl!.TEXTURE_2D, 0, gl!.RGBA8, w, h, 0, gl!.RGBA, gl!.UNSIGNED_BYTE, null);
    const fbo = gl!.createFramebuffer()!;
    gl!.bindFramebuffer(gl!.FRAMEBUFFER, fbo);
    gl!.framebufferTexture2D(gl!.FRAMEBUFFER, gl!.COLOR_ATTACHMENT0, gl!.TEXTURE_2D, tex, 0);
    gl!.bindFramebuffer(gl!.FRAMEBUFFER, null);
    return { fbo, tex, width: w, height: h };
  }

  function ensureFbo(target: FboTarget | null, w: number, h: number): FboTarget {
    if (target && target.width === w && target.height === h) return target;
    if (target) {
      gl!.deleteFramebuffer(target.fbo);
      gl!.deleteTexture(target.tex);
    }
    return makeFbo(w, h);
  }

  const uVideo = gl.getUniformLocation(mainProgram, "uVideo");
  const uLut = gl.getUniformLocation(mainProgram, "uLut");
  const uVideoSize = gl.getUniformLocation(mainProgram, "uVideoSize");
  const uCanvasSize = gl.getUniformLocation(mainProgram, "uCanvasSize");
  const uFocus = gl.getUniformLocation(mainProgram, "uFocus");
  const uFit = gl.getUniformLocation(mainProgram, "uFit");
  const uVignetteCenter = gl.getUniformLocation(mainProgram, "uVignetteCenter");
  const uVignetteStrength = gl.getUniformLocation(mainProgram, "uVignetteStrength");
  const uGrainStrength = gl.getUniformLocation(mainProgram, "uGrainStrength");
  const uFrameSeed = gl.getUniformLocation(mainProgram, "uFrameSeed");

  const ebSource = gl.getUniformLocation(extractBlurProgram, "uSource");
  const ebTexelSize = gl.getUniformLocation(extractBlurProgram, "uTexelSize");
  const ebDirection = gl.getUniformLocation(extractBlurProgram, "uDirection");
  const ebSigma = gl.getUniformLocation(extractBlurProgram, "uSigma");
  const ebThreshold = gl.getUniformLocation(extractBlurProgram, "uThreshold");
  const ebTint = gl.getUniformLocation(extractBlurProgram, "uTint");

  const cGraded = gl.getUniformLocation(compositeProgram, "uGraded");
  const cGlow = gl.getUniformLocation(compositeProgram, "uGlow");
  const cHalationStrength = gl.getUniformLocation(compositeProgram, "uHalationStrength");
  const cGrainStrength = gl.getUniformLocation(compositeProgram, "uGrainStrength");
  const cFrameSeed = gl.getUniformLocation(compositeProgram, "uFrameSeed");
  const cCanvasSize = gl.getUniformLocation(compositeProgram, "uCanvasSize");

  let loadedKey: string | null = null;
  let lutReady = false;
  let frameCounter = 0;

  const setLut: LutRenderer["setLut"] = (grid, size, cacheKey) => {
    if (loadedKey === cacheKey) return;
    // RGB float grid -> RGB8 (matches the render side's 8-bit-parity choice
    // documented in compositor.py's _transform_vf).
    const rgb8 = new Uint8Array(grid.length);
    for (let i = 0; i < grid.length; i++) {
      rgb8[i] = Math.max(0, Math.min(255, Math.round(grid[i] * 255)));
    }
    gl.bindTexture(gl.TEXTURE_3D, lutTex);
    // The LUT grid must NOT be Y-flipped (unlike the video texture in draw()):
    // its axes are color channels, not screen space. draw() sets FLIP_Y=true
    // for the video, so reset it here or the LUT's green axis would invert.
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.texImage3D(gl.TEXTURE_3D, 0, gl.RGB8, size, size, size, 0, gl.RGB, gl.UNSIGNED_BYTE, rgb8);
    loadedKey = cacheKey;
    lutReady = true;
  };

  /** The shared "grade" draw: video -> LUT -> vignette (+ grain, when
   * `grainStrength>0`) into whatever framebuffer is currently bound
   * (screen, or `gradedFbo` for the halation pipeline's first pass). */
  function drawGraded(
    videoEl: HTMLVideoElement, fit: "cover" | "contain", focus: { cx: number; cy: number },
    vignette: VignetteParams | null | undefined, grainStrength: number, w: number, h: number,
  ) {
    gl!.useProgram(mainProgram);
    bindQuad(mainProgram);

    gl!.activeTexture(gl!.TEXTURE0);
    gl!.bindTexture(gl!.TEXTURE_2D, videoTex);
    // A video's first row is its TOP, but the shader maps screen-bottom to
    // texture v=0, so upload flipped to match the plain <video> orientation
    // (otherwise the graded canvas renders upside down). setLut() resets this
    // to false for the LUT upload.
    gl!.pixelStorei(gl!.UNPACK_FLIP_Y_WEBGL, true);
    gl!.texImage2D(gl!.TEXTURE_2D, 0, gl!.RGB, gl!.RGB, gl!.UNSIGNED_BYTE, videoEl);
    gl!.uniform1i(uVideo, 0);

    gl!.activeTexture(gl!.TEXTURE1);
    gl!.bindTexture(gl!.TEXTURE_3D, lutTex);
    gl!.uniform1i(uLut, 1);

    gl!.uniform2f(uVideoSize, videoEl.videoWidth || w, videoEl.videoHeight || h);
    gl!.uniform2f(uCanvasSize, w, h);
    gl!.uniform2f(uFocus, focus.cx, focus.cy);
    gl!.uniform1i(uFit, fit === "contain" ? 1 : 0);
    gl!.uniform2f(uVignetteCenter, vignette?.cx ?? 0.5, vignette?.cy ?? 0.5);
    gl!.uniform1f(uVignetteStrength, vignette?.strength ?? 0);
    gl!.uniform1f(uGrainStrength, grainStrength);
    gl!.uniform1f(uFrameSeed, frameCounter);

    gl!.drawArrays(gl!.TRIANGLES, 0, 6);
  }

  const draw: LutRenderer["draw"] = (videoEl, fit, focus, softLocal) => {
    if (!lutReady) return;
    const w = videoEl.clientWidth || videoEl.videoWidth || 1;
    const h = videoEl.clientHeight || videoEl.videoHeight || 1;
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
    frameCounter++;

    const vignette = softLocal?.vignette ?? null;
    const halationStrength = softLocal?.halation?.strength ?? 0;
    const grainStrength = softLocal?.grain?.strength ?? 0;

    if (halationStrength <= 0.001) {
      // No halation: identical single-pass path to before this plan (grain
      // folded into the same draw, zero extra cost when it's also 0).
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, canvas.width, canvas.height);
      drawGraded(videoEl, fit, focus, vignette, grainStrength, canvas.width, canvas.height);
      return;
    }

    // Halation pipeline: grade -> FBO, extract+tint+blurH -> FBO,
    // blurV -> FBO, composite (screen-blend + grain) -> screen.
    gradedFbo = ensureFbo(gradedFbo, canvas.width, canvas.height);
    glowHFbo = ensureFbo(glowHFbo, canvas.width, canvas.height);
    glowFbo = ensureFbo(glowFbo, canvas.width, canvas.height);

    gl.bindFramebuffer(gl.FRAMEBUFFER, gradedFbo.fbo);
    gl.viewport(0, 0, canvas.width, canvas.height);
    drawGraded(videoEl, fit, focus, vignette, 0, canvas.width, canvas.height);

    const sigma = Math.max(1.0, HALATION_SIGMA * (canvas.height / HALATION_SIGMA_REF_H));
    const texel: [number, number] = [1.0 / canvas.width, 1.0 / canvas.height];

    // Pass 2: horizontal extract + tint + blur (gradedFbo -> glowHFbo).
    gl.bindFramebuffer(gl.FRAMEBUFFER, glowHFbo.fbo);
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.useProgram(extractBlurProgram);
    bindQuad(extractBlurProgram);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, gradedFbo.tex);
    gl.uniform1i(ebSource, 0);
    gl.uniform2f(ebTexelSize, texel[0], texel[1]);
    gl.uniform2f(ebDirection, 1, 0);
    gl.uniform1f(ebSigma, sigma);
    gl.uniform1f(ebThreshold, HALATION_THRESHOLD);
    gl.uniform3f(ebTint, HALATION_TINT[0], HALATION_TINT[1], HALATION_TINT[2]);
    gl.drawArrays(gl.TRIANGLES, 0, 6);

    // Pass 3: vertical blur (glowHFbo -> glowFbo), no re-thresholding/tint
    // (uThreshold<0 skips it -- already isolated+tinted by pass 2).
    gl.bindFramebuffer(gl.FRAMEBUFFER, glowFbo.fbo);
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.useProgram(extractBlurProgram);
    bindQuad(extractBlurProgram);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, glowHFbo.tex);
    gl.uniform1i(ebSource, 0);
    gl.uniform2f(ebTexelSize, texel[0], texel[1]);
    gl.uniform2f(ebDirection, 0, 1);
    gl.uniform1f(ebSigma, sigma);
    gl.uniform1f(ebThreshold, -1.0);
    gl.drawArrays(gl.TRIANGLES, 0, 6);

    // Pass 4: composite -- screen-blend the glow over the graded picture,
    // then grain, into the visible canvas.
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.useProgram(compositeProgram);
    bindQuad(compositeProgram);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, gradedFbo.tex);
    gl.uniform1i(cGraded, 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, glowFbo.tex);
    gl.uniform1i(cGlow, 1);
    gl.uniform1f(cHalationStrength, halationStrength);
    gl.uniform1f(cGrainStrength, grainStrength);
    gl.uniform1f(cFrameSeed, frameCounter);
    gl.uniform2f(cCanvasSize, canvas.width, canvas.height);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
  };

  const dispose = () => {
    gl.deleteTexture(videoTex);
    gl.deleteTexture(lutTex);
    gl.deleteBuffer(quad);
    gl.deleteProgram(mainProgram);
    gl.deleteProgram(extractBlurProgram);
    gl.deleteProgram(compositeProgram);
    for (const target of [gradedFbo, glowHFbo, glowFbo]) {
      if (!target) continue;
      gl.deleteFramebuffer(target.fbo);
      gl.deleteTexture(target.tex);
    }
  };

  return { setLut, draw, dispose, canvas };
}
