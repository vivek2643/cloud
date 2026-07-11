/**
 * WebGL2 3D-LUT preview shader — the frontend half of the Fork A parity
 * contract (color_grading.plan.md SS4): samples the SAME baked `.cube` the
 * backend bakes for export (`grade.lut_bake.bake_cube_text`, served by
 * `GET /api/grade/cube`) through a real trilinear-filtered `TEXTURE_3D`, so
 * the math is whatever the GPU's native 3D-texture sampling does — not a
 * second hand-rolled interpolation implementation that could drift from
 * ffmpeg's `lut3d` filter.
 *
 * Scope note: this renders the CDL/creative-LUT color transform, plus an
 * optional soft-local vignette (SS9 -- see backend `grade/softlocal.py` for
 * why this is an approximate-parity effect by design, unlike the LUT
 * itself). The geometric framing (object-fit cover/contain + focus point)
 * is emulated in the fragment shader via `uFit`/`uFocus` (mirroring
 * `applyFrameStyle`'s CSS object-fit/object-position math, since a canvas
 * has no native object-fit); rotate/zoom stay a CSS transform on the canvas
 * element itself, identical to how the plain `<video>` path already does
 * it. Preview of an animated `motion` (zoompan) clip renders its static
 * midpoint frame rather than animating — the same simplification
 * `applyFrameStyle` already makes for `object-position` on a motion clip,
 * just not (yet) animated in WebGL. None of this affects export, which
 * always uses the full transform.
 */

const VERTEX_SRC = `#version 300 es
in vec2 aPos;
out vec2 vUv;
void main() {
  vUv = aPos * 0.5 + 0.5;
  gl_Position = vec4(aPos, 0.0, 1.0);
}
`;

// uFit: 0 = cover (crops the SOURCE to fill the canvas), 1 = contain (pads
// the CANVAS with black to keep the whole source visible). uFocus:
// normalized (cx, cy) bias for where the crop/pad window sits -- mirrors
// applyFrameStyle's objectPosition (anchor point or motion-midpoint focus).
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

  outColor = vec4(graded, 1.0);
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

export interface VignetteParams {
  cx: number;
  cy: number;
  strength: number;
}

export interface LutRenderer {
  /** Upload a new baked `.cube`'s grid (parsed by `parseCubeText`). No-op
   * (skips re-upload) if the same `cacheKey` is already loaded. */
  setLut: (grid: Float32Array, size: number, cacheKey: string) => void;
  /** Draw one frame: sample `videoEl`'s current picture through the loaded
   * LUT, emulating `fit`/`focus` framing, into the renderer's canvas.
   * `vignette` (SS9) is optional -- omitted/null draws with no vignette. */
  draw: (
    videoEl: HTMLVideoElement,
    fit: "cover" | "contain",
    focus: { cx: number; cy: number },
    vignette?: VignetteParams | null
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
  const fs = compileShader(gl, gl.FRAGMENT_SHADER, FRAGMENT_SRC);
  const program = gl.createProgram()!;
  gl.attachShader(program, vs);
  gl.attachShader(program, fs);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(`LUT program link failed: ${gl.getProgramInfoLog(program)}`);
  }
  gl.deleteShader(vs);
  gl.deleteShader(fs);

  const quad = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, quad);
  gl.bufferData(
    gl.ARRAY_BUFFER,
    new Float32Array([-1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1]),
    gl.STATIC_DRAW
  );
  const aPos = gl.getAttribLocation(program, "aPos");

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

  const uVideo = gl.getUniformLocation(program, "uVideo");
  const uLut = gl.getUniformLocation(program, "uLut");
  const uVideoSize = gl.getUniformLocation(program, "uVideoSize");
  const uCanvasSize = gl.getUniformLocation(program, "uCanvasSize");
  const uFocus = gl.getUniformLocation(program, "uFocus");
  const uFit = gl.getUniformLocation(program, "uFit");
  const uVignetteCenter = gl.getUniformLocation(program, "uVignetteCenter");
  const uVignetteStrength = gl.getUniformLocation(program, "uVignetteStrength");

  let loadedKey: string | null = null;
  let lutReady = false;

  const setLut: LutRenderer["setLut"] = (grid, size, cacheKey) => {
    if (loadedKey === cacheKey) return;
    // RGB float grid -> RGB8 (matches the render side's 8-bit-parity choice
    // documented in compositor.py's _transform_vf).
    const rgb8 = new Uint8Array(grid.length);
    for (let i = 0; i < grid.length; i++) {
      rgb8[i] = Math.max(0, Math.min(255, Math.round(grid[i] * 255)));
    }
    gl.bindTexture(gl.TEXTURE_3D, lutTex);
    gl.texImage3D(gl.TEXTURE_3D, 0, gl.RGB8, size, size, size, 0, gl.RGB, gl.UNSIGNED_BYTE, rgb8);
    loadedKey = cacheKey;
    lutReady = true;
  };

  const draw: LutRenderer["draw"] = (videoEl, fit, focus, vignette) => {
    if (!lutReady) return;
    const w = videoEl.clientWidth || videoEl.videoWidth || 1;
    const h = videoEl.clientHeight || videoEl.videoHeight || 1;
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.useProgram(program);

    gl.bindBuffer(gl.ARRAY_BUFFER, quad);
    gl.enableVertexAttribArray(aPos);
    gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, videoTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, videoEl);
    gl.uniform1i(uVideo, 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_3D, lutTex);
    gl.uniform1i(uLut, 1);

    gl.uniform2f(uVideoSize, videoEl.videoWidth || w, videoEl.videoHeight || h);
    gl.uniform2f(uCanvasSize, canvas.width, canvas.height);
    gl.uniform2f(uFocus, focus.cx, focus.cy);
    gl.uniform1i(uFit, fit === "contain" ? 1 : 0);
    gl.uniform2f(uVignetteCenter, vignette?.cx ?? 0.5, vignette?.cy ?? 0.5);
    gl.uniform1f(uVignetteStrength, vignette?.strength ?? 0);

    gl.drawArrays(gl.TRIANGLES, 0, 6);
  };

  const dispose = () => {
    gl.deleteTexture(videoTex);
    gl.deleteTexture(lutTex);
    gl.deleteBuffer(quad);
    gl.deleteProgram(program);
  };

  return { setLut, draw, dispose, canvas };
}
