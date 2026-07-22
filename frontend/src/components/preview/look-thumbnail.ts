/**
 * frontend_look_gallery.plan.md: live per-look thumbnails for the Look
 * gallery -- a bundled reference still (`public/look-thumb-ref.jpg`) put
 * through a look's baked cube, so the swatch shows the real color
 * transform instead of a text label.
 *
 * Deliberately NOT `lut-gl.ts`'s `createLutRenderer`: that renderer is
 * built around a `<video>` texture, cover/contain framing, and the
 * halation/grain FBO pipeline -- none of which a still-image color swatch
 * needs (thumbnails are COLOR ONLY; halation/grain are spatial and can't
 * bake into a cube -- see the film badge in color-grade-view.tsx for how
 * that gap is surfaced to the user, not hidden). This module is a much
 * smaller sibling: one shared offscreen WebGL2 context + a single-pass
 * "sample the reference image through a 3D LUT" program, reused for every
 * look and rendered to a `<canvas>` per call, then snapshotted to a data
 * URL and cached -- so 17 looks cost one shared context, one image
 * upload, and 17 cheap draws, never 17 WebGL contexts.
 *
 * Cube fetch reuses `grade-cube-client.ts::prefetchGradeCube` with a
 * SYNTHETIC `ResolvedGrade` (identity CDL + this look's `look_engine`
 * params) -- `gradeCubeUrl` already encodes `look_engine` into the cache
 * key/URL, so this rides the exact same `/api/grade/cube` endpoint and
 * in-memory cube cache the live preview uses, no new backend surface.
 */
import type { ResolvedGrade } from "@/lib/api";
import { prefetchGradeCube } from "./grade-cube-client";

const REF_IMAGE_SRC = "/look-thumb-ref.jpg";
const THUMB_W = 240;
const THUMB_H = 135;

const IDENTITY_CDL: Required<ResolvedGrade["cdl"]> = {
  slope: [1, 1, 1],
  offset: [0, 0, 0],
  power: [1, 1, 1],
  sat: 1,
};

const VERTEX_SRC = `#version 300 es
in vec2 aPos;
out vec2 vUv;
void main() {
  vUv = aPos * 0.5 + 0.5;
  gl_Position = vec4(aPos, 0.0, 1.0);
}
`;

// Single-pass: sample the reference still, then the 3D LUT at that color --
// no fit/focus/vignette/grain (a still swatch needs none of it).
const FRAGMENT_SRC = `#version 300 es
precision highp float;
precision highp sampler3D;
in vec2 vUv;
out vec4 outColor;
uniform sampler2D uImage;
uniform sampler3D uLut;
void main() {
  vec3 src = texture(uImage, vUv).rgb;
  vec3 graded = texture(uLut, clamp(src, 0.0, 1.0)).rgb;
  outColor = vec4(clamp(graded, 0.0, 1.0), 1.0);
}
`;

interface Renderer {
  gl: WebGL2RenderingContext;
  canvas: HTMLCanvasElement;
  program: WebGLProgram;
  uImage: WebGLUniformLocation | null;
  uLut: WebGLUniformLocation | null;
  imageTex: WebGLTexture;
  lutTex: WebGLTexture;
}

let rendererPromise: Promise<Renderer | null> | null = null;

function compileShader(gl: WebGL2RenderingContext, type: number, src: string): WebGLShader | null {
  const shader = gl.createShader(type);
  if (!shader) return null;
  gl.shaderSource(shader, src);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    gl.deleteShader(shader);
    return null;
  }
  return shader;
}

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`failed to load ${src}`));
    img.src = src;
  });
}

/** Lazily builds the ONE shared context/program/textures, loading the
 * reference still once. Resolves null (never throws) on any failure -- no
 * WebGL2, shader compile error, or image load error -- so callers fail
 * open into the flat-swatch fallback.
 *
 * grade_pipeline_standardize.plan.md Part B: a `null` outcome is NEVER
 * cached on `rendererPromise` -- only a successful `Renderer` is. If the
 * FIRST call ever fails (e.g. the reference still hadn't landed yet behind
 * a stale dev-server 404, or a transient WebGL context hiccup), earlier code
 * cached that `null` on the shared promise for the rest of the page session,
 * so every card fell into the flat-swatch fallback forever, even once the
 * underlying cause was gone -- a hard reload was the only fix. Clearing the
 * promise on failure lets the NEXT call retry from scratch instead. */
async function getRenderer(): Promise<Renderer | null> {
  if (!rendererPromise) {
    rendererPromise = buildRenderer();
  }
  const renderer = await rendererPromise;
  if (!renderer) rendererPromise = null;
  return renderer;
}

async function buildRenderer(): Promise<Renderer | null> {
  const canvas = document.createElement("canvas");
  canvas.width = THUMB_W;
  canvas.height = THUMB_H;
  const gl = canvas.getContext("webgl2", { premultipliedAlpha: false, antialias: false });
  if (!gl) return null;

  const vs = compileShader(gl, gl.VERTEX_SHADER, VERTEX_SRC);
  const fs = compileShader(gl, gl.FRAGMENT_SHADER, FRAGMENT_SRC);
  if (!vs || !fs) return null;
  const program = gl.createProgram();
  if (!program) return null;
  gl.attachShader(program, vs);
  gl.attachShader(program, fs);
  gl.linkProgram(program);
  gl.deleteShader(vs);
  gl.deleteShader(fs);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) return null;

  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);

  const quad = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, quad);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1]), gl.STATIC_DRAW);
  const aPos = gl.getAttribLocation(program, "aPos");
  gl.enableVertexAttribArray(aPos);
  gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);

  let image: HTMLImageElement;
  try {
    image = await loadImage(REF_IMAGE_SRC);
  } catch {
    return null;
  }

  const imageTex = gl.createTexture();
  if (!imageTex) return null;
  gl.bindTexture(gl.TEXTURE_2D, imageTex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  // Screen-space v=0 at top matches this image's natural (un-flipped)
  // orientation -- no video-style flip needed for a still <img>.
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, image);

  const lutTex = gl.createTexture();
  if (!lutTex) return null;
  gl.bindTexture(gl.TEXTURE_3D, lutTex);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);

  return {
    gl, canvas, program,
    uImage: gl.getUniformLocation(program, "uImage"),
    uLut: gl.getUniformLocation(program, "uLut"),
    imageTex, lutTex,
  };
}

function synthGrade(lookParams: Record<string, unknown>): ResolvedGrade {
  return {
    cdl: IDENTITY_CDL,
    working_space: "rec709_v1",
    creative_lut_ref: null,
    soft_local: null,
    tone_contrast: 0,
    look_engine: lookParams,
  };
}

const urlCache = new Map<string, string>();
const pending = new Map<string, Promise<string | null>>();

/** Synchronous cache read -- null if not (yet) rendered. */
export function getCachedLookThumbnail(lookId: string): string | null {
  return urlCache.get(lookId) ?? null;
}

/** Renders (or returns the cached) thumbnail for one engine look: fetches
 * its cube (identity CDL + `lookParams`, via the shared cube cache),
 * draws the reference still through it on the ONE shared context, and
 * caches the result as a data URL keyed by `lookId`. Never throws --
 * resolves `null` on any failure (no WebGL2, cube 404, decode error) so
 * callers render a flat-swatch fallback instead of a broken card. */
export function requestLookThumbnail(
  lookId: string,
  lookParams: Record<string, unknown> | undefined
): Promise<string | null> {
  const cached = urlCache.get(lookId);
  if (cached) return Promise.resolve(cached);
  const inFlight = pending.get(lookId);
  if (inFlight) return inFlight;
  if (!lookParams) return Promise.resolve(null);

  const p = (async () => {
    try {
      const renderer = await getRenderer();
      if (!renderer) return null;
      const cube = await prefetchGradeCube(synthGrade(lookParams));
      if (!cube) return null;

      const { gl, canvas, program, uImage, uLut, imageTex, lutTex } = renderer;
      const rgb8 = new Uint8Array(cube.grid.length);
      for (let i = 0; i < cube.grid.length; i++) {
        rgb8[i] = Math.max(0, Math.min(255, Math.round(cube.grid[i] * 255)));
      }
      gl.bindTexture(gl.TEXTURE_3D, lutTex);
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
      gl.texImage3D(gl.TEXTURE_3D, 0, gl.RGB8, cube.size, cube.size, cube.size, 0, gl.RGB, gl.UNSIGNED_BYTE, rgb8);

      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.useProgram(program);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, imageTex);
      gl.uniform1i(uImage, 0);
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_3D, lutTex);
      gl.uniform1i(uLut, 1);
      gl.drawArrays(gl.TRIANGLES, 0, 6);

      const url = canvas.toDataURL("image/png");
      urlCache.set(lookId, url);
      return url;
    } catch {
      return null;
    } finally {
      pending.delete(lookId);
    }
  })();
  pending.set(lookId, p);
  return p;
}
