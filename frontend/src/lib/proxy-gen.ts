// Client-side analysis-proxy generation (see client_proxy.plan.md).
//
// The desktop app decodes the local video ONCE (via mediabunny, WebCodecs under
// the hood) and emits two tiny MP4 proxies that upload in seconds while the raw
// uploads in the background -- decoupling analysis from the multi-GB raw:
//
//   Proxy A -- 480p @ 1fps + mono 16 kHz AAC. Feeds L2 (Gemini perception) and,
//              via a server-side WAV demux, the whole speech/audio L1 stack.
//              16 kHz mono matches exactly what the server would demux from the
//              raw, so ASR/audio quality is unchanged.
//   Proxy B -- 160x90 @ 10fps, video-only. Feeds motion_dynamics (optical flow).
//
// Everything here is BEST-EFFORT: if WebCodecs is unavailable, the input codec
// can't be decoded, or anything throws, we return null and the caller uploads
// the raw normally -- the server then regenerates every analysis input from the
// raw exactly as before. Proxies are an optimization, never a requirement.

export interface DualProxies {
  proxyA: Blob;
  proxyB: Blob;
}

/** Feature-detect the WebCodecs encode+decode path mediabunny relies on. */
export function canGenerateProxies(): boolean {
  return (
    typeof window !== "undefined" &&
    typeof (window as unknown as { VideoEncoder?: unknown }).VideoEncoder !== "undefined" &&
    typeof (window as unknown as { VideoDecoder?: unknown }).VideoDecoder !== "undefined"
  );
}

// mediabunny is dynamically imported so it stays out of the main bundle and
// never runs during SSR. Typed via the installed package.
type Mediabunny = typeof import("mediabunny");

async function runConversion(
  mb: Mediabunny,
  file: File,
  video: Parameters<Mediabunny["Conversion"]["init"]>[0]["video"],
  audio: Parameters<Mediabunny["Conversion"]["init"]>[0]["audio"],
): Promise<Blob | null> {
  const input = new mb.Input({ source: new mb.BlobSource(file), formats: mb.ALL_FORMATS });
  const output = new mb.Output({ format: new mb.Mp4OutputFormat(), target: new mb.BufferTarget() });
  const conversion = await mb.Conversion.init({ input, output, video, audio });
  if (!conversion.isValid) return null;
  await conversion.execute();
  const buffer = output.target.buffer;
  if (!buffer) return null;
  return new Blob([buffer], { type: "video/mp4" });
}

/**
 * Produce both analysis proxies from a single local video File, or null if the
 * environment/codec can't support it (caller falls back to raw-only upload).
 */
export async function generateProxies(file: File): Promise<DualProxies | null> {
  if (!canGenerateProxies()) return null;

  let mb: Mediabunny;
  try {
    mb = await import("mediabunny");
  } catch {
    return null;
  }

  // No decodable video track (audio-only container, image, unknown codec) ->
  // let the server handle it from the raw.
  try {
    const probe = new mb.Input({ source: new mb.BlobSource(file), formats: mb.ALL_FORMATS });
    if (!(await probe.getPrimaryVideoTrack())) return null;
  } catch {
    return null;
  }

  const proxyA = await runConversion(
    mb,
    file,
    { height: 480, frameRate: 1, codec: "avc", bitrate: 1_200_000 },
    { codec: "aac", numberOfChannels: 1, sampleRate: 16_000, bitrate: 96_000 },
  );
  if (!proxyA) return null;

  const proxyB = await runConversion(
    mb,
    file,
    { width: 160, height: 90, fit: "contain", frameRate: 10, codec: "avc", bitrate: 300_000 },
    { discard: true },
  );
  if (!proxyB) return null;

  return { proxyA, proxyB };
}
