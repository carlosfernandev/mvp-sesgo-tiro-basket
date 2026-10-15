// Central API configuration and fetch helpers.
//
// WHY every fetch goes through this module instead of calling fetch inline:
// the API lives on the same origin in both dev (Vite proxies `/api/*` to
// Flask on port 5097) and production (reverse proxy), so components must
// never embed a hostname or port. If the API moves, only API_BASE changes.

export const API_BASE = "";

async function fetchJson<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) {
    throw new Error(`Error ${res.status} al solicitar ${path}`);
  }
  return (await res.json()) as T;
}

import type {
  AnalyzeResult,
  SessionDetail,
  SessionSummary,
  UploadOptions,
} from "./types";

export function fetchSessions(): Promise<SessionSummary[]> {
  return fetchJson<SessionSummary[]>("/api/sesiones");
}

export function fetchSessionDetail(id: string): Promise<SessionDetail> {
  return fetchJson<SessionDetail>(
    `/api/sesiones/${encodeURIComponent(id)}`
  );
}

// WHY a plain path builder (not fetch) for media: the <video> element and
// the profile <img> stream bytes themselves, so they need a URL string, not
// parsed JSON. Keeping the shape here guarantees they stay same-origin too.
export function videoUrl(id: string): string {
  return `${API_BASE}/api/sesiones/${encodeURIComponent(id)}/video`;
}

export function perfilUrl(id: string): string {
  return `${API_BASE}/api/sesiones/${encodeURIComponent(id)}/perfil`;
}

// Client-side mirror of the server upload cap (packages/api/app.py,
// MAX_UPLOAD_BYTES): rejecting an oversized file here saves the user from
// uploading 200 MB just to receive the 413 the server would send anyway.
export const MAX_UPLOAD_BYTES = 200 * 1024 * 1024;

export function formatMegabytes(bytes: number): string {
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export interface UploadProgress {
  loaded: number;
  total: number | null;
  // WHY two stages instead of one bar: the browser reports bytes SENT, but
  // the slow part is server-side (MediaPipe runs per frame, tens of
  // seconds) with no progress channel, so once the bytes are out the UI
  // must switch to an honest indeterminate "processing" state rather than
  // freezing a full bar and looking hung.
  stage: "uploading" | "processing";
}

// Error carrying the HTTP status so the panel can explain each documented
// failure (400 undecodable, 422 no frames / no person, 413 too large) with
// the server's own message instead of a generic fallback.
export class UploadError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = "UploadError";
    this.status = status;
  }
}

function uploadErrorMessage(status: number, serverMessage: string): string {
  const fallback =
    status === 413
      ? "El archivo supera el límite de 200 MB."
      : status === 422
        ? "El vídeo no contiene un tiro analizable."
        : status === 400
          ? "El archivo no es un vídeo MP4 válido."
          : `La API respondió con el estado ${status}.`;
  const detail = serverMessage !== "" ? serverMessage : fallback;
  return `No se pudo analizar el vídeo (error ${status}): ${detail}`;
}

// Upload an MP4 for analysis. Resolves with the POST /api/analizar JSON;
// rejects with UploadError carrying the server's message on HTTP errors.
export function uploadVideo(
  file: File,
  onProgress?: (p: UploadProgress) => void
): Promise<AnalyzeResult>;
export function uploadVideo(
  file: File,
  options?: UploadOptions,
  onProgress?: (p: UploadProgress) => void
): Promise<AnalyzeResult>;
export function uploadVideo(
  file: File,
  optionsOrProgress?: UploadOptions | ((p: UploadProgress) => void),
  onProgress?: (p: UploadProgress) => void
): Promise<AnalyzeResult> {
  let opts: UploadOptions = {};
  let progressCb: ((p: UploadProgress) => void) | undefined = onProgress;

  if (typeof optionsOrProgress === "function") {
    progressCb = optionsOrProgress;
    opts = {};
  } else if (optionsOrProgress) {
    opts = optionsOrProgress;
  }

  const notify = (p: UploadProgress) => {
    if (progressCb) {
      progressCb(p);
    }
  };

  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();

    const params = new URLSearchParams();
    if (opts.side) {
      params.set("side", opts.side);
    }
    if (opts.feedback !== undefined) {
      params.set("feedback", opts.feedback ? "true" : "false");
    }
    const query = params.toString();
    const url = query ? `${API_BASE}/api/analizar?${query}` : `${API_BASE}/api/analizar`;

    xhr.open("POST", url);
    // WHY the raw File as the body with an explicit Content-Type instead of
    // FormData multipart: the endpoint accepts both, but a raw body avoids
    // multipart framing overhead on payloads near the 200 MB cap and is the
    // documented recommended path. The type is set literally because
    // file.type is empty on some systems for .mp4 files.
    xhr.setRequestHeader("Content-Type", "video/mp4");
    xhr.responseType = "json";
    // WHY no timeout is set: real runs take tens of seconds server-side, so
    // any fixed timeout risks killing a healthy analysis; the user cancels
    // by choosing another file or leaving the panel.
    //
    // WHY XMLHttpRequest instead of fetch: fetch exposes no upload-progress
    // events for the request body, so a fetch-based bar could only animate
    // fake progress. xhr.upload.onprogress reports the real bytes sent by
    // the browser, which is the only genuine progress signal available.
    xhr.upload.onprogress = (e: ProgressEvent) => {
      notify({
        loaded: e.loaded,
        total: e.lengthComputable ? e.total : null,
        stage: "uploading",
      });
    };
    // WHY report "processing" from upload.onload and not from the first
    // progress event at 100%: some browsers coalesce the final event, so
    // the explicit load signal is the reliable handoff between "bytes
    // traveling" and "server thinking".
    xhr.upload.onload = () => {
      notify({ loaded: file.size, total: file.size, stage: "processing" });
    };
    xhr.onload = () => {
      const body = xhr.response as { error?: unknown } | null;
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(xhr.response as AnalyzeResult);
        return;
      }
      const serverMessage =
        body != null && typeof body.error === "string" ? body.error : "";
      reject(new UploadError(uploadErrorMessage(xhr.status, serverMessage), xhr.status));
    };
    xhr.onerror = () => {
      reject(
        new UploadError(
          "No se pudo contactar con la API. Verifique que esté en ejecución e inténtelo de nuevo.",
          0
        )
      );
    };
    notify({ loaded: 0, total: file.size, stage: "uploading" });
    xhr.send(file);
  });
}

