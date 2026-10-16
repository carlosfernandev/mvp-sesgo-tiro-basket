import { useRef, useState } from "react";
import {
  MAX_UPLOAD_BYTES,
  UploadError,
  formatMegabytes,
  uploadVideo,
} from "../api";
import type { UploadProgress } from "../api";
import type { AnalyzeResult, DominantSide } from "../types";

interface Props {
  onResult: (result: AnalyzeResult) => void;
}

type Phase = "idle" | "busy" | "error";

interface SideOption {
  id: DominantSide;
  title: string;
  desc: string;
}

const SIDE_OPTIONS: SideOption[] = [
  {
    id: "auto",
    title: "Automático (Recomendado)",
    desc: "Detecta el lado según visibilidad",
  },
  {
    id: "right",
    title: "Diestro (Derecho)",
    desc: "Cadena hombro-codo-muñeca derecha",
  },
  {
    id: "left",
    title: "Zurdo (Izquierdo)",
    desc: "Cadena hombro-codo-muñeca izquierda",
  },
];

// WHY validate by file extension and not by file.type: several desktop
// environments report an empty MIME type for .mp4 files, so trusting
// file.type alone would reject valid uploads; the extension check plus the
// server-side probe (which truly verifies the container) is the honest pair.
function isMp4Name(name: string): boolean {
  return name.toLowerCase().endsWith(".mp4");
}

export default function UploadPanel({ onResult }: Props) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [side, setSide] = useState<DominantSide>("auto");
  const [feedback, setFeedback] = useState<boolean>(true);
  const [phase, setPhase] = useState<Phase>("idle");
  const [progress, setProgress] = useState<UploadProgress | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement | null>(null);

  const handleFileSelected = (file: File) => {
    if (!isMp4Name(file.name)) {
      setPhase("error");
      setError(
        "Solo se admiten archivos MP4 (.mp4). Seleccione un vídeo en ese formato."
      );
      return;
    }
    // WHY reject oversized files before sending a byte: the server would
    // answer 413 anyway, but only after the whole payload traveled; failing
    // fast here saves minutes on a slow connection.
    if (file.size > MAX_UPLOAD_BYTES) {
      setPhase("error");
      setError(
        `El archivo pesa ${formatMegabytes(file.size)} y supera el límite de ${formatMegabytes(MAX_UPLOAD_BYTES)}. Recorte o comprima el vídeo e inténtelo de nuevo.`
      );
      return;
    }
    setSelectedFile(file);
    setPhase("idle");
    setError(null);
    setProgress(null);
  };

  const handleRemoveFile = () => {
    if (phase === "busy") return;
    setSelectedFile(null);
    setPhase("idle");
    setError(null);
    setProgress(null);
    if (inputRef.current) {
      inputRef.current.value = "";
    }
  };

  const handleStartAnalysis = () => {
    if (!selectedFile || phase === "busy") return;

    setPhase("busy");
    setError(null);
    setProgress({ loaded: 0, total: selectedFile.size, stage: "uploading" });

    uploadVideo(selectedFile, { side, feedback }, setProgress)
      .then((result) => {
        setPhase("idle");
        setProgress(null);
        onResult(result);
      })
      .catch((err: unknown) => {
        setPhase("error");
        setProgress(null);
        setError(
          err instanceof UploadError || err instanceof Error
            ? err.message
            : "Error desconocido durante la subida."
        );
      });
  };

  const percent =
    progress != null && progress.total != null && progress.total > 0
      ? Math.min(100, Math.round((progress.loaded / progress.total) * 100))
      : null;

  const busy = phase === "busy";
  const processing = progress?.stage === "processing";

  return (
    <section aria-label="Subir vídeo para analizar">
      <h2>Subir vídeo</h2>
      <p className="hint">
        Seleccione o arrastre un archivo MP4 (máximo {formatMegabytes(MAX_UPLOAD_BYTES)}) y configure los parámetros biomecánicos antes de iniciar el análisis.
      </p>

      {/* Hidden file input for selection and replacement */}
      <input
        ref={inputRef}
        type="file"
        accept="video/mp4,.mp4"
        hidden
        disabled={busy}
        aria-label="Seleccionar archivo MP4"
        onChange={(e) => {
          const file = e.target.files?.[0];
          // WHY reset input value: choosing the same file twice would
          // otherwise fire no change event.
          e.target.value = "";
          if (file && !busy) handleFileSelected(file);
        }}
      />

      {selectedFile == null ? (
        <div
          className={dragging ? "dropzone dragging" : "dropzone"}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            if (busy) return;
            const file = e.dataTransfer.files?.[0];
            if (file) handleFileSelected(file);
          }}
        >
          <p>
            {dragging
              ? "Suelte el archivo para prepararlo"
              : "Arrastre aquí su archivo MP4"}
          </p>
          <button
            type="button"
            disabled={busy}
            onClick={() => inputRef.current?.click()}
          >
            Seleccionar archivo MP4
          </button>
        </div>
      ) : (
        <>
          <div className="staged-card">
            <div className="staged-file-info">
              <div className="staged-header">
                <span className="staged-file-name" title={selectedFile.name}>
                  {selectedFile.name}
                </span>
                <span className="staged-badge">Listo para analizar</span>
              </div>
              <p className="staged-meta">
                Tamaño: {formatMegabytes(selectedFile.size)}
              </p>
            </div>
            <div className="staged-actions">
              <button
                type="button"
                className="secondary-btn"
                disabled={busy}
                onClick={() => inputRef.current?.click()}
              >
                Cambiar vídeo
              </button>
              <button
                type="button"
                className="secondary-btn"
                disabled={busy}
                onClick={handleRemoveFile}
              >
                Quitar
              </button>
            </div>
          </div>

          <div className="config-section">
            <h3>Configuración del análisis</h3>

            <div className="option-group">
              <label className="option-label" id="side-selector-label">
                Lado dominante del tirador
              </label>
              <div
                className="side-selector"
                role="radiogroup"
                aria-labelledby="side-selector-label"
              >
                {SIDE_OPTIONS.map((opt) => {
                  const isSelected = side === opt.id;
                  return (
                    <label
                      key={opt.id}
                      className={`side-option ${isSelected ? "active" : ""}`}
                    >
                      <input
                        type="radio"
                        name="dominant-side"
                        value={opt.id}
                        checked={isSelected}
                        disabled={busy}
                        onChange={() => setSide(opt.id)}
                        className="sr-only"
                      />
                      <span className="side-option-title">{opt.title}</span>
                      <span className="side-option-desc">{opt.desc}</span>
                    </label>
                  );
                })}
              </div>
            </div>

            <div className="option-group">
              <div className="feedback-toggle">
                <label className="toggle-label">
                  <input
                    type="checkbox"
                    checked={feedback}
                    disabled={busy}
                    onChange={(e) => setFeedback(e.target.checked)}
                  />
                  <div className="toggle-content">
                    <span className="toggle-title">
                      Generar vídeo con anotaciones biomecánicas
                    </span>
                    <span className="toggle-desc">
                      Renderiza el vídeo con el esqueleto trazado, arcos de ángulos y HUD
                      de fases. Desmárquelo si solo desea las curvas y datos con mayor rapidez.
                    </span>
                  </div>
                </label>
              </div>
            </div>

            <div className="action-row">
              <button
                type="button"
                className="submit-btn"
                disabled={busy}
                onClick={handleStartAnalysis}
              >
                {busy
                  ? processing
                    ? "Procesando en el servidor…"
                    : "Subiendo vídeo…"
                  : "Iniciar análisis de tiro"}
              </button>
            </div>
          </div>
        </>
      )}

      {progress != null && (
        <div className="upload-progress-card" aria-live="polite">
          {processing ? (
            <p className="progress-status">
              Procesando vídeo en el servidor… Esto puede tardar decenas de
              segundos porque cada fotograma se analiza por separado.
            </p>
          ) : (
            <p className="progress-status">
              Subiendo vídeo…{" "}
              {percent != null
                ? `${percent} %`
                : `${formatMegabytes(progress.loaded)} enviados`}
            </p>
          )}
          <div
            className="progress"
            role="progressbar"
            aria-label={processing ? "Procesando vídeo" : "Subiendo vídeo"}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={processing ? undefined : (percent ?? undefined)}
          >
            <div
              className={
                processing ? "progress-fill indeterminate" : "progress-fill"
              }
              style={processing || percent == null ? undefined : { width: `${percent}%` }}
            />
          </div>
        </div>
      )}

      {phase === "error" && error != null && (
        <div className="error-alert" role="alert">
          <p className="error">{error}</p>
          {selectedFile != null && (
            <button
              type="button"
              className="retry-btn"
              onClick={handleStartAnalysis}
            >
              Reintentar análisis
            </button>
          )}
        </div>
      )}
    </section>
  );
}
