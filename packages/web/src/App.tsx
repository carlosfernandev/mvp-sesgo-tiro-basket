import { useCallback, useEffect, useState } from "react";
import { fetchSessionDetail, fetchSessions, perfilUrl } from "./api";
import type { AnalyzeResult, SessionDetail, SessionSummary } from "./types";
import AnalysisDetail from "./components/AnalysisDetail";
import EmptyState from "./components/EmptyState";
import HistoryPanel from "./components/HistoryPanel";
import UploadPanel from "./components/UploadPanel";

type Tab = "upload" | "history";

type HistoryState =
  | { kind: "loading" }
  | { kind: "empty" }
  | { kind: "ready"; sessions: SessionSummary[] }
  | { kind: "error"; message: string };

export default function App() {
  const [tab, setTab] = useState<Tab>("upload");
  const [uploadResult, setUploadResult] = useState<AnalyzeResult | null>(null);

  const [history, setHistory] = useState<HistoryState>({ kind: "loading" });
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<SessionDetail | null>(null);
  const [detailError, setDetailError] = useState<string | null>(null);

  const loadHistory = useCallback(() => {
    setHistory({ kind: "loading" });
    setDetail(null);
    setDetailError(null);
    fetchSessions()
      .then((sessions) => {
        if (sessions.length === 0) {
          setHistory({ kind: "empty" });
          setSelectedId(null);
        } else {
          setHistory({ kind: "ready", sessions });
          // WHY keep the previous selection when it still exists: reloading
          // the list (e.g. after a new upload) must not yank the user away
          // from the record they are reading.
          setSelectedId((prev) =>
            prev != null && sessions.some((s) => s.id === prev)
              ? prev
              : sessions[0].id
          );
        }
      })
      .catch((err: unknown) => {
        setHistory({
          kind: "error",
          message: err instanceof Error ? err.message : "Error desconocido",
        });
      });
  }, []);

  // WHY load on entering the tab instead of on mount: the upload tab is the
  // landing view, and the history is re-read every time the tab opens so a
  // just-uploaded analysis (its CSV is persisted server-side) shows up
  // without a manual refresh.
  useEffect(() => {
    if (tab === "history") loadHistory();
  }, [tab, loadHistory]);

  useEffect(() => {
    if (selectedId == null) return;
    setDetail(null);
    setDetailError(null);
    fetchSessionDetail(selectedId)
      .then(setDetail)
      .catch((err: unknown) => {
        setDetailError(err instanceof Error ? err.message : "Error desconocido");
      });
  }, [selectedId]);

  const frameCount = detail?.frame_count ?? detail?.rows.length ?? null;

  return (
    <main className="app">
      <h1>Análisis de tiro</h1>
      <nav className="topnav" aria-label="Navegación principal">
        <button
          type="button"
          className={tab === "upload" ? "nav-btn active" : "nav-btn"}
          aria-current={tab === "upload" ? "page" : undefined}
          onClick={() => setTab("upload")}
        >
          Subir vídeo
        </button>
        <button
          type="button"
          className={tab === "history" ? "nav-btn active" : "nav-btn"}
          aria-current={tab === "history" ? "page" : undefined}
          onClick={() => setTab("history")}
        >
          Historial de registros
        </button>
      </nav>

      {tab === "upload" && (
        <>
          <UploadPanel
            onResult={(result) => {
              setUploadResult(result);
            }}
          />
          {uploadResult != null && (
            <section aria-label="Resultado del análisis">
              <h2>Resultado: {uploadResult.id}</h2>
              {/*
                WHY video_url directly as the <video> source instead of
                fetching a Blob: the endpoint supports HTTP Range (206), so
                the browser streams and seeks server-side; buffering the
                whole file into a Blob URL would duplicate up to 200 MB in
                memory for no benefit. Ephemeral by design: this URL dies
                with the server process, which is why historical records
                show a notice instead of a player.
              */}
              <AnalysisDetail
                key={uploadResult.id}
                sessionId={uploadResult.id}
                rows={uploadResult.rows}
                phaseSpans={uploadResult.phase_spans}
                fps={uploadResult.fps}
                frameCount={uploadResult.frame_count}
                videoSrc={uploadResult.video_url}
                warnings={uploadResult.warnings}
              />
            </section>
          )}
        </>
      )}

      {tab === "history" && (
        <>
          {history.kind === "loading" && <p>Cargando historial…</p>}
          {history.kind === "error" && (
            <p className="error" role="alert">
              No se pudo cargar el historial: {history.message}. Verifique
              que la API esté en ejecución e inténtelo de nuevo.{" "}
              <button type="button" onClick={loadHistory}>
                Reintentar
              </button>
            </p>
          )}
          {history.kind === "empty" && (
            <>
              <EmptyState />
              <p>
                También puede subir un vídeo ahora mismo en la pestaña{" "}
                <button type="button" onClick={() => setTab("upload")}>
                  Subir vídeo
                </button>
                .
              </p>
            </>
          )}
          {history.kind === "ready" && (
            <div className="layout">
              <div>
                <HistoryPanel
                  sessions={history.sessions}
                  selectedId={selectedId}
                  onSelect={setSelectedId}
                />
                <p>
                  <button type="button" onClick={loadHistory}>
                    Actualizar historial
                  </button>
                </p>
              </div>
              <div>
                {detailError != null && (
                  <p className="error" role="alert">
                    No se pudo cargar el registro: {detailError}
                  </p>
                )}
                {detail == null && detailError == null && (
                  <p>Cargando registro…</p>
                )}
                {/*
                  WHY videoSrc is always null here: history entries persist
                  only the CSV — the video is never stored — so rendering a
                  player would either break (404) or mislead. AnalysisDetail
                  shows the explanatory notice instead.
                */}
                {detail != null && (
                  <AnalysisDetail
                    key={detail.id}
                    sessionId={detail.id}
                    rows={detail.rows}
                    phaseSpans={detail.phase_spans}
                    fps={detail.fps}
                    frameCount={frameCount}
                    videoSrc={null}
                    perfilSrc={detail.has_perfil ? perfilUrl(detail.id) : null}
                  />
                )}
              </div>
            </div>
          )}
        </>
      )}
    </main>
  );
}
