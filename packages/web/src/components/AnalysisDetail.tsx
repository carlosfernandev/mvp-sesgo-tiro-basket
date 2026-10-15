import { useCallback, useRef, useState } from "react";
import { frameToTime } from "../types";
import type { Phase, PhaseSpan, PlottableRow } from "../types";
import AngleChart from "./AngleChart";
import PhaseTimeline from "./PhaseTimeline";
import VideoPlayer from "./VideoPlayer";

interface Props {
  sessionId: string;
  rows: PlottableRow[];
  phaseSpans: Partial<Record<Phase, PhaseSpan>>;
  fps: number | null;
  frameCount: number | null;
  // WHY a nullable video source instead of two components: the ephemeral
  // upload result and the historical record share everything (synced chart,
  // phase strip, seek logic) except the picture. Passing null selects the
  // historical branch, which shows an explicit notice INSTEAD of a player —
  // so a record without video can never render a broken <video> element.
  videoSrc: string | null;
  warnings?: string[];
  perfilSrc?: string | null;
}

// Shared detail view for both worlds: the ephemeral post-upload result
// (videoSrc set, video + synced curve) and the historical record
// (videoSrc null, notice + curve + phases, no player at all).
export default function AnalysisDetail({
  sessionId,
  rows,
  phaseSpans,
  fps,
  frameCount,
  videoSrc,
  warnings = [],
  perfilSrc = null,
}: Props) {
  const [currentFrame, setCurrentFrame] = useState(0);
  const videoRef = useRef<HTMLVideoElement | null>(null);

  // WHY seek through the video element instead of state alone: the element
  // owns the picture — setting currentTime is what actually moves the image
  // (and fires the timeupdate that re-syncs the chart cursor). Updating
  // React state without touching the element would move the cursor while
  // the picture stays frozen, splitting the two views apart.
  const seekToFrame = useCallback(
    (frame: number) => {
      if (fps == null || fps <= 0) return;
      const max = rows.length > 0 ? rows[rows.length - 1].frame : frame;
      const clamped = Math.min(Math.max(Math.round(frame), 0), max);
      const video = videoRef.current;
      if (videoSrc != null && video) {
        if (Number.isFinite(video.duration) && video.duration > 0) {
          video.currentTime = Math.min(frameToTime(clamped, fps), video.duration);
        } else {
          // WHY still set currentTime with unknown duration: metadata may
          // not have loaded yet on first hover; the browser clamps it once
          // it can.
          video.currentTime = frameToTime(clamped, fps);
        }
      }
      setCurrentFrame(clamped);
    },
    [rows, fps, videoSrc]
  );

  return (
    <div className="detail">
      {videoSrc != null ? (
        <VideoPlayer
          ref={videoRef}
          sessionId={sessionId}
          videoSrc={videoSrc}
          fps={fps}
          frameCount={frameCount}
          currentFrame={currentFrame}
          onFrameChange={setCurrentFrame}
        />
      ) : (
        <section aria-label="Aviso sobre el vídeo">
          <h2>Registro: {sessionId}</h2>
          <p className="notice-historical" role="note">
            Este es un registro histórico: el vídeo no se conserva. Solo se
            guardan los datos del análisis (curvas y fases), por eso aquí no
            hay reproductor de vídeo. El vídeo anotado solo está disponible
            justo después de subirlo, en la pestaña «Subir vídeo».
          </p>
        </section>
      )}
      {warnings.length > 0 && (
        <ul className="warnings" aria-label="Advertencias del análisis">
          {warnings.map((w) => (
            <li key={w}>{w}</li>
          ))}
        </ul>
      )}
      {frameCount != null && (
        <PhaseTimeline
          phaseSpans={phaseSpans}
          frameCount={frameCount}
          currentFrame={currentFrame}
          onSeek={seekToFrame}
        />
      )}
      <AngleChart
        rows={rows}
        phaseSpans={phaseSpans}
        currentFrame={currentFrame}
        onSeek={seekToFrame}
      />
      {perfilSrc != null && (
        <section aria-label="Perfil angular">
          <h3>Perfil angular de referencia</h3>
          <img
            className="perfil"
            src={perfilSrc}
            alt={`Perfil angular de la sesión ${sessionId}`}
          />
        </section>
      )}
    </div>
  );
}
