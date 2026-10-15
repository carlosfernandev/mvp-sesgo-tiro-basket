import { PHASES, PHASE_COLORS, PHASE_LABELS } from "../types";
import type { Phase, PhaseSpan } from "../types";

interface Props {
  phaseSpans: Partial<Record<Phase, PhaseSpan>>;
  frameCount: number;
  currentFrame: number;
  onSeek: (frame: number) => void;
}

// Horizontal phase strip aligned with the video: each segment width is
// proportional to its frame share, and clicking a segment seeks the video
// to that phase start.
export default function PhaseTimeline({ phaseSpans, frameCount, currentFrame, onSeek }: Props) {
  if (frameCount <= 0) return null;
  return (
    <section aria-label="Fases del tiro">
      <h3>Fases</h3>
      <div
        className="phase-strip"
        role="slider"
        aria-label="Línea de tiempo de fases"
        aria-valuemin={0}
        aria-valuemax={frameCount - 1}
        aria-valuenow={currentFrame}
      >
        {PHASES.filter((p) => phaseSpans[p] != null).map((phase) => {
          const span = phaseSpans[phase] as PhaseSpan;
          const widthPct = (span.frames / frameCount) * 100;
          return (
            <button
              key={phase}
              type="button"
              title={`${PHASE_LABELS[phase]}: fotogramas ${span.start_frame}–${span.end_frame}`}
              className="phase-segment"
              style={{ width: `${widthPct}%`, backgroundColor: PHASE_COLORS[phase] }}
              onClick={() => onSeek(span.start_frame)}
            >
              <span className="phase-segment-label">{PHASE_LABELS[phase]}</span>
            </button>
          );
        })}
        {/* Playhead marker positioned by frame fraction so it tracks the video. */}
        <div
          className="phase-playhead"
          style={{ left: `${(currentFrame / frameCount) * 100}%` }}
        />
      </div>
    </section>
  );
}
