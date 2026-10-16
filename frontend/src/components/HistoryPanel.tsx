import { PHASES, PHASE_COLORS, PHASE_LABELS } from "../types";
import type { SessionSummary } from "../types";

interface Props {
  sessions: SessionSummary[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}

// "Historial de registros": top-level listing of every persisted analysis.
// Each entry shows frame count, fps and the per-phase frame breakdown so
// runs are distinguishable without opening them.
export default function HistoryPanel({ sessions, selectedId, onSelect }: Props) {
  return (
    <section aria-label="Historial de registros">
      <h2>Historial de registros</h2>
      <ul className="session-list">
        {sessions.map((s) => {
          const active = s.id === selectedId;
          const meta =
            s.frame_count != null
              ? `${s.frame_count} fotogramas · ${s.fps != null ? `${s.fps.toFixed(1)} fps` : "fps desconocidos"}`
              : "Sesión con errores";
          return (
            <li key={s.id}>
              <button
                type="button"
                className={active ? "session-item active" : "session-item"}
                aria-pressed={active}
                onClick={() => onSelect(s.id)}
              >
                <span className="session-id">{s.id}</span>
                <span className="session-meta">{meta}</span>
                <span className="phase-breakdown">
                  {PHASES.filter((p) => s.phase_spans[p] != null).map((p) => {
                    const span = s.phase_spans[p];
                    if (span == null) return null;
                    return (
                      <span
                        key={p}
                        className="phase-chip"
                        title={`${PHASE_LABELS[p]}: fotogramas ${span.start_frame}–${span.end_frame}`}
                      >
                        <span
                          className="phase-dot"
                          style={{ backgroundColor: PHASE_COLORS[p] }}
                        />
                        {PHASE_LABELS[p]}: {span.frames}
                      </span>
                    );
                  })}
                </span>
                {s.error != null && s.error !== "" && (
                  <span className="session-error">Error: {s.error}</span>
                )}
              </button>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
