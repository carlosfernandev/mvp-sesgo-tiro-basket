import type { SessionSummary } from "../types";

interface Props {
  sessions: SessionSummary[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}

// Session picker: plain list, each entry shows the frame count and fps so
// the user can tell runs apart before loading rows.
export default function SessionPicker({ sessions, selectedId, onSelect }: Props) {
  return (
    <section aria-label="Sesiones disponibles">
      <h2>Sesiones</h2>
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
