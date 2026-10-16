// Shared shapes mirroring the Flask API contract exactly.
//
// WHY the accented keys below are kept verbatim: the API serves CSV column
// names untouched (`ángulo_codo`, `ángulo_rodilla`, `ángulo_hombro` —
// verified against a live `/api/sesiones/<id>` response), and any silent
// "normalization" to unaccented names would compile fine while reading
// `undefined` at runtime. The explicit ASCII aliases in ChartPoint exist so
// Recharts dataKeys stay plain identifiers; the mapping happens in one place
// (toChartPoints) where a typo fails loudly instead of hiding in a chart.

// The five phase values the pipeline can emit. `sin_deteccion` is not a
// real phase — it marks frames without detection — so every visual
// component must render it as "no data" (grey), never as a colored phase.
export type Phase =
  | "preparacion"
  | "elevacion"
  | "liberacion"
  | "seguimiento"
  | "sin_deteccion";

export const PHASES: Phase[] = [
  "preparacion",
  "elevacion",
  "liberacion",
  "seguimiento",
  "sin_deteccion",
];

// Human-readable Spanish labels for the UI (code stays in English).
export const PHASE_LABELS: Record<Phase, string> = {
  preparacion: "Preparación",
  elevacion: "Elevación",
  liberacion: "Liberación",
  seguimiento: "Seguimiento",
  sin_deteccion: "Sin detección",
};

// Distinct fill per phase; grey for sin_deteccion so missing data reads as
// missing, not as a fifth stage of the shot.
export const PHASE_COLORS: Record<Phase, string> = {
  preparacion: "#3b82f6",
  elevacion: "#22c55e",
  liberacion: "#f59e0b",
  seguimiento: "#a855f7",
  sin_deteccion: "#9ca3af",
};

export interface PhaseSpan {
  start_frame: number;
  end_frame: number;
  frames: number;
}

// One CSV row, keyed exactly as the API returns it.
export interface ShotRow {
  frame: number;
  timestamp: number;
  ángulo_codo: number;
  ángulo_rodilla: number;
  ángulo_hombro: number;
  fase_asignada: string;
}

export interface SessionSummary {
  id: string;
  fps: number | null;
  frame_count: number | null;
  has_perfil: boolean;
  has_video: boolean;
  video: string | null;
  phase_spans: Partial<Record<Phase, PhaseSpan>>;
  error?: string;
}

export interface SessionDetail extends SessionSummary {
  rows: ShotRow[];
}

// Chart-friendly projection with ASCII keys.
//
// WHY map instead of passing ShotRow straight into Recharts: Recharts
// resolves series via string dataKeys, and a dataKey with an accent works
// only as long as every participant spells the accent identically —
// invisible to linters and to TypeScript. Mapping once, explicitly, turns a
// whole class of silent chart bugs into a compile error at this boundary.
export interface ChartPoint {
  frame: number;
  codo: number | null;
  rodilla: number | null;
  hombro: number | null;
  fase: string;
}

// One analysis row from POST /api/analizar: same CSV shape as ShotRow, but
// the angle cells may be null.
//
// WHY a separate interface instead of reusing ShotRow: the upload endpoint
// serializes NaN (frames with no pose detection) as JSON null, so its rows
// are truthfully nullable while the GET endpoints coerce every cell to a
// finite float at parse time (an unparsable cell marks the whole session
// with an error instead). Sharing one type would either lie about GET rows
// or force needless null checks on them; the PlottableRow union below is
// the single boundary where both are accepted for drawing.
export interface UploadRow {
  frame: number;
  timestamp: number;
  ángulo_codo: number | null;
  ángulo_rodilla: number | null;
  ángulo_hombro: number | null;
  fase_asignada: string;
}

// Either row flavor the charts know how to draw. A null angle becomes a
// gap in the line (Recharts skips null points), which reads as "no
// detection" on that stretch instead of a fabricated zero.
export type PlottableRow = ShotRow | UploadRow;

export function toChartPoints(rows: PlottableRow[]): ChartPoint[] {
  return rows.map((r) => ({
    frame: r.frame,
    codo: r["ángulo_codo"],
    rodilla: r["ángulo_rodilla"],
    hombro: r["ángulo_hombro"],
    fase: r.fase_asignada,
  }));
}

// WHY frame math goes through helpers: the video element only knows seconds
// (currentTime) while the analysis only knows frames, so every sync site
// (readout, chart cursor, seeking) must apply the same conversion or the
// two views drift by one frame in opposite directions.
export function frameToTime(frame: number, fps: number): number {
  return frame / fps;
}

export function timeToFrame(currentTime: number, fps: number): number {
  return Math.floor(currentTime * fps);
}

export type DominantSide = "auto" | "left" | "right";

export interface UploadOptions {
  side?: DominantSide;
  feedback?: boolean;
}

// Success payload of POST /api/analizar, mirroring the backend contract:
// only the CSV is persisted (csv path), the annotated video is ephemeral
// and fetched separately from video_url (null when feedback=false), and rows may carry null angles.
export interface AnalyzeResult {
  id: string;
  frame_count: number;
  fps: number;
  width: number;
  height: number;
  side?: DominantSide;
  phase_spans: Partial<Record<Phase, PhaseSpan>>;
  rows: UploadRow[];
  csv: string;
  video_url: string | null;
  warnings: string[];
}

