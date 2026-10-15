import { useMemo, useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { PHASE_COLORS, PHASE_LABELS, toChartPoints } from "../types";
import type { Phase, PhaseSpan, PlottableRow } from "../types";

interface Props {
  rows: PlottableRow[];
  phaseSpans: Partial<Record<Phase, PhaseSpan>>;
  currentFrame: number;
  onSeek: (frame: number) => void;
}

// Angle chart with two-way video sync: hovering seeks the video, and the
// playback cursor (ReferenceLine) follows the video while it plays.
export default function AngleChart({ rows, phaseSpans, currentFrame, onSeek }: Props) {
  const [showRodilla, setShowRodilla] = useState(false);
  const [showHombro, setShowHombro] = useState(false);

  // WHY memoize the projection: toChartPoints allocates one object per
  // frame, and rebuilding it on every video timeupdate (several per second)
  // would churn garbage while the user scrubs.
  const data = useMemo(() => toChartPoints(rows), [rows]);

  const lastFrame = rows.length > 0 ? rows[rows.length - 1].frame : 0;

  return (
    <section aria-label="Evolución de ángulos">
      <h3>Ángulos por fotograma</h3>
      <div className="series-toggles">
        <span className="series-main">Ángulo del codo (principal)</span>
        <label>
          <input
            type="checkbox"
            checked={showRodilla}
            onChange={(e) => setShowRodilla(e.target.checked)}
          />
          Ángulo de la rodilla
        </label>
        <label>
          <input
            type="checkbox"
            checked={showHombro}
            onChange={(e) => setShowHombro(e.target.checked)}
          />
          Ángulo del hombro
        </label>
      </div>
      <div className="chart-wrap">
        <ResponsiveContainer width="100%" height={320}>
          <LineChart
            data={data}
            margin={{ top: 8, right: 16, bottom: 8, left: 0 }}
            onMouseMove={(state) => {
              // WHY activeLabel and not chartX pixels: Recharts already maps
              // the pointer to the nearest frame datum, so seeking by label
              // lands exactly on a real frame instead of a fractional guess.
              const label = state?.activeLabel;
              if (label == null) return;
              const frame = Number(label);
              if (Number.isFinite(frame)) onSeek(frame);
            }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              dataKey="frame"
              type="number"
              domain={[0, lastFrame]}
              label={{ value: "Fotograma", position: "insideBottomRight", offset: -2 }}
            />
            <YAxis label={{ value: "Grados", angle: -90, position: "insideLeft" }} />
            <Tooltip
              formatter={(value, name) => {
                const label =
                  name === "codo"
                    ? "Ángulo del codo"
                    : name === "rodilla"
                      ? "Ángulo de la rodilla"
                      : "Ángulo del hombro";
                // WHY the null guard: upload rows carry null angles on frames
                // without detection (NaN serialized as null), and hovering a
                // gap must read as missing data, not as a 0.0° measurement.
                if (value == null) return ["sin datos", label];
                return [`${Number(value).toFixed(1)}°`, label];
              }}
              labelFormatter={(frame) => `Fotograma ${frame}`}
            />
            {/* Phase background bands, drawn from phase_spans so the chart
                and the timeline strip share one source of truth. */}
            {(
              Object.entries(phaseSpans) as [Phase, PhaseSpan | undefined][]
            ).map(([phase, span]) => {
              if (span == null) return null;
              return (
                <ReferenceArea
                  key={phase}
                  x1={span.start_frame}
                  x2={span.end_frame}
                  fill={PHASE_COLORS[phase]}
                  fillOpacity={phase === "sin_deteccion" ? 0.25 : 0.12}
                  label={{
                    value: PHASE_LABELS[phase],
                    position: "insideTopLeft",
                    fontSize: 11,
                  }}
                />
              );
            })}
            <Line type="monotone" dataKey="codo" name="codo" stroke="#2563eb" dot={false} strokeWidth={2} />
            {showRodilla && (
              <Line type="monotone" dataKey="rodilla" name="rodilla" stroke="#16a34a" dot={false} strokeWidth={1.5} />
            )}
            {showHombro && (
              <Line type="monotone" dataKey="hombro" name="hombro" stroke="#ea580c" dot={false} strokeWidth={1.5} />
            )}
            {/* Playback cursor: re-rendered from the video clock via props. */}
            <ReferenceLine x={currentFrame} stroke="#111827" strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}
