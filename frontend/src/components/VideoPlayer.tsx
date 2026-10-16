import { forwardRef } from "react";
import { timeToFrame } from "../types";

interface Props {
  sessionId: string;
  videoSrc: string;
  fps: number | null;
  frameCount: number | null;
  currentFrame: number;
  onFrameChange: (frame: number) => void;
}

// Video player with a live frame readout.
//
// WHY the readout derives from `video.currentTime * fps` instead of trusting
// row timestamps: the <video> element is the playback clock — it is the only
// thing that knows where playback actually is during scrubbing, buffering
// and variable-rate play. Timestamps describe capture time; currentTime
// describes what the user sees right now, so mapping it through fps keeps
// the frame number glued to the visible picture.
const VideoPlayer = forwardRef<HTMLVideoElement, Props>(function VideoPlayer(
  { sessionId, videoSrc, fps, frameCount, currentFrame, onFrameChange },
  ref
) {
  const readout =
    fps != null && fps > 0
      ? `Fotograma ${currentFrame}${frameCount != null ? ` / ${frameCount - 1}` : ""}`
      : "FPS no disponibles";

  return (
    <section aria-label="Vídeo de la sesión">
      <h2>Sesión: {sessionId}</h2>
      <video
        key={sessionId}
        ref={ref}
        src={videoSrc}
        controls
        preload="metadata"
        className="video"
        onTimeUpdate={(e) => {
          // WHY read the event target instead of the forwarded ref: during
          // rapid seeking the ref may already point at the next session's
          // element while this event belongs to the old one.
          if (fps == null || fps <= 0) return;
          const t = (e.target as HTMLVideoElement).currentTime;
          const max = frameCount != null ? frameCount - 1 : Number.MAX_SAFE_INTEGER;
          onFrameChange(Math.min(Math.max(timeToFrame(t, fps), 0), max));
        }}
      />
      <p className="frame-readout" aria-live="polite">
        {readout}
      </p>
    </section>
  );
});

export default VideoPlayer;
