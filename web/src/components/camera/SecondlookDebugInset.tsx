import useSWR from "swr";
import { useEffect, useState } from "react";

import axios from "axios";
import { z } from "zod";

const box = z.tuple([
  z.number().finite(),
  z.number().finite(),
  z.number().finite(),
  z.number().finite(),
]);
const frameSchema = z.object({
  version: z.number(),
  jpeg: z.string().max(6 * 1024 * 1024),
  frame_time: z.number().finite(),
  width: z.number().positive(),
  height: z.number().positive(),
  primary_box: box,
  sl_box: box.nullable(),
  label: z.string(),
  secondary_label: z.string().nullable(),
  primary_score: z.number(),
  secondary_score: z.number().nullable(),
});
const resultSchema = z.discriminatedUnion("state", [
  z.object({ state: z.literal("disabled"), frame: z.null() }),
  z.object({ state: z.literal("no_result"), frame: z.null() }),
  z.object({ state: z.literal("ready"), frame: frameSchema }),
]);
const fetchFrame = async (url: string) =>
  resultSchema.parse((await axios.get<unknown>(url)).data);

export default function SecondlookDebugInset({ camera }: { camera: string }) {
  const { data, error } = useSWR(`${camera}/secondlook`, fetchFrame, {
    refreshInterval: 1000,
  });
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, []);
  const frame = data?.state === "ready" ? data.frame : null;
  const age = frame ? Math.max(0, now / 1000 - frame.frame_time) : 0;
  return (
    <section
      aria-label="Secondlook analysed frame"
      className="my-2 rounded-lg border p-2"
    >
      <h3>Secondlook — latest analysed frame (not current video)</h3>
      {error ? (
        <p>Secondlook debug unavailable.</p>
      ) : !data ? (
        <p>Loading analysed frame…</p>
      ) : data.state === "disabled" ? (
        <p>Secondlook disabled for this camera.</p>
      ) : !frame ? (
        <p>No analysed result yet.</p>
      ) : (
        <>
          <p>
            Frame age: {age.toFixed(1)} s{" "}
            {age > 10 ? "— stale; not current video" : ""}
          </p>
          <p>
            Primary: {frame.label} {frame.primary_score.toFixed(3)} ·
            Secondlook: {frame.secondary_label ?? "no matched classification"}{" "}
            {frame.secondary_score?.toFixed(3) ?? ""}
          </p>
          <div className="relative max-w-xl">
            <img
              src={`data:image/jpeg;base64,${frame.jpeg}`}
              className="block w-full"
              alt="Secondlook analysed frame"
            />
            {frame.version === 1 ? (
              <svg
                viewBox={`0 0 ${frame.width} ${frame.height}`}
                className="pointer-events-none absolute inset-0 size-full"
                preserveAspectRatio="xMidYMid meet"
                aria-label="Boxes aligned to analysed frame"
              >
                {[
                  { box: frame.primary_box, color: "#22d3ee" },
                  { box: frame.sl_box, color: "#fbbf24" },
                ].map(
                  ({ box, color }, index) =>
                    box && (
                      <rect
                        key={index}
                        x={box[0]}
                        y={box[1]}
                        width={Math.max(0, box[2] - box[0])}
                        height={Math.max(0, box[3] - box[1])}
                        fill="none"
                        stroke={color}
                        strokeWidth={2}
                        vectorEffect="non-scaling-stroke"
                      />
                    ),
                )}
              </svg>
            ) : (
              <p>Overlay unavailable: no aligned provenance.</p>
            )}
          </div>
          <p className="text-sm">
            Cyan: primary · Amber: Secondlook. Boxes belong only to this still.
          </p>
        </>
      )}
    </section>
  );
}
