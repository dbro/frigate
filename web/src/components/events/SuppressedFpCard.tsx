import { useMemo, useState } from "react";
import { FiInfo } from "react-icons/fi";
import useSWR from "swr";
import { baseUrl } from "@/api/baseUrl";
import { FrigateConfig } from "@/types/frigateConfig";
import { SuppressedFpItem } from "@/types/reviewTrust";
import { use24HourTime, useFormattedTimestamp } from "@/hooks/use-date-utils";
import { formatUnixTimestampToDateTimeWithCentiseconds } from "@/utils/dateUtil";
import { getIconForLabel } from "@/utils/iconUtil";
import TimeAgo from "@/components/dynamic/TimeAgo";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";

function formatPct(score: number | null | undefined): string {
  if (score == null || !Number.isFinite(score)) {
    return "—";
  }
  return `${Math.round(score * 100)}%`;
}

function formatDuration(sec: number | null | undefined): string {
  if (sec == null || !Number.isFinite(sec)) {
    return "—";
  }
  if (sec < 60) {
    return `${sec.toFixed(2)}s`;
  }
  const mins = Math.floor(sec / 60);
  const rem = sec - mins * 60;
  return `${mins}m ${rem.toFixed(1)}s`;
}

type SuppressedFpCardProps = {
  item: SuppressedFpItem;
  onOpenRecording: (camera: string, startTime: number) => void;
  onOpenInCorrections?: () => void;
};

export default function SuppressedFpCard({
  item,
  onOpenRecording,
  onOpenInCorrections,
}: SuppressedFpCardProps) {
  const { data: config } = useSWR<FrigateConfig>("config");
  const is24Hour = use24HourTime(config);
  const [imgFailed, setImgFailed] = useState(false);

  const stillUrl =
    item.example_id && item.has_still
      ? `${baseUrl}api/secondlook/save_and_review/${encodeURIComponent(item.example_id)}/still.jpg`
      : null;

  const formattedShortDate = useFormattedTimestamp(
    item.start_time,
    is24Hour ? "%b %-d, %H:%M" : "%b %-d, %-I:%M %p",
    config?.ui?.timezone,
  );

  const fullStartTime = useMemo(
    () =>
      formatUnixTimestampToDateTimeWithCentiseconds(
        item.start_time,
        is24Hour,
        config?.ui?.timezone,
      ),
    [item.start_time, is24Hour, config?.ui?.timezone],
  );

  return (
    <div className="review-item relative flex flex-col overflow-hidden rounded-lg border border-rose-400/60 bg-card">
      <div
        role="button"
        tabIndex={0}
        onClick={() => onOpenRecording(item.camera, item.start_time)}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            onOpenRecording(item.camera, item.start_time);
          }
        }}
        className="relative aspect-video cursor-pointer overflow-hidden bg-muted"
      >
        {stillUrl && !imgFailed ? (
          <img
            src={stillUrl}
            alt={`Auto-suppressed ${item.label} on ${item.camera}`}
            className="size-full object-cover"
            loading="lazy"
            onError={() => setImgFailed(true)}
          />
        ) : (
          <div className="flex size-full flex-col items-center justify-center gap-1.5 bg-muted/80 px-4 text-center text-xs text-muted-foreground">
            <span className="rounded-full bg-rose-400/20 p-2 text-rose-500">
              {getIconForLabel(item.label, "object", "size-5")}
            </span>
            <span className="font-medium capitalize text-foreground">
              {item.label} · {item.camera}
            </span>
            <span>Auto-Suppressed FP (click to open recording)</span>
          </div>
        )}

        {/* Top gradient & object chip */}
        <div className="pointer-events-none absolute inset-x-0 top-0 z-10 h-[30%] bg-gradient-to-b from-black/60 to-transparent" />
        <div className="absolute left-3 top-2 z-20 flex items-center gap-1.5">
          <div className="inline-flex items-center gap-1 rounded-full bg-rose-500/90 px-2 py-0.5 text-[11px] font-medium text-white shadow">
            {getIconForLabel(item.label, "object", "size-3 text-white")}
            <span className="capitalize">{item.label}</span>
          </div>
        </div>

        {/* Chart-colored Info Badge matching Auto-Suppressed FP (#e17087) */}
        <Tooltip delayDuration={150}>
          <TooltipTrigger asChild>
            <button
              type="button"
              aria-label="Clip details: Auto-Suppressed FP"
              onClick={(e) => e.stopPropagation()}
              onMouseDown={(e) => e.stopPropagation()}
              className="absolute right-2 top-2 z-40 flex size-6 items-center justify-center rounded-full border border-rose-300/70 bg-rose-400 text-white shadow-md transition-transform hover:scale-110"
            >
              <FiInfo className="size-3.5 stroke-[2.4]" />
            </button>
          </TooltipTrigger>
          <TooltipContent
            side="left"
            align="start"
            className="z-50 w-72 border border-border bg-card p-3 text-xs text-foreground shadow-xl"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="space-y-2">
              <div className="flex items-center gap-1.5 border-b border-border pb-1.5 font-semibold">
                <span
                  className="size-2.5 shrink-0 rounded-sm"
                  style={{ backgroundColor: "#e17087" }}
                />
                <span>Auto-Suppressed FP</span>
              </div>
              <div className="space-y-1 text-[11px]">
                <div className="flex justify-between gap-2">
                  <span className="text-muted-foreground">Camera</span>
                  <span className="font-medium">{item.camera}</span>
                </div>
                <div className="flex justify-between gap-2">
                  <span className="text-muted-foreground">Start time</span>
                  <span className="font-mono font-medium">{fullStartTime}</span>
                </div>
                <div className="flex justify-between gap-2">
                  <span className="text-muted-foreground">Track duration</span>
                  <span className="font-mono">
                    {formatDuration(item.duration_sec)}
                  </span>
                </div>
                {item.zones.length > 0 && (
                  <div className="flex justify-between gap-2">
                    <span className="text-muted-foreground">Zones</span>
                    <span className="truncate text-right font-medium">
                      {item.zones.join(", ")}
                    </span>
                  </div>
                )}
              </div>
              <div className="border-t border-border pt-1.5 text-[11px]">
                <div className="mb-1 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
                  Secondlook Suppression
                </div>
                <div className="flex justify-between gap-2">
                  <span className="text-muted-foreground">Object</span>
                  <span className="font-medium capitalize">{item.label}</span>
                </div>
                <div className="flex justify-between gap-2">
                  <span className="text-muted-foreground">Primary score</span>
                  <span className="font-mono">
                    {formatPct(item.primary_score)}
                  </span>
                </div>
                <div className="flex justify-between gap-2">
                  <span className="text-muted-foreground">
                    Secondlook score
                  </span>
                  <span className="font-mono">{formatPct(item.sl_score)}</span>
                </div>
                {item.factor != null && (
                  <div className="flex justify-between gap-2">
                    <span className="text-muted-foreground">Scale factor</span>
                    <span className="font-mono">{item.factor.toFixed(2)}×</span>
                  </div>
                )}
              </div>
            </div>
          </TooltipContent>
        </Tooltip>

        {/* Bottom timestamp overlay */}
        <div className="pointer-events-none absolute inset-x-0 bottom-0 z-10 h-[24%] bg-gradient-to-t from-black/70 to-transparent">
          <div className="mx-3 flex h-full items-end justify-between pb-1 text-sm text-white">
            <TimeAgo time={item.start_time * 1000} dense />
            <span>{formattedShortDate}</span>
          </div>
        </div>
      </div>

      {/* Footer matching ReviewFeedbackBar height */}
      <div className="flex items-center justify-between gap-2 border-t border-border bg-card px-2.5 py-1.5 text-xs">
        <span className="inline-flex min-w-0 items-center gap-1.5 truncate font-medium text-rose-500 dark:text-rose-400">
          <span
            className="size-2 shrink-0 rounded-sm"
            style={{ backgroundColor: "#e17087" }}
          />
          <span className="truncate">Auto-Suppressed FP</span>
        </span>
        {onOpenInCorrections && (
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              onOpenInCorrections();
            }}
            className="shrink-0 rounded-md border border-border px-2 py-0.5 text-[11px] font-medium text-muted-foreground transition hover:bg-muted hover:text-foreground"
          >
            Corrections →
          </button>
        )}
      </div>
    </div>
  );
}
