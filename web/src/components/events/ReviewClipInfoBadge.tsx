import { useMemo } from "react";
import { FiInfo } from "react-icons/fi";
import useSWR from "swr";
import { cn } from "@/lib/utils";
import { FrigateConfig } from "@/types/frigateConfig";
import { ReviewSegment } from "@/types/review";
import {
  ReviewClipMeta,
  ReviewFeedbackItem,
  TrustCategoryFilter,
} from "@/types/reviewTrust";
import { use24HourTime } from "@/hooks/use-date-utils";
import { formatUnixTimestampToDateTimeWithCentiseconds } from "@/utils/dateUtil";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";

type CategoryVisual = {
  label: string;
  badgeClass: string;
  dotColor: string;
};

const CATEGORY_VISUALS: Record<TrustCategoryFilter, CategoryVisual> = {
  alerts: {
    label: "Alerts (Auto)",
    badgeClass: "bg-blue-500 text-white border-blue-400/70",
    dotColor: "#3b82f6",
  },
  detections: {
    label: "Detections (Auto)",
    badgeClass: "bg-slate-500 text-white border-slate-400/70",
    dotColor: "#64748b",
  },
  rescued_fn: {
    label: "Rescued FN (Auto)",
    badgeClass: "bg-teal-400 text-slate-950 border-teal-300/80",
    dotColor: "#2dd4bf",
  },
  auto_suppressed_fp: {
    label: "Auto-Suppressed FP",
    badgeClass: "bg-rose-400 text-white border-rose-300/70",
    dotColor: "#e17087",
  },
  auto_misclassified: {
    label: "Misclassified (Auto)",
    badgeClass: "bg-yellow-400 text-slate-950 border-yellow-300/80",
    dotColor: "#facc15",
  },
  background_clips: {
    label: "Background Clip",
    badgeClass: "bg-background/90 text-foreground border-foreground/70",
    dotColor: "#94a3b8",
  },
  fp: {
    label: "FP (Human-Edited)",
    badgeClass: "bg-rose-500 text-white border-rose-400/70",
    dotColor: "#f43f5e",
  },
  fn: {
    label: "FN (Human-Edited)",
    badgeClass: "bg-red-600 text-white border-red-400/70",
    dotColor: "#dc2626",
  },
  human_misclassified: {
    label: "Misclassified (Human-Edited)",
    badgeClass: "bg-yellow-400 text-slate-950 border-yellow-300/80",
    dotColor: "#facc15",
  },
  golden: {
    label: "Golden Clip",
    badgeClass: "bg-amber-500 text-slate-950 border-amber-300/80",
    dotColor: "#f59e0b",
  },
};

function formatDuration(sec: number | null | undefined): string {
  if (sec == null || !Number.isFinite(sec)) {
    return "In progress";
  }
  if (sec < 60) {
    return `${sec.toFixed(2)}s`;
  }
  const mins = Math.floor(sec / 60);
  const rem = sec - mins * 60;
  return `${mins}m ${rem.toFixed(1)}s`;
}

function formatPct(score: number | null | undefined): string | null {
  if (score == null || !Number.isFinite(score)) {
    return null;
  }
  return `${Math.round(score * 100)}%`;
}

type ReviewClipInfoBadgeProps = {
  review: ReviewSegment;
  meta?: ReviewClipMeta;
  feedback?: ReviewFeedbackItem;
  isGolden?: boolean;
};

export default function ReviewClipInfoBadge({
  review,
  meta,
  feedback,
  isGolden = false,
}: ReviewClipInfoBadgeProps) {
  const { data: config } = useSWR<FrigateConfig>("config");
  const is24Hour = use24HourTime(config);

  const effectiveCategory: TrustCategoryFilter = useMemo(() => {
    if (feedback?.verdict === "fp") return "fp";
    if (feedback?.verdict === "fn") return "fn";
    if (feedback?.verdict === "misclassified") return "human_misclassified";
    if (meta?.category) return meta.category;
    return review.severity === "detection" ? "detections" : "alerts";
  }, [feedback?.verdict, meta?.category, review.severity]);

  const visual = CATEGORY_VISUALS[effectiveCategory];
  const isAlert = review.severity === "alert";

  const startTimeText = useMemo(
    () =>
      formatUnixTimestampToDateTimeWithCentiseconds(
        review.start_time,
        is24Hour,
        config?.ui?.timezone,
      ),
    [review.start_time, is24Hour, config?.ui?.timezone],
  );

  const durationSec = useMemo(() => {
    if (meta?.duration_sec != null) return meta.duration_sec;
    if (review.end_time != null) {
      return Math.max(0, review.end_time - review.start_time);
    }
    return null;
  }, [meta?.duration_sec, review.end_time, review.start_time]);

  const objectsList = useMemo(() => {
    if (meta?.objects && meta.objects.length > 0) {
      return meta.objects;
    }
    return (review.data.objects ?? []).map((lbl) => ({
      id: null,
      label: lbl.replace("-verified", ""),
      sub_label: null,
      score: null,
      top_score: null,
    }));
  }, [meta?.objects, review.data.objects]);

  const zonesList = meta?.zones?.length
    ? meta.zones
    : (review.data.zones ?? []);

  const timing = meta?.timing ?? null;
  const secondlook = meta?.secondlook ?? null;
  const isGood = feedback?.verdict === "good" || meta?.is_good;

  return (
    <Tooltip delayDuration={150}>
      <TooltipTrigger asChild>
        <button
          type="button"
          aria-label={`Clip details: ${visual.label}`}
          onClick={(e) => e.stopPropagation()}
          onMouseDown={(e) => e.stopPropagation()}
          className={cn(
            "absolute right-2 top-2 z-40 flex size-6 items-center justify-center rounded-full border shadow-md transition-transform hover:scale-110",
            visual.badgeClass,
          )}
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
          {/* Header: Chart Category Color + Label */}
          <div className="flex flex-wrap items-center justify-between gap-1.5 border-b border-border pb-1.5">
            <div className="flex items-center gap-1.5 font-semibold">
              <span
                className="size-2.5 shrink-0 rounded-sm"
                style={{ backgroundColor: visual.dotColor }}
              />
              <span>{visual.label}</span>
            </div>
            <div className="flex items-center gap-1">
              {isGood && (
                <span className="rounded bg-emerald-500/20 px-1.5 py-0.5 text-[10px] font-medium text-emerald-600 dark:text-emerald-400">
                  Good
                </span>
              )}
              {isGolden && (
                <span className="rounded bg-amber-500/20 px-1.5 py-0.5 text-[10px] font-medium text-amber-600 dark:text-amber-400">
                  🏆 Golden
                </span>
              )}
            </div>
          </div>

          {/* Core Clip Metadata */}
          <div className="space-y-1 text-[11px]">
            <div className="flex justify-between gap-2">
              <span className="text-muted-foreground">Camera</span>
              <span className="font-medium">{review.camera}</span>
            </div>
            <div className="flex justify-between gap-2">
              <span className="text-muted-foreground">Start time</span>
              <span className="font-mono font-medium">{startTimeText}</span>
            </div>
            <div className="flex justify-between gap-2">
              <span className="text-muted-foreground">Duration</span>
              <span className="font-mono">{formatDuration(durationSec)}</span>
            </div>
            {zonesList.length > 0 && (
              <div className="flex justify-between gap-2">
                <span className="text-muted-foreground">Zones</span>
                <span className="truncate text-right font-medium">
                  {zonesList.join(", ")}
                </span>
              </div>
            )}
            <div className="flex justify-between gap-2">
              <span className="text-muted-foreground">Status</span>
              <span>
                {review.has_been_reviewed ? "Reviewed" : "Unreviewed"}
              </span>
            </div>
          </div>

          {/* Objects & Scores */}
          {objectsList.length > 0 && (
            <div className="border-t border-border pt-1.5">
              <div className="mb-1 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
                Objects &amp; Scores
              </div>
              <div className="space-y-0.5 text-[11px]">
                {objectsList.map((obj, idx) => {
                  const topPct = formatPct(obj.top_score);
                  const scorePct = formatPct(obj.score);
                  const displayPct = topPct ?? scorePct;
                  return (
                    <div
                      key={`${obj.id ?? obj.label}-${idx}`}
                      className="flex items-center justify-between gap-2"
                    >
                      <span className="truncate capitalize">
                        {obj.label}
                        {obj.sub_label ? ` (${obj.sub_label})` : ""}
                      </span>
                      <span className="font-mono font-medium">
                        {displayPct ?? "—"}
                        {topPct && scorePct && topPct !== scorePct
                          ? ` (init ${scorePct})`
                          : ""}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Secondlook Details */}
          {secondlook && (
            <div className="border-t border-border pt-1.5 text-[11px]">
              <div className="mb-1 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
                Secondlook
              </div>
              {secondlook.live === "misclass" && secondlook.misclass_label && (
                <div className="flex justify-between gap-2">
                  <span className="text-muted-foreground">Reclassified</span>
                  <span className="font-medium text-yellow-600 dark:text-yellow-400">
                    {secondlook.label} → {secondlook.misclass_label}
                  </span>
                </div>
              )}
              {secondlook.primary_score != null &&
                secondlook.sl_score != null && (
                  <div className="flex justify-between gap-2">
                    <span className="text-muted-foreground">Scores</span>
                    <span className="font-mono">
                      Primary {formatPct(secondlook.primary_score)} · SL{" "}
                      {formatPct(secondlook.sl_score)}
                    </span>
                  </div>
                )}
              {secondlook.factor != null && (
                <div className="flex justify-between gap-2">
                  <span className="text-muted-foreground">Scale factor</span>
                  <span className="font-mono">
                    {secondlook.factor.toFixed(2)}×
                  </span>
                </div>
              )}
            </div>
          )}

          {/* Human Feedback Details */}
          {feedback && feedback.verdict !== "clear" && (
            <div className="border-t border-border pt-1.5 text-[11px]">
              <div className="mb-1 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
                Human Feedback
              </div>
              <div className="flex justify-between gap-2">
                <span className="text-muted-foreground">Verdict</span>
                <span className="font-medium uppercase">
                  {feedback.verdict}
                  {feedback.corrected_label
                    ? ` → ${feedback.corrected_label}`
                    : ""}
                </span>
              </div>
              {feedback.reason && (
                <div className="mt-0.5 text-muted-foreground">
                  {feedback.reason}
                </div>
              )}
            </div>
          )}

          {/* Per-Clip Alert / Detection Timing (pure timestamp differences) */}
          <div className="border-t border-border pt-1.5 text-[11px]">
            <div className="mb-1 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
              {isAlert ? "Alert Timing" : "Detection Timing"}
            </div>
            {timing ? (
              <div className="space-y-0.5">
                <div className="flex justify-between gap-2">
                  <span className="inline-flex items-center gap-1.5 text-muted-foreground">
                    <span className="size-2 rounded-sm bg-slate-400" />
                    Discovery (Request)
                  </span>
                  <span className="font-mono">
                    {(timing.discovery_ms / 1000).toFixed(2)}s
                  </span>
                </div>
                <div className="flex justify-between gap-2">
                  <span className="inline-flex items-center gap-1.5 text-amber-600 dark:text-amber-400">
                    <span className="size-2 rounded-sm bg-amber-500" />
                    Tracking
                  </span>
                  <span className="font-mono">
                    {((timing.tracking_ms ?? 0) / 1000).toFixed(2)}s
                  </span>
                </div>
                <div className="flex justify-between gap-2">
                  <span className="inline-flex items-center gap-1.5 text-violet-600 dark:text-violet-400">
                    <span className="size-2 rounded-sm bg-violet-500" />
                    Confirmation
                  </span>
                  <span className="font-mono">
                    {(timing.confirmation_ms / 1000).toFixed(2)}s
                  </span>
                </div>
                {isAlert && (timing.zone_delay_ms ?? 0) > 0 && (
                  <div className="flex justify-between gap-2">
                    <span className="inline-flex items-center gap-1.5 text-cyan-600 dark:text-cyan-400">
                      <span className="size-2 rounded-sm bg-cyan-500" />
                      Zone / Alert
                    </span>
                    <span className="font-mono">
                      {((timing.zone_delay_ms ?? 0) / 1000).toFixed(2)}s
                    </span>
                  </div>
                )}
                <div className="mt-0.5 flex justify-between gap-2 border-t border-border/60 pt-0.5 font-semibold">
                  <span>Total</span>
                  <span
                    className={cn(
                      "font-mono",
                      timing.total_ms <= 2000
                        ? "text-emerald-600 dark:text-emerald-400"
                        : "text-amber-600 dark:text-amber-400",
                    )}
                  >
                    {(timing.total_ms / 1000).toFixed(2)}s
                  </span>
                </div>
              </div>
            ) : (
              <div className="text-muted-foreground">
                No timing recorded for this clip
              </div>
            )}
          </div>
        </div>
      </TooltipContent>
    </Tooltip>
  );
}
