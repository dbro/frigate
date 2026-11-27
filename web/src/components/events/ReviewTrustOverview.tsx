import {
  MouseEvent as ReactMouseEvent,
  ReactNode,
  useCallback,
  useEffect,
  useMemo,
  useState,
} from "react";
import { createPortal } from "react-dom";
import axios from "axios";
import useSWR from "swr";
import { cn } from "@/lib/utils";
import { ReviewFilter } from "@/types/review";
import {
  GoldenClipSummary,
  ReviewClipMeta,
  ReviewFeedbackItem,
  ReviewOverviewBarSize,
  ReviewOverviewResponse,
  ReviewOverviewSeverity,
  SuppressedFpItem,
  TrustCategoryFilter,
  TrustOverviewBucket,
} from "@/types/reviewTrust";
import GoldenClipsDialog from "./GoldenClipsDialog";

const BAR_SIZE_OPTIONS: {
  value: ReviewOverviewBarSize;
  label: string;
  perBar: string;
  span: string;
  seconds: number;
}[] = [
  {
    value: "1m",
    label: "1m",
    perBar: "1 min per bar",
    span: "Last 30 minutes",
    seconds: 60,
  },
  {
    value: "10m",
    label: "10m",
    perBar: "10 min per bar",
    span: "Last 5 hours",
    seconds: 600,
  },
  {
    value: "1h",
    label: "1h",
    perBar: "1 hr per bar",
    span: "Last 30 hours",
    seconds: 3600,
  },
  {
    value: "4h",
    label: "4h",
    perBar: "4 hr per bar",
    span: "Last 5 days",
    seconds: 14400,
  },
  {
    value: "24h",
    label: "24h",
    perBar: "24 hr per bar",
    span: "Last 30 days",
    seconds: 86400,
  },
];

const NUM_BARS = 30;
const NICE_TIMING_STEPS = [2.5, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30];

function computeNiceTimingMax(rawMaxSec: number): number {
  const bounded = Math.max(2.5, rawMaxSec);
  for (const step of NICE_TIMING_STEPS) {
    if (bounded <= step) {
      return step;
    }
  }
  return Math.ceil(bounded / 5) * 5;
}

function formatTimingTick(sec: number): string {
  if (sec >= 10 && Math.abs(sec - Math.round(sec)) < 0.01) {
    return `${Math.round(sec)}s`;
  }
  return `${sec.toFixed(1)}s`;
}

type HoverState =
  | {
      kind: "bar";
      bucket: TrustOverviewBucket;
      x: number;
      y: number;
    }
  | {
      kind: "golden";
      bucket: TrustOverviewBucket;
      clip: GoldenClipSummary;
      x: number;
      y: number;
    }
  | {
      kind: "timing";
      bucket: TrustOverviewBucket;
      x: number;
      y: number;
    }
  | null;

function formatBucketTick(
  bucket: TrustOverviewBucket,
  barSize: ReviewOverviewBarSize,
): string {
  if (bucket.start_time > 0) {
    const date = new Date(bucket.start_time * 1000);
    if (barSize === "1m" || barSize === "10m" || barSize === "1h") {
      return `${String(date.getHours()).padStart(2, "0")}:${String(
        date.getMinutes(),
      ).padStart(2, "0")}`;
    }
    if (barSize === "4h") {
      const days = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
      return `${days[date.getDay()]} ${String(date.getHours()).padStart(2, "0")}h`;
    }
    return `${date.getMonth() + 1}/${date.getDate()}`;
  }

  const remaining = NUM_BARS - 1 - bucket.index;
  if (barSize === "1m") return `-${remaining}m`;
  if (barSize === "10m") return `-${remaining * 10}m`;
  if (barSize === "1h") return `-${remaining}h`;
  if (barSize === "4h") return `-${remaining * 4}h`;
  return `-${remaining}d`;
}

function formatBucketRange(bucket: TrustOverviewBucket): string {
  if (!bucket.start_time || !bucket.end_time) {
    return `Bar ${bucket.index + 1}`;
  }
  const start = new Date(bucket.start_time * 1000);
  const end = new Date(bucket.end_time * 1000);
  const fmt = (d: Date) =>
    `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
  return `${fmt(start)} – ${fmt(end)}`;
}

function createEmptyBuckets(
  barSize: ReviewOverviewBarSize,
): TrustOverviewBucket[] {
  const meta =
    BAR_SIZE_OPTIONS.find((opt) => opt.value === barSize) ??
    BAR_SIZE_OPTIONS[2];
  const nowSec = Math.floor(Date.now() / 1000);
  const alignedEnd = nowSec - (nowSec % meta.seconds) + meta.seconds;
  return Array.from({ length: NUM_BARS }, (_, index) => {
    const end = alignedEnd - (NUM_BARS - 1 - index) * meta.seconds;
    const start = end - meta.seconds;
    return {
      index,
      start_time: start,
      end_time: end,
      above: {
        alerts: 0,
        detections: 0,
        rescued_fn: 0,
        auto_suppressed_fp: 0,
        auto_misclassified: 0,
        background_clips: 0,
      },
      below: {
        fn: 0,
        fp: 0,
        misclassified: 0,
      },
      golden_clips: [],
      timing: {
        count: 0,
        avg_discovery_ms: 0,
        avg_tracking_ms: 0,
        avg_resource_busy_ms: 0,
        avg_confirmation_ms: 0,
        avg_zone_delay_ms: 0,
        avg_total_ms: 0,
        slowest_ms: 0,
        slowest_review_id: null,
      },
    };
  });
}

type ReviewTrustOverviewProps = {
  severity: ReviewOverviewSeverity;
  filter?: ReviewFilter;
  showReviewed: boolean;
  selectedTimeRange?: { start: number; end: number } | null;
  activeCategory?: TrustCategoryFilter | null;
  openGoldenDialogSignal?: number;
  onSelectCategory?: (category: TrustCategoryFilter | null) => void;
  onSelectTimeRange?: (start: number, end: number) => void;
  onOpenReviewId?: (reviewId: string) => void;
  onGoldenClipClick?: (clip: GoldenClipSummary) => void;
  onGoldenReviewIdsChange?: (reviewIds: Set<string>) => void;
  onFeedbackMapChange?: (
    feedbackByReviewId: Record<string, ReviewFeedbackItem>,
  ) => void;
  onReviewMetaMapChange?: (
    reviewMetaById: Record<string, ReviewClipMeta>,
  ) => void;
  onSuppressedFpItemsChange?: (items: SuppressedFpItem[]) => void;
  onUnreviewedFnCountChange?: (count: number) => void;
};

export default function ReviewTrustOverview({
  severity,
  filter,
  showReviewed,
  selectedTimeRange,
  activeCategory: controlledActiveCategory,
  openGoldenDialogSignal,
  onSelectCategory,
  onSelectTimeRange,
  onOpenReviewId,
  onGoldenClipClick,
  onGoldenReviewIdsChange,
  onFeedbackMapChange,
  onReviewMetaMapChange,
  onSuppressedFpItemsChange,
  onUnreviewedFnCountChange,
}: ReviewTrustOverviewProps) {
  const [barSize, setBarSize] = useState<ReviewOverviewBarSize>("1h");
  const [timingExpanded, setTimingExpanded] = useState(false);
  const [goldenDialogOpen, setGoldenDialogOpen] = useState(false);
  const [hover, setHover] = useState<HoverState>(null);
  const [internalActiveCategory, setInternalActiveCategory] =
    useState<TrustCategoryFilter | null>(null);

  const activeCategory =
    controlledActiveCategory !== undefined
      ? controlledActiveCategory
      : internalActiveCategory;

  const handleToggleCategory = useCallback(
    (category: TrustCategoryFilter | null) => {
      const next =
        category !== null && activeCategory === category ? null : category;
      if (onSelectCategory) {
        onSelectCategory(next);
      } else {
        setInternalActiveCategory(next);
      }
    },
    [activeCategory, onSelectCategory],
  );

  useEffect(() => {
    if (openGoldenDialogSignal && openGoldenDialogSignal > 0) {
      setGoldenDialogOpen(true);
    }
  }, [openGoldenDialogSignal]);

  const queryParams = useMemo(() => {
    const params: Record<string, string | number | boolean> = {
      severity,
      bar_size: barSize,
    };
    if (!showReviewed) {
      params.reviewed = 0;
    }
    if (filter?.cameras?.length) {
      params.cameras = filter.cameras.join(",");
    }
    if (filter?.labels?.length) {
      params.labels = filter.labels.join(",");
    }
    if (filter?.zones?.length) {
      params.zones = filter.zones.join(",");
    }
    if (filter?.sub_labels?.length) {
      params.sub_labels = filter.sub_labels.join(",");
    }
    if (filter?.min_score != null) {
      params.min_score = filter.min_score;
    }
    if (filter?.max_score != null) {
      params.max_score = filter.max_score;
    }
    return params;
  }, [severity, barSize, showReviewed, filter]);

  const { data } = useSWR<ReviewOverviewResponse>(
    ["review/overview", queryParams],
    ([path, params]) => axios.get(path, { params }).then((res) => res.data),
    {
      revalidateOnFocus: false,
      refreshInterval: 30000,
    },
  );

  useEffect(() => {
    if (data?.feedback_by_review_id && onFeedbackMapChange) {
      onFeedbackMapChange(data.feedback_by_review_id);
    }
  }, [data?.feedback_by_review_id, onFeedbackMapChange]);

  useEffect(() => {
    if (data?.review_meta_by_id && onReviewMetaMapChange) {
      onReviewMetaMapChange(data.review_meta_by_id);
    }
  }, [data?.review_meta_by_id, onReviewMetaMapChange]);

  useEffect(() => {
    if (onSuppressedFpItemsChange) {
      onSuppressedFpItemsChange(data?.suppressed_fp_items ?? []);
    }
  }, [data?.suppressed_fp_items, onSuppressedFpItemsChange]);

  useEffect(() => {
    if (
      typeof data?.unreviewed_fn_count === "number" &&
      onUnreviewedFnCountChange
    ) {
      onUnreviewedFnCountChange(data.unreviewed_fn_count);
    }
  }, [data?.unreviewed_fn_count, onUnreviewedFnCountChange]);

  const buckets = useMemo(() => {
    if (data?.buckets && data.buckets.length === NUM_BARS) {
      return data.buckets;
    }
    if (data?.buckets && data.buckets.length > 0) {
      const padded = createEmptyBuckets(barSize);
      data.buckets.forEach((b, idx) => {
        if (idx < NUM_BARS) {
          padded[idx] = b;
        }
      });
      return padded;
    }
    return createEmptyBuckets(barSize);
  }, [data?.buckets, barSize]);

  useEffect(() => {
    if (!onGoldenReviewIdsChange) {
      return;
    }
    const ids = new Set<string>();
    for (const bucket of buckets) {
      for (const clip of bucket.golden_clips ?? []) {
        if (clip.review_id) {
          ids.add(clip.review_id);
        }
      }
    }
    onGoldenReviewIdsChange(ids);
  }, [buckets, onGoldenReviewIdsChange]);

  const barMeta = useMemo(
    () =>
      BAR_SIZE_OPTIONS.find((opt) => opt.value === barSize) ??
      BAR_SIZE_OPTIONS[2],
    [barSize],
  );

  const isAlerts = severity === "alert";
  const goldenTotal = data?.golden_total ?? 0;

  // Chart geometry matching mockup (30 bars)
  const W = 1120;
  const H = 260;
  const padL = 44;
  const padR = 14;
  const padT = 30;
  const padB = 26;
  const plotW = W - padL - padR;
  const zeroY = 176;
  const upH = zeroY - padT;
  const downH = H - padB - zeroY;

  const { niceUp, niceDown } = useMemo(() => {
    let maxUpVal = 5;
    let maxDownVal = 2;

    for (const b of buckets) {
      const aboveAlerts = b.above.alerts ?? b.above.tp ?? 0;
      const aboveDetections = b.above.detections ?? b.above.tp ?? 0;
      const upSum = isAlerts
        ? aboveAlerts +
          (b.above.rescued_fn ?? 0) +
          (b.above.auto_misclassified ?? 0)
        : aboveDetections +
          (b.above.auto_suppressed_fp ?? 0) +
          (b.above.auto_misclassified ?? 0) +
          (b.above.background_clips ?? 0);
      const downSum = isAlerts
        ? (b.below.fp ?? 0) + (b.below.misclassified ?? 0)
        : (b.below.fn ?? 0) + (b.below.fp ?? 0) + (b.below.misclassified ?? 0);

      if (upSum > maxUpVal) maxUpVal = upSum;
      if (downSum > maxDownVal) maxDownVal = downSum;
    }

    return {
      niceUp: Math.max(5, Math.ceil(maxUpVal / 5) * 5),
      niceDown: Math.max(2, Math.ceil(maxDownVal / 2) * 2),
    };
  }, [buckets, isAlerts]);

  const slotW = plotW / NUM_BARS;
  const barW = Math.max(12, slotW - 10);

  // Timing chart geometry
  const timingH = 145;
  const timingPadT = 16;
  const timingPadB = 24;
  const timingPlotH = timingH - timingPadT - timingPadB;
  const maxTimingSec = useMemo(() => {
    let rawMax = 2.5;
    for (const b of buckets) {
      const discSec = (b.timing.avg_discovery_ms ?? 0) / 1000;
      const trkSec = (b.timing.avg_tracking_ms ?? 0) / 1000;
      const confSec = (b.timing.avg_confirmation_ms ?? 0) / 1000;
      const zoneSec = isAlerts ? (b.timing.avg_zone_delay_ms ?? 0) / 1000 : 0;
      const avgTotalSec = discSec + trkSec + confSec + zoneSec;
      const slowestSec = (b.timing.slowest_ms ?? 0) / 1000;
      rawMax = Math.max(rawMax, slowestSec, avgTotalSec);
    }
    return computeNiceTimingMax(rawMax);
  }, [buckets, isAlerts]);

  const timingTicks = useMemo(
    () => [0, maxTimingSec / 2, maxTimingSec],
    [maxTimingSec],
  );

  const handleBarMove = useCallback(
    (evt: ReactMouseEvent<SVGGElement>, bucket: TrustOverviewBucket) => {
      setHover({
        kind: "bar",
        bucket,
        x: evt.clientX,
        y: evt.clientY,
      });
    },
    [],
  );

  const handleGoldenMove = useCallback(
    (
      evt: ReactMouseEvent<SVGGElement>,
      bucket: TrustOverviewBucket,
      clip: GoldenClipSummary,
    ) => {
      evt.stopPropagation();
      setHover({
        kind: "golden",
        bucket,
        clip,
        x: evt.clientX,
        y: evt.clientY,
      });
    },
    [],
  );

  const handleTimingMove = useCallback(
    (evt: ReactMouseEvent<SVGGElement>, bucket: TrustOverviewBucket) => {
      setHover({
        kind: "timing",
        bucket,
        x: evt.clientX,
        y: evt.clientY,
      });
    },
    [],
  );

  const clearHover = useCallback(() => setHover(null), []);

  const tooltipStyle = useMemo(() => {
    if (!hover || typeof window === "undefined") {
      return undefined;
    }
    const pad = 14;
    const width = hover.kind === "golden" ? 260 : 210;
    const height = hover.kind === "golden" ? 108 : 165;
    let left = hover.x + pad;
    let top = hover.y - height - 8;
    if (left + width > window.innerWidth - 12) {
      left = hover.x - width - pad;
    }
    if (top < 12) {
      top = hover.y + pad;
    }
    return { left, top };
  }, [hover]);

  return (
    <div className="mb-3 w-full space-y-2 px-1 md:px-2">
      {/* Main Trust Overview Card */}
      <div className="rounded-xl border border-border bg-card p-3 md:p-4">
        {/* Card Header: Title + Legend + Bar Size + Golden Clips Button */}
        <div className="mb-2.5 flex flex-wrap items-start justify-between gap-3">
          <div>
            <div className="flex flex-wrap items-center gap-2">
              <h2 className="text-sm font-semibold text-foreground md:text-base">
                {isAlerts ? "Alerts Overview" : "Detections Overview"}
              </h2>
              <button
                type="button"
                onClick={() => setGoldenDialogOpen(true)}
                className="inline-flex items-center gap-1.5 rounded-lg border border-amber-500/40 bg-amber-500/10 px-2.5 py-1 text-xs font-medium text-amber-400 transition hover:bg-amber-500/20"
              >
                <span>🏆</span>
                <span>
                  {goldenTotal}{" "}
                  {goldenTotal === 1 ? "Golden Clip" : "Golden Clips"}
                </span>
                <span className="text-[10px] text-muted-foreground">▾</span>
              </button>
            </div>
            <p className="mt-0.5 text-xs text-muted-foreground">
              30 bars · {barMeta.perBar} ({barMeta.span}) · Above = Automatic,
              Below = Human-Edited
            </p>
          </div>

          <div className="flex flex-col items-start gap-2 sm:items-end">
            {/* Bar size selector */}
            <div className="flex items-center gap-2">
              <span className="text-xs text-muted-foreground">
                Bar size (30 bars):
              </span>
              <div className="inline-flex rounded-lg border border-border bg-background p-0.5">
                {BAR_SIZE_OPTIONS.map((opt) => {
                  const active = barSize === opt.value;
                  return (
                    <button
                      key={opt.value}
                      type="button"
                      onClick={() => setBarSize(opt.value)}
                      className={cn(
                        "rounded-md px-2 py-0.5 text-xs transition",
                        active
                          ? "bg-primary font-medium text-primary-foreground"
                          : "text-muted-foreground hover:text-foreground",
                      )}
                    >
                      {opt.label}
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Two-row Clickable Legend (click any item to filter) */}
            {(() => {
              const renderLegendBtn = (
                category: TrustCategoryFilter,
                label: string,
                swatch: ReactNode,
                extraClass?: string,
              ) => {
                const isActive = activeCategory === category;
                const isDimmed =
                  activeCategory !== null && activeCategory !== category;
                return (
                  <button
                    key={category}
                    type="button"
                    aria-pressed={isActive}
                    title={
                      isActive ? `Clear ${label} filter` : `Filter by ${label}`
                    }
                    onClick={() => handleToggleCategory(category)}
                    className={cn(
                      "inline-flex cursor-pointer items-center gap-1 rounded-md px-1.5 py-0.5 transition",
                      isActive
                        ? "bg-primary/15 font-semibold text-foreground ring-1 ring-primary/60"
                        : isDimmed
                          ? "opacity-45 hover:bg-muted/70 hover:opacity-100"
                          : "hover:bg-muted/70",
                      extraClass,
                    )}
                  >
                    {swatch}
                    <span>{label}</span>
                  </button>
                );
              };

              return isAlerts ? (
                <div className="flex flex-col items-start gap-1 text-[11px] sm:items-end">
                  <div className="flex flex-wrap items-center gap-1.5">
                    <span className="mr-0.5 font-medium text-muted-foreground">
                      Above (Auto):
                    </span>
                    {renderLegendBtn(
                      "alerts",
                      "Alerts",
                      <span className="size-2.5 rounded-sm bg-blue-500" />,
                    )}
                    {renderLegendBtn(
                      "rescued_fn",
                      "Rescued FN",
                      <span className="size-2.5 rounded-sm bg-teal-400" />,
                    )}
                    {renderLegendBtn(
                      "auto_misclassified",
                      "Misclassified",
                      <span className="size-2.5 rounded-sm bg-yellow-400" />,
                    )}
                  </div>
                  <div className="flex flex-wrap items-center gap-1.5">
                    <span className="mr-0.5 font-medium text-muted-foreground">
                      Below (Human-Edited):
                    </span>
                    {renderLegendBtn(
                      "fp",
                      "FP",
                      <span className="size-2.5 rounded-sm bg-rose-500" />,
                    )}
                    {renderLegendBtn(
                      "human_misclassified",
                      "Misclassified",
                      <span className="size-2.5 rounded-sm bg-yellow-400" />,
                    )}
                    {renderLegendBtn(
                      "golden",
                      "Golden Clip",
                      <span>🏆</span>,
                      "ml-0.5 text-amber-500 dark:text-amber-400",
                    )}
                    {activeCategory !== null && (
                      <button
                        type="button"
                        onClick={() => handleToggleCategory(null)}
                        className="ml-1 rounded-md border border-border px-1.5 py-0.5 text-[10px] font-medium text-muted-foreground transition hover:bg-muted hover:text-foreground"
                      >
                        Clear ×
                      </button>
                    )}
                  </div>
                </div>
              ) : (
                <div className="flex flex-col items-start gap-1 text-[11px] sm:items-end">
                  <div className="flex flex-wrap items-center gap-1.5">
                    <span className="mr-0.5 font-medium text-muted-foreground">
                      Above (Auto):
                    </span>
                    {renderLegendBtn(
                      "detections",
                      "Detections",
                      <span className="size-2.5 rounded-sm bg-slate-500" />,
                    )}
                    {renderLegendBtn(
                      "auto_suppressed_fp",
                      "Auto-Suppressed FP",
                      <span className="size-2.5 rounded-sm bg-rose-400/80" />,
                    )}
                    {renderLegendBtn(
                      "auto_misclassified",
                      "Misclassified",
                      <span className="size-2.5 rounded-sm bg-yellow-400" />,
                    )}
                    {renderLegendBtn(
                      "background_clips",
                      "Background Clips",
                      <span className="size-2.5 rounded-sm border border-foreground/60 bg-transparent" />,
                    )}
                  </div>
                  <div className="flex flex-wrap items-center gap-1.5">
                    <span className="mr-0.5 font-medium text-muted-foreground">
                      Below (Human-Edited):
                    </span>
                    {renderLegendBtn(
                      "fn",
                      "FN",
                      <span className="size-2.5 rounded-sm bg-red-600" />,
                    )}
                    {renderLegendBtn(
                      "fp",
                      "FP",
                      <span className="size-2.5 rounded-sm bg-rose-500" />,
                    )}
                    {renderLegendBtn(
                      "human_misclassified",
                      "Misclassified",
                      <span className="size-2.5 rounded-sm bg-yellow-400" />,
                    )}
                    {renderLegendBtn(
                      "golden",
                      "Golden Clip",
                      <span>🏆</span>,
                      "ml-0.5 text-amber-500 dark:text-amber-400",
                    )}
                    {activeCategory !== null && (
                      <button
                        type="button"
                        onClick={() => handleToggleCategory(null)}
                        className="ml-1 rounded-md border border-border px-1.5 py-0.5 text-[10px] font-medium text-muted-foreground transition hover:bg-muted hover:text-foreground"
                      >
                        Clear ×
                      </button>
                    )}
                  </div>
                </div>
              );
            })()}
          </div>
        </div>

        {/* Diverging Stacked Bar SVG */}
        <div className="relative w-full">
          <svg
            className="w-full select-none overflow-visible"
            viewBox={`0 0 ${W} ${H}`}
          >
            {/* Above zero gridlines & unsigned Y-axis labels */}
            {[0, niceUp * 0.5, niceUp].map((val, idx) => {
              const y = zeroY - (val / niceUp) * upH;
              return (
                <g key={`up-grid-${idx}`}>
                  <line
                    x1={padL}
                    x2={W - padR}
                    y1={y}
                    y2={y}
                    stroke="currentColor"
                    strokeOpacity={val === 0 ? 0.45 : 0.1}
                    strokeWidth={val === 0 ? 1.5 : 1}
                  />
                  <text
                    x={padL - 8}
                    y={y + 4}
                    textAnchor="end"
                    fill="currentColor"
                    fillOpacity={0.65}
                    fontSize={11}
                  >
                    {Math.round(val)}
                  </text>
                </g>
              );
            })}

            {/* Below zero gridlines & UNSIGNED positive Y-axis labels (no minus sign!) */}
            {[niceDown * 0.5, niceDown].map((val, idx) => {
              const y = zeroY + (val / niceDown) * downH;
              return (
                <g key={`down-grid-${idx}`}>
                  <line
                    x1={padL}
                    x2={W - padR}
                    y1={y}
                    y2={y}
                    stroke="currentColor"
                    strokeOpacity={0.1}
                    strokeDasharray="3 3"
                  />
                  <text
                    x={padL - 8}
                    y={y + 4}
                    textAnchor="end"
                    fill="currentColor"
                    fillOpacity={0.65}
                    fontSize={11}
                  >
                    {Math.round(val)}
                  </text>
                </g>
              );
            })}

            {/* 30 Bars */}
            {buckets.map((bucket, idx) => {
              const cx = padL + idx * slotW + slotW / 2;
              const bx = cx - barW / 2;
              const isSelected =
                selectedTimeRange != null &&
                selectedTimeRange.start === bucket.start_time &&
                selectedTimeRange.end === bucket.end_time;

              const upSegments: {
                category: TrustCategoryFilter;
                val: number;
                color: string;
                outline?: string;
              }[] = isAlerts
                ? [
                    {
                      category: "alerts",
                      val: bucket.above.alerts ?? bucket.above.tp ?? 0,
                      color: "#3b82f6",
                    },
                    {
                      category: "rescued_fn",
                      val: bucket.above.rescued_fn ?? 0,
                      color: "#2dd4bf",
                    },
                    {
                      category: "auto_misclassified",
                      val: bucket.above.auto_misclassified ?? 0,
                      color: "#facc15",
                    },
                  ]
                : [
                    {
                      category: "detections",
                      val: bucket.above.detections ?? bucket.above.tp ?? 0,
                      color: "#64748b",
                    },
                    {
                      category: "auto_suppressed_fp",
                      val: bucket.above.auto_suppressed_fp ?? 0,
                      color: "#e17087",
                    },
                    {
                      category: "auto_misclassified",
                      val: bucket.above.auto_misclassified ?? 0,
                      color: "#facc15",
                    },
                    {
                      category: "background_clips",
                      val: bucket.above.background_clips ?? 0,
                      color: "transparent",
                      outline: "currentColor",
                    },
                  ];

              let curUpY = zeroY;
              const upRects = upSegments
                .filter((seg) => seg.val > 0)
                .map((seg, segIdx) => {
                  const rawH = Math.max(1.5, (seg.val / niceUp) * upH);
                  const clampedTop = Math.max(padT, curUpY - rawH);
                  const h = Math.max(0, curUpY - clampedTop);
                  curUpY = clampedTop;
                  const dimmed =
                    activeCategory !== null && activeCategory !== seg.category;
                  return (
                    <rect
                      key={`up-${segIdx}`}
                      x={bx}
                      y={curUpY}
                      width={barW}
                      height={h}
                      rx={1.5}
                      fill={seg.color}
                      fillOpacity={seg.outline ? undefined : dimmed ? 0.16 : 1}
                      stroke={seg.outline}
                      strokeOpacity={
                        seg.outline ? (dimmed ? 0.16 : 0.75) : undefined
                      }
                      strokeWidth={seg.outline ? 1.5 : undefined}
                    />
                  );
                });
              const topOfBarY = curUpY;

              const downSegments: {
                category: TrustCategoryFilter;
                val: number;
                color: string;
              }[] = isAlerts
                ? [
                    { category: "fp", val: bucket.below.fp, color: "#f43f5e" },
                    {
                      category: "human_misclassified",
                      val: bucket.below.misclassified,
                      color: "#facc15",
                    },
                  ]
                : [
                    { category: "fn", val: bucket.below.fn, color: "#dc2626" },
                    { category: "fp", val: bucket.below.fp, color: "#f43f5e" },
                    {
                      category: "human_misclassified",
                      val: bucket.below.misclassified,
                      color: "#facc15",
                    },
                  ];

              let curDownY = zeroY;
              const downRects = downSegments
                .filter((seg) => seg.val > 0)
                .map((seg, segIdx) => {
                  const rawH = Math.max(2, (seg.val / niceDown) * downH);
                  const clampedBottom = Math.min(H - padB, curDownY + rawH);
                  const h = Math.max(0, clampedBottom - curDownY);
                  const rectY = curDownY;
                  curDownY = clampedBottom;
                  const dimmed =
                    activeCategory !== null && activeCategory !== seg.category;
                  return (
                    <rect
                      key={`down-${segIdx}`}
                      x={bx}
                      y={rectY}
                      width={barW}
                      height={h}
                      rx={1.5}
                      fill={seg.color}
                      fillOpacity={dimmed ? 0.16 : 1}
                    />
                  );
                });

              const hitTop = Math.min(zeroY - 12, Math.max(padT, topOfBarY));
              const hitBottom = Math.max(zeroY + 12, curDownY);
              const firstGolden = bucket.golden_clips?.[0];
              const pinY = Math.max(14, topOfBarY - 12);
              const goldenDimmed =
                activeCategory !== null && activeCategory !== "golden";

              return (
                <g key={bucket.index}>
                  {/* Selected range highlight */}
                  {isSelected && (
                    <rect
                      x={bx - 3}
                      y={padT - 4}
                      width={barW + 6}
                      height={H - padB - padT + 4}
                      rx={4}
                      fill="currentColor"
                      fillOpacity={0.08}
                      stroke="#3b82f6"
                      strokeOpacity={0.6}
                      strokeWidth={1}
                    />
                  )}

                  {/* Bar group: ONLY describes this bar's counts */}
                  <g
                    className="cursor-pointer transition-opacity hover:opacity-85"
                    onMouseMove={(e) => handleBarMove(e, bucket)}
                    onMouseLeave={clearHover}
                    onClick={() =>
                      onSelectTimeRange?.(bucket.start_time, bucket.end_time)
                    }
                  >
                    {upRects}
                    {downRects}
                    <rect
                      x={bx - 2}
                      y={hitTop}
                      width={barW + 4}
                      height={Math.max(24, hitBottom - hitTop)}
                      fill="transparent"
                    />
                  </g>

                  {/* Separate Trophy Pin group: ONLY describes Golden Clip */}
                  {firstGolden && (
                    <g
                      className="cursor-pointer transition-transform hover:-translate-y-0.5"
                      opacity={goldenDimmed ? 0.2 : 1}
                      onMouseMove={(e) =>
                        handleGoldenMove(e, bucket, firstGolden)
                      }
                      onMouseLeave={clearHover}
                      onClick={(e) => {
                        e.stopPropagation();
                        if (onGoldenClipClick) {
                          onGoldenClipClick(firstGolden);
                        } else if (firstGolden.review_id && onOpenReviewId) {
                          onOpenReviewId(firstGolden.review_id);
                        } else {
                          setGoldenDialogOpen(true);
                        }
                      }}
                    >
                      <circle
                        cx={cx}
                        cy={pinY}
                        r={10}
                        fill="#f59e0b"
                        fillOpacity={0.18}
                        stroke="#f59e0b"
                        strokeWidth={1.2}
                      />
                      <text
                        x={cx}
                        y={pinY + 4}
                        textAnchor="middle"
                        fontSize={11}
                      >
                        🏆
                      </text>
                    </g>
                  )}

                  {/* X-axis tick label every 5 bars + last bar */}
                  {(idx % 5 === 0 || idx === NUM_BARS - 1) && (
                    <text
                      x={cx}
                      y={H - 6}
                      textAnchor="middle"
                      fill="currentColor"
                      fillOpacity={0.6}
                      fontSize={11}
                    >
                      {formatBucketTick(bucket, barSize)}
                    </text>
                  )}
                </g>
              );
            })}
          </svg>
        </div>
      </div>

      {/* Collapsible Alert / Detection Timing Accordion (collapsed by default) */}
      <div className="overflow-hidden rounded-xl border border-border bg-card">
        <button
          type="button"
          onClick={() => setTimingExpanded((prev) => !prev)}
          aria-expanded={timingExpanded}
          className="flex w-full items-center justify-between px-4 py-2.5 text-left transition hover:bg-background/40"
        >
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-xs text-muted-foreground">
              {timingExpanded ? "▾" : "▸"}
            </span>
            <span className="text-sm font-medium text-foreground">
              {isAlerts ? "Alert Timing" : "Detection Timing"}
            </span>
            <span className="text-xs text-muted-foreground">
              {isAlerts
                ? "(Alerted events · Avg Discovery + Tracking + Confirmation + Zone vs 2.0s Goal)"
                : "(Detected events · Avg Discovery + Tracking + Confirmation vs 2.0s Goal)"}
            </span>
          </div>
          <div className="hidden items-center gap-3 text-xs text-muted-foreground sm:flex">
            <span className="inline-flex items-center gap-1">
              <span className="size-2.5 rounded-sm bg-slate-400" />
              Discovery
            </span>
            <span className="inline-flex items-center gap-1">
              <span className="size-2.5 rounded-sm bg-amber-500" />
              Tracking
            </span>
            <span className="inline-flex items-center gap-1">
              <span className="size-2.5 rounded-sm bg-violet-500" />
              Confirmation
            </span>
            {isAlerts && (
              <span className="inline-flex items-center gap-1">
                <span className="size-2.5 rounded-sm bg-cyan-500" />
                Zone / Alert
              </span>
            )}
            <span className="inline-flex items-center gap-1">
              <span className="size-2 rounded-full bg-cyan-400" />
              Slowest
            </span>
          </div>
        </button>

        {timingExpanded && (
          <div className="border-t border-border px-4 pb-4 pt-2">
            <svg
              className="w-full select-none overflow-visible"
              viewBox={`0 0 ${W} ${timingH}`}
            >
              {/* Dynamic Y-axis gridlines & labels */}
              {timingTicks.map((sec, tIdx) => {
                const y =
                  timingPadT + timingPlotH - (sec / maxTimingSec) * timingPlotH;
                return (
                  <g key={`timing-grid-${tIdx}`}>
                    <line
                      x1={padL}
                      x2={W - padR}
                      y1={y}
                      y2={y}
                      stroke="currentColor"
                      strokeOpacity={sec === 0 ? 0.35 : 0.12}
                    />
                    <text
                      x={padL - 8}
                      y={y + 4}
                      textAnchor="end"
                      fill="currentColor"
                      fillOpacity={0.7}
                      fontSize={11}
                    >
                      {formatTimingTick(sec)}
                    </text>
                  </g>
                );
              })}

              {/* 2.0s Goal dashed reference line & label */}
              {2.0 <= maxTimingSec &&
                (() => {
                  const goalY =
                    timingPadT +
                    timingPlotH -
                    (2.0 / maxTimingSec) * timingPlotH;
                  return (
                    <g>
                      <line
                        x1={padL}
                        x2={W - padR}
                        y1={goalY}
                        y2={goalY}
                        stroke="currentColor"
                        strokeOpacity={0.5}
                        strokeDasharray="5 4"
                      />
                      <text
                        x={W - padR - 4}
                        y={Math.max(timingPadT + 10, goalY - 4)}
                        textAnchor="end"
                        fill="currentColor"
                        fillOpacity={0.75}
                        fontSize={10}
                      >
                        2.0s Goal
                      </text>
                    </g>
                  );
                })()}

              {!buckets.some((b) => (b.timing?.count ?? 0) > 0) && (
                <text
                  x={padL + plotW / 2}
                  y={timingPadT + timingPlotH / 2 + 4}
                  textAnchor="middle"
                  fill="currentColor"
                  fillOpacity={0.55}
                  fontSize={12}
                >
                  {isAlerts
                    ? "No alert timing samples in this window yet (recorded live for new alerts)"
                    : "No detection timing samples in this window yet (recorded live for new detections)"}
                </text>
              )}

              {buckets.map((bucket, idx) => {
                const cx = padL + idx * slotW + slotW / 2;
                const bx = cx - barW / 2;
                const hasTiming = bucket.timing.count > 0;

                const discSec = (bucket.timing.avg_discovery_ms ?? 0) / 1000;
                const trkSec = (bucket.timing.avg_tracking_ms ?? 0) / 1000;
                const confSec = (bucket.timing.avg_confirmation_ms ?? 0) / 1000;
                const zoneSec = isAlerts
                  ? (bucket.timing.avg_zone_delay_ms ?? 0) / 1000
                  : 0;
                const slowestSec = (bucket.timing.slowest_ms ?? 0) / 1000;

                let curY = timingPadT + timingPlotH;
                const segs = isAlerts
                  ? [
                      { sec: discSec, color: "#64748b" },
                      { sec: trkSec, color: "#f59e0b" },
                      { sec: confSec, color: "#8b5cf6" },
                      { sec: zoneSec, color: "#06b6d4" },
                    ]
                  : [
                      { sec: discSec, color: "#64748b" },
                      { sec: trkSec, color: "#f59e0b" },
                      { sec: confSec, color: "#8b5cf6" },
                    ];

                const slowestY = Math.max(
                  timingPadT + 3,
                  Math.min(
                    timingPadT + timingPlotH - 3,
                    timingPadT +
                      timingPlotH -
                      (Math.min(maxTimingSec, slowestSec) / maxTimingSec) *
                        timingPlotH,
                  ),
                );

                return (
                  <g key={`timing-${bucket.index}`}>
                    {hasTiming && (
                      <g
                        className="cursor-pointer transition-opacity hover:opacity-85"
                        onMouseMove={(e) => handleTimingMove(e, bucket)}
                        onMouseLeave={clearHover}
                        onClick={() => {
                          if (
                            bucket.timing.slowest_review_id &&
                            onOpenReviewId
                          ) {
                            onOpenReviewId(bucket.timing.slowest_review_id);
                          }
                        }}
                      >
                        {segs.map((seg, sIdx) => {
                          const rawH = Math.max(
                            0,
                            (seg.sec / maxTimingSec) * timingPlotH,
                          );
                          const clampedTop = Math.max(timingPadT, curY - rawH);
                          const h = Math.max(0, curY - clampedTop);
                          curY = clampedTop;
                          return h > 0 ? (
                            <rect
                              key={sIdx}
                              x={bx}
                              y={curY}
                              width={barW}
                              height={h}
                              rx={1}
                              fill={seg.color}
                            />
                          ) : null;
                        })}
                        <circle cx={cx} cy={slowestY} r={3.5} fill="#22d3ee" />
                        <rect
                          x={bx - 2}
                          y={timingPadT}
                          width={barW + 4}
                          height={timingPlotH}
                          fill="transparent"
                        />
                      </g>
                    )}

                    {(idx % 5 === 0 || idx === NUM_BARS - 1) && (
                      <text
                        x={cx}
                        y={timingH - 5}
                        textAnchor="middle"
                        fill="currentColor"
                        fillOpacity={0.6}
                        fontSize={11}
                      >
                        {formatBucketTick(bucket, barSize)}
                      </text>
                    )}
                  </g>
                );
              })}
            </svg>
          </div>
        )}
      </div>

      {/* Strictly separated hover tooltips rendered via portal */}
      {hover &&
        tooltipStyle &&
        typeof document !== "undefined" &&
        createPortal(
          hover.kind === "golden" ? (
            <div
              className="pointer-events-none fixed z-50 w-[260px] rounded-lg border border-amber-500/50 bg-card p-2.5 text-xs shadow-xl"
              style={tooltipStyle}
            >
              <div className="flex items-center gap-2.5">
                <div className="flex h-11 w-14 shrink-0 items-center justify-center rounded border border-border bg-muted text-lg">
                  🏆
                </div>
                <div className="min-w-0">
                  <div className="font-semibold text-amber-500 dark:text-amber-400">
                    Golden Clip · {hover.clip.id}
                  </div>
                  <div className="truncate font-medium text-foreground">
                    {hover.clip.camera} · {formatBucketRange(hover.bucket)}
                  </div>
                  <div className="truncate text-muted-foreground">
                    {hover.clip.condition_tags.join(", ") || "Untagged"}
                    {hover.clip.object_label
                      ? ` · ${hover.clip.object_label}`
                      : ""}
                  </div>
                  <div className="mt-0.5 text-blue-500 dark:text-blue-400">
                    Kind: {hover.clip.kind} · Click to open →
                  </div>
                </div>
              </div>
            </div>
          ) : hover.kind === "bar" ? (
            <div
              className="pointer-events-none fixed z-50 min-w-[200px] rounded-lg border border-border bg-card px-3 py-2 text-xs shadow-xl"
              style={tooltipStyle}
            >
              <div className="mb-1.5 border-b border-border pb-1 font-semibold">
                {formatBucketRange(hover.bucket)} ·{" "}
                {isAlerts ? "Alerts Bar" : "Detections Bar"}
              </div>
              {isAlerts ? (
                <div className="space-y-0.5">
                  <div className="flex justify-between gap-4">
                    <span className="text-blue-500 dark:text-blue-400">
                      Alerts (Auto)
                    </span>
                    <span className="font-mono font-semibold">
                      {hover.bucket.above.alerts}
                    </span>
                  </div>
                  <div className="flex justify-between gap-4">
                    <span className="text-teal-500 dark:text-teal-400">
                      Rescued FN (Auto)
                    </span>
                    <span className="font-mono font-semibold">
                      {hover.bucket.above.rescued_fn}
                    </span>
                  </div>
                  <div className="flex justify-between gap-4">
                    <span className="text-yellow-600 dark:text-yellow-400">
                      Misclassified (Auto)
                    </span>
                    <span className="font-mono font-semibold">
                      {hover.bucket.above.auto_misclassified}
                    </span>
                  </div>
                  <div className="my-1 border-t border-border" />
                  <div className="flex justify-between gap-4">
                    <span className="text-rose-500 dark:text-rose-400">
                      FP (Human-Edited)
                    </span>
                    <span className="font-mono font-semibold">
                      {hover.bucket.below.fp}
                    </span>
                  </div>
                  <div className="flex justify-between gap-4">
                    <span className="text-yellow-600 dark:text-yellow-400">
                      Misclassified (Human)
                    </span>
                    <span className="font-mono font-semibold">
                      {hover.bucket.below.misclassified}
                    </span>
                  </div>
                </div>
              ) : (
                <div className="space-y-0.5">
                  <div className="flex justify-between gap-4">
                    <span className="text-muted-foreground">
                      Detections (Auto)
                    </span>
                    <span className="font-mono font-semibold">
                      {hover.bucket.above.detections}
                    </span>
                  </div>
                  <div className="flex justify-between gap-4">
                    <span className="text-rose-500 dark:text-rose-300">
                      Auto-Suppressed FP
                    </span>
                    <span className="font-mono font-semibold">
                      {hover.bucket.above.auto_suppressed_fp}
                    </span>
                  </div>
                  <div className="flex justify-between gap-4">
                    <span className="text-yellow-600 dark:text-yellow-400">
                      Misclassified (Auto)
                    </span>
                    <span className="font-mono font-semibold">
                      {hover.bucket.above.auto_misclassified}
                    </span>
                  </div>
                  <div className="flex justify-between gap-4">
                    <span className="text-foreground">□ Background Clips</span>
                    <span className="font-mono font-semibold">
                      {hover.bucket.above.background_clips}
                    </span>
                  </div>
                  <div className="my-1 border-t border-border" />
                  <div className="flex justify-between gap-4">
                    <span className="text-red-500 dark:text-red-400">
                      FN (Human-Edited)
                    </span>
                    <span className="font-mono font-semibold">
                      {hover.bucket.below.fn}
                    </span>
                  </div>
                  <div className="flex justify-between gap-4">
                    <span className="text-rose-500 dark:text-rose-400">
                      FP (Human-Edited)
                    </span>
                    <span className="font-mono font-semibold">
                      {hover.bucket.below.fp}
                    </span>
                  </div>
                  <div className="flex justify-between gap-4">
                    <span className="text-yellow-600 dark:text-yellow-400">
                      Misclassified (Human)
                    </span>
                    <span className="font-mono font-semibold">
                      {hover.bucket.below.misclassified}
                    </span>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div
              className="pointer-events-none fixed z-50 min-w-[205px] rounded-lg border border-border bg-card px-3 py-2 text-xs shadow-xl"
              style={tooltipStyle}
            >
              <div className="mb-1 border-b border-border pb-1 font-semibold">
                {formatBucketRange(hover.bucket)} ·{" "}
                {isAlerts ? "Alert Timing" : "Detection Timing"} (
                {hover.bucket.timing.segment_count != null &&
                hover.bucket.timing.segment_count > hover.bucket.timing.count
                  ? `${hover.bucket.timing.count} of ${hover.bucket.timing.segment_count}`
                  : hover.bucket.timing.count}
                )
              </div>
              <div className="space-y-0.5">
                <div className="flex justify-between gap-4">
                  <span className="text-muted-foreground">
                    Discovery (Reverse Norfair)
                  </span>
                  <span className="font-mono">
                    {(
                      (hover.bucket.timing.avg_discovery_ms ?? 0) / 1000
                    ).toFixed(2)}
                    s
                  </span>
                </div>
                <div className="flex justify-between gap-4">
                  <span className="text-amber-600 dark:text-amber-400">
                    Tracking
                  </span>
                  <span className="font-mono">
                    {(
                      (hover.bucket.timing.avg_tracking_ms ?? 0) / 1000
                    ).toFixed(2)}
                    s
                  </span>
                </div>
                <div className="flex justify-between gap-4">
                  <span className="text-violet-600 dark:text-violet-400">
                    Confirmation
                  </span>
                  <span className="font-mono">
                    {(
                      (hover.bucket.timing.avg_confirmation_ms ?? 0) / 1000
                    ).toFixed(2)}
                    s
                  </span>
                </div>
                {isAlerts &&
                  (hover.bucket.timing.avg_zone_delay_ms ?? 0) > 0 && (
                    <div className="flex justify-between gap-4">
                      <span className="text-cyan-600 dark:text-cyan-400">
                        Zone / Alert
                      </span>
                      <span className="font-mono">
                        {(
                          (hover.bucket.timing.avg_zone_delay_ms ?? 0) / 1000
                        ).toFixed(2)}
                        s
                      </span>
                    </div>
                  )}
                <div className="my-1 border-t border-border" />
                <div className="flex justify-between gap-4 font-semibold">
                  <span>Average Total</span>
                  <span className="font-mono">
                    {(
                      ((hover.bucket.timing.avg_discovery_ms ?? 0) +
                        (hover.bucket.timing.avg_tracking_ms ?? 0) +
                        (hover.bucket.timing.avg_confirmation_ms ?? 0) +
                        (isAlerts
                          ? (hover.bucket.timing.avg_zone_delay_ms ?? 0)
                          : 0)) /
                      1000
                    ).toFixed(2)}
                    s
                  </span>
                </div>
                <div className="flex justify-between gap-4 text-cyan-600 dark:text-cyan-400">
                  <span>
                    {isAlerts ? "Slowest Alert" : "Slowest Detection"}
                  </span>
                  <span className="font-mono">
                    {((hover.bucket.timing.slowest_ms ?? 0) / 1000).toFixed(2)}s
                  </span>
                </div>
                {hover.bucket.timing.slowest_review_id && (
                  <div className="pt-0.5 text-[11px] text-cyan-600 dark:text-cyan-300">
                    Click to open slowest review →
                  </div>
                )}
              </div>
            </div>
          ),
          document.body,
        )}

      <GoldenClipsDialog
        open={goldenDialogOpen}
        onOpenChange={setGoldenDialogOpen}
        onSelectClip={(clip) => {
          if (onGoldenClipClick) {
            onGoldenClipClick(clip);
          } else if (clip.review_id && onOpenReviewId) {
            onOpenReviewId(clip.review_id);
          }
        }}
      />
    </div>
  );
}
