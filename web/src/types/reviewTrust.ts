export type ReviewOverviewBarSize = "1m" | "10m" | "1h" | "4h" | "24h";

export type ReviewOverviewSeverity = "alert" | "detection";

export type ReviewFeedbackVerdict =
  | "good"
  | "fp"
  | "fn"
  | "misclassified"
  | "clear";

export interface ReviewFeedbackItem {
  review_id?: string;
  verdict: ReviewFeedbackVerdict;
  corrected_label?: string | null;
  reason?: string | null;
  updated_at?: number;
}

export interface ReviewFeedbackBody {
  verdict: ReviewFeedbackVerdict;
  corrected_label?: string | null;
  reason?: string | null;
}

export interface GoldenClipSummary {
  id: string;
  review_id?: string | null;
  camera: string;
  kind: string;
  condition_tags: string[];
  object_label?: string | null;
  notes?: string | null;
  start_time?: number;
  end_time?: number;
  created_at?: number;
  file_path?: string;
  export_path?: string;
  thumb_path?: string | null;
}

export interface CreateGoldenClipBody {
  kind: string;
  condition_tags: string[];
  object_label?: string | null;
  notes?: string | null;
}

export interface GoldenClipsResponse {
  total: number;
  coverage: Record<string, number>;
  clips: GoldenClipSummary[];
}

export interface TrustOverviewBucketAbove {
  alerts: number;
  detections: number;
  rescued_fn: number;
  auto_suppressed_fp: number;
  auto_misclassified: number;
  background_clips: number;
}

export interface TrustOverviewBucketBelow {
  fn: number;
  fp: number;
  misclassified: number;
}

export interface TrustOverviewBucketTiming {
  count: number;
  avg_discovery_ms: number;
  avg_resource_busy_ms: number;
  avg_confirmation_ms: number;
  avg_total_ms: number;
  slowest_ms: number;
  slowest_review_id: string | null;
}

export interface TrustOverviewBucket {
  index: number;
  start_time: number;
  end_time: number;
  above: TrustOverviewBucketAbove;
  below: TrustOverviewBucketBelow;
  golden_clips: GoldenClipSummary[];
  timing: TrustOverviewBucketTiming;
}

export interface ReviewOverviewResponse {
  severity: ReviewOverviewSeverity;
  bar_size: ReviewOverviewBarSize;
  bar_seconds: number;
  num_bars: number;
  start_time: number;
  end_time: number;
  golden_total: number;
  unreviewed_fn_count: number;
  buckets: TrustOverviewBucket[];
  feedback_by_review_id: Record<string, ReviewFeedbackItem>;
}
