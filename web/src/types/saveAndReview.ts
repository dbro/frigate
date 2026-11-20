export type AnalysedEvidence = {
  version: 1;
  width: number;
  height: number;
  frame_time: number;
  primary_box: number[];
  sl_box: number[] | null;
  crop_region: number[];
  secondary_label: string;
  secondary_score: number | null;
};
export type SecondlookExampleKind =
  | "false_positive"
  | "false_negative"
  | "misclass";

/** Which model's answer is the training label, or excluded from training. */
export type SecondlookVerdict = "primary_right" | "secondlook_right" | "ignore";
export type SecondlookExampleMark = SecondlookVerdict | null;

export type SecondlookExample = {
  evidence?: AnalysedEvidence | null;
  group_key?: string | null;
  id: string;
  track_id: string;
  camera: string;
  kind: SecondlookExampleKind;
  label: string;
  misclass_label: string | null;
  primary_score: number | null;
  sl_score: number | null;
  factor: number | null;
  threshold: number | null;
  raw_median: number | null;
  box: number[] | null;
  sl_box: number[] | null;
  frame_time: number | null;
  model_id: string | null;
  primary_model_id: string | null;
  still_path: string;
  event_id: string | null;
  has_clip: boolean;
  zones: string[];
  mark: SecondlookExampleMark;
  created_at: number;
  marked_at: number | null;
  /** Pre-verdict accept/reject value kept by the lossless migration. */
  legacy_mark?: "accept" | "reject" | null;
  /** Did the live pipeline act on it? From Frigate's Event table at read time. */
  live_outcome?: LiveOutcome;
};

export type LiveOutcome =
  | "rescued"
  | "not_rescued"
  | "suppressed"
  | "not_suppressed"
  | "none";

/** Badge text, alarm flag and (for the alarm ones) why it may have happened. */
export const LIVE_OUTCOME_BADGES: Record<
  Exclude<LiveOutcome, "none">,
  { text: string; alarm: boolean; reason?: string }
> = {
  rescued: { text: "Rescued: alert raised", alarm: false },
  suppressed: { text: "Suppressed: alert prevented", alarm: false },
  not_rescued: {
    text: "Not rescued",
    alarm: true,
    reason:
      "With Secondlook's score it would have reached the threshold, but it was not released: the answer came too late (after the hold, or after the object left).",
  },
  not_suppressed: {
    text: "Not suppressed",
    alarm: true,
    reason:
      "With Secondlook's score it would have stayed below the threshold, but it was released anyway: the answer came too late (after the hold, or after release).",
  },
};

export type SaveAndReviewResponse = {
  examples: SecondlookExample[];
  unreviewed: number;
  total: number;
  has_more: boolean;
  hidden_edge_small?: number;
};

export const KIND_LABELS: Record<SecondlookExampleKind | "all", string> = {
  all: "All",
  false_positive: "False positive",
  false_negative: "False negative",
  misclass: "Misclass",
};

/** Box colours; the matching verdict buttons reuse them. */
export const PRIMARY_COLOR = "#00ffff";
export const SECONDLOOK_COLOR = "#ff9900";

/** Stored outcome filter: acted = FN rescued / FP suppressed (from event_id). */
export type CorrectionOutcome = "all" | "acted" | "not_acted";

/** Plain labels for the outcome filter, per kind. */
export const OUTCOME_FILTER_LABELS: Record<
  "false_positive" | "false_negative",
  Record<CorrectionOutcome, string>
> = {
  false_positive: { all: "All", acted: "Suppressed", not_acted: "Not suppressed" },
  false_negative: { all: "All", acted: "Rescued", not_acted: "Not rescued" },
};

export type CorrectionGrouping = "none" | "short" | "long";

export type LongTermGroupSample = {
  id: string;
  time: number;
  kind: SecondlookExampleKind;
  label: string;
  misclass_label: string | null;
  camera: string;
  primary_score: number | null;
  sl_score: number | null;
  factor: number | null;
  mark: SecondlookExampleMark;
  box: number[] | null;
  evidence: AnalysedEvidence | null;
};

/** One long-term group: fixed-bin key plus the merged grid cells. */
export type LongTermGroup = {
  /** Opaque key; sent back as `group` for drill-in, bulk verdicts and masks. */
  key: string;
  camera: string;
  kind: SecondlookExampleKind;
  outcome: "acted" | "not_acted" | null;
  label: string;
  misclass_label: string | null;
  /** Size bucket index into size_labels; null when the location is unknown. */
  size: number | null;
  cells: [number, number][];
  cell_counts: [number, number, number][];
  count: number;
  unreviewed: number;
  first_seen: number;
  last_seen: number;
  extent: [number, number, number, number] | null;
  samples: LongTermGroupSample[];
};

export type LongTermGroupsResponse = {
  groups: LongTermGroup[];
  total_groups: number;
  total_rows: number;
  has_more: boolean;
  grid: { columns: number; rows: number };
  size_labels: string[];
};

export type MaskSuggestion = {
  camera: string;
  label: string;
  name: string;
  friendly_name: string;
  members: number;
  points: "boxes" | "bottom";
  margin: number;
  polygon: [number, number][];
  coordinates: string;
  area_fraction: number;
  bottom_centres_inside: number;
  existing_masks: string[];
  yaml: string;
  boxes: [number, number, number, number][];
  preview_id: string;
};
