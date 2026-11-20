import { ReviewFilter } from "@/types/review";
import {
  CorrectionOutcome,
  SecondlookExampleKind,
} from "@/types/saveAndReview";

export type CorrectionKind = SecondlookExampleKind | "all";

/** The acted-on filter applies only to FP and FN (misclass has no outcome). */
export function outcomeApplies(kind: CorrectionKind): boolean {
  return kind === "false_positive" || kind === "false_negative";
}

export function correctionParams(
  filter: ReviewFilter | undefined,
  kind: CorrectionKind,
  showReviewed: boolean,
  hideEdgeSmall: boolean,
  outcome: CorrectionOutcome = "all",
) {
  return {
    after: filter?.after,
    before: filter?.before,
    cameras: filter?.cameras?.join(","),
    labels: filter?.labels?.join(","),
    zones: filter?.zones?.join(","),
    kind: kind === "all" ? undefined : kind,
    outcome: outcomeApplies(kind) && outcome !== "all" ? outcome : undefined,
    reviewed: showReviewed ? undefined : false,
    hide_edge_small: hideEdgeSmall ? true : undefined,
  };
}
export type CorrectionParams = ReturnType<typeof correctionParams>;
