import { useCallback, useState } from "react";
import axios from "axios";
import { useSWRConfig } from "swr";
import { toast } from "sonner";
import { cn } from "@/lib/utils";
import { ReviewSegment } from "@/types/review";
import {
  CreateGoldenClipBody,
  ReviewFeedbackBody,
  ReviewFeedbackItem,
  ReviewFeedbackVerdict,
} from "@/types/reviewTrust";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

const GOLDEN_CONDITION_TAGS = [
  { key: "day", label: "☀️ Day" },
  { key: "night_ir", label: "🌙 Night IR" },
  { key: "rain", label: "🌧️ Rain" },
  { key: "wind", label: "💨 Wind" },
  { key: "snow", label: "❄️ Snow" },
] as const;

type ReviewFeedbackBarProps = {
  review: ReviewSegment;
  feedback?: ReviewFeedbackItem;
  className?: string;
};

export default function ReviewFeedbackBar({
  review,
  feedback,
  className,
}: ReviewFeedbackBarProps) {
  const { mutate } = useSWRConfig();
  const [submittingVerdict, setSubmittingVerdict] = useState<string | null>(
    null,
  );
  const [classPopoverOpen, setClassPopoverOpen] = useState(false);
  const [correctedLabel, setCorrectedLabel] = useState(
    feedback?.corrected_label ?? "",
  );
  const [goldenPopoverOpen, setGoldenPopoverOpen] = useState(false);
  const [selectedTags, setSelectedTags] = useState<string[]>(["day"]);
  const [goldenNotes, setGoldenNotes] = useState("");
  const [savingGolden, setSavingGolden] = useState(false);

  const currentVerdict = feedback?.verdict;
  const primaryObject =
    review.data.objects[0]?.replace("-verified", "") ?? null;

  const revalidateOverview = useCallback(async () => {
    await mutate(
      (key) =>
        (Array.isArray(key) && key[0] === "review/overview") ||
        key === "review/golden_clips",
    );
  }, [mutate]);

  const submitFeedback = useCallback(
    async (verdict: ReviewFeedbackVerdict, labelOverride?: string | null) => {
      const nextVerdict: ReviewFeedbackVerdict =
        verdict !== "misclassified" && currentVerdict === verdict
          ? "clear"
          : verdict;

      const body: ReviewFeedbackBody = {
        verdict: nextVerdict,
        corrected_label:
          nextVerdict === "misclassified" ? (labelOverride ?? null) : null,
        reason: null,
      };

      setSubmittingVerdict(verdict);
      try {
        await axios.post(
          `review/${encodeURIComponent(review.id)}/feedback`,
          body,
        );
        await revalidateOverview();
        if (nextVerdict === "clear") {
          toast.success("Cleared review feedback", { position: "top-center" });
        } else {
          toast.success(`Saved feedback: ${nextVerdict.toUpperCase()}`, {
            position: "top-center",
          });
        }
      } catch {
        toast.error("Failed to save review feedback", {
          position: "top-center",
        });
      } finally {
        setSubmittingVerdict(null);
      }
    },
    [currentVerdict, review.id, revalidateOverview],
  );

  const handleSaveGolden = useCallback(async () => {
    setSavingGolden(true);
    const inferredKind =
      currentVerdict === "fn"
        ? "Human FN"
        : currentVerdict === "fp"
          ? "Human FP"
          : currentVerdict === "misclassified"
            ? "Misclassified"
            : review.severity === "alert"
              ? "Alert Specimen"
              : "Detection Specimen";

    const body: CreateGoldenClipBody = {
      kind: inferredKind,
      condition_tags: selectedTags.length > 0 ? selectedTags : ["day"],
      object_label: correctedLabel.trim() || primaryObject,
      notes: goldenNotes.trim() || null,
    };

    try {
      await axios.post(`review/${encodeURIComponent(review.id)}/golden`, body);
      await revalidateOverview();
      setGoldenPopoverOpen(false);
      setGoldenNotes("");
      toast.success(`Saved golden clip for ${review.camera}`, {
        position: "top-center",
      });
    } catch {
      toast.error("Failed to save golden clip", { position: "top-center" });
    } finally {
      setSavingGolden(false);
    }
  }, [
    correctedLabel,
    currentVerdict,
    goldenNotes,
    primaryObject,
    revalidateOverview,
    review.camera,
    review.id,
    review.severity,
    selectedTags,
  ]);

  const toggleTag = useCallback((tag: string) => {
    setSelectedTags((prev) =>
      prev.includes(tag) ? prev.filter((t) => t !== tag) : [...prev, tag],
    );
  }, []);

  return (
    <div
      className={cn(
        "flex items-center justify-between gap-1 border-t border-border bg-card px-2 py-1.5",
        className,
      )}
      onClick={(e) => e.stopPropagation()}
    >
      {review.severity === "detection" ? (
        <button
          type="button"
          disabled={submittingVerdict !== null}
          onClick={() => submitFeedback("fn")}
          className={cn(
            "flex-1 rounded-md border px-1.5 py-1 text-xs font-medium transition",
            currentVerdict === "fn"
              ? "border-red-500 bg-red-500/25 text-red-300"
              : "border-red-500/40 text-red-400 hover:bg-red-500/20",
          )}
          title="Mark false negative (missed alert)"
        >
          FN
        </button>
      ) : (
        <button
          type="button"
          disabled={submittingVerdict !== null}
          onClick={() => submitFeedback("good")}
          className={cn(
            "flex-1 rounded-md border px-1.5 py-1 text-xs font-medium transition",
            currentVerdict === "good"
              ? "border-emerald-500 bg-emerald-500/25 text-emerald-300"
              : "border-emerald-500/40 text-emerald-400 hover:bg-emerald-500/20",
          )}
          title="Confirm good alert"
        >
          Good
        </button>
      )}

      <button
        type="button"
        disabled={submittingVerdict !== null}
        onClick={() => submitFeedback("fp")}
        className={cn(
          "flex-1 rounded-md border px-1.5 py-1 text-xs font-medium transition",
          currentVerdict === "fp"
            ? "border-rose-500 bg-rose-500/25 text-rose-300"
            : "border-rose-500/40 text-rose-400 hover:bg-rose-500/20",
        )}
        title="Mark false positive"
      >
        FP
      </button>

      <Popover open={classPopoverOpen} onOpenChange={setClassPopoverOpen}>
        <PopoverTrigger asChild>
          <button
            type="button"
            disabled={submittingVerdict !== null}
            className={cn(
              "flex-1 rounded-md border px-1.5 py-1 text-xs font-medium transition",
              currentVerdict === "misclassified"
                ? "border-yellow-500 bg-yellow-500/25 text-yellow-300"
                : "border-yellow-500/40 text-yellow-400 hover:bg-yellow-500/20",
            )}
            title="Mark misclassified label"
          >
            Class
          </button>
        </PopoverTrigger>
        <PopoverContent className="w-60 space-y-2 p-3" align="center">
          <div className="text-xs font-medium">
            Correct class{primaryObject ? ` (was ${primaryObject})` : ""}
          </div>
          <Input
            value={correctedLabel}
            onChange={(e) => setCorrectedLabel(e.target.value)}
            placeholder="e.g. fox, person, car"
            className="h-8 text-xs"
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                submitFeedback("misclassified", correctedLabel.trim() || null);
                setClassPopoverOpen(false);
              }
            }}
          />
          <div className="flex items-center justify-end gap-1.5">
            {currentVerdict === "misclassified" && (
              <Button
                type="button"
                size="sm"
                variant="ghost"
                className="h-7 px-2 text-xs"
                onClick={() => {
                  submitFeedback("clear");
                  setClassPopoverOpen(false);
                }}
              >
                Clear
              </Button>
            )}
            <Button
              type="button"
              size="sm"
              className="h-7 px-2.5 text-xs"
              onClick={() => {
                submitFeedback("misclassified", correctedLabel.trim() || null);
                setClassPopoverOpen(false);
              }}
            >
              Save
            </Button>
          </div>
        </PopoverContent>
      </Popover>

      <Popover open={goldenPopoverOpen} onOpenChange={setGoldenPopoverOpen}>
        <PopoverTrigger asChild>
          <button
            type="button"
            className="flex-1 whitespace-nowrap rounded-md border border-amber-500/40 px-1.5 py-1 text-xs font-medium text-amber-400 transition hover:bg-amber-500/20"
            title="Save as Golden Clip"
          >
            🏆 Save
          </button>
        </PopoverTrigger>
        <PopoverContent className="w-64 space-y-2.5 p-3" align="end">
          <div className="text-xs font-semibold text-amber-400">
            🏆 Save Golden Clip
          </div>
          <div className="flex flex-wrap gap-1">
            {GOLDEN_CONDITION_TAGS.map((tag) => {
              const active = selectedTags.includes(tag.key);
              return (
                <button
                  key={tag.key}
                  type="button"
                  onClick={() => toggleTag(tag.key)}
                  className={cn(
                    "rounded-md border px-2 py-0.5 text-[11px] transition",
                    active
                      ? "border-amber-500/60 bg-amber-500/20 text-amber-300"
                      : "border-border bg-background text-muted-foreground hover:text-foreground",
                  )}
                >
                  {tag.label}
                </button>
              );
            })}
          </div>
          <Input
            value={goldenNotes}
            onChange={(e) => setGoldenNotes(e.target.value)}
            placeholder="Optional notes (e.g. Headlight glare)"
            className="h-8 text-xs"
          />
          <div className="flex justify-end gap-1.5">
            <Button
              type="button"
              size="sm"
              variant="ghost"
              className="h-7 px-2 text-xs"
              onClick={() => setGoldenPopoverOpen(false)}
            >
              Cancel
            </Button>
            <Button
              type="button"
              size="sm"
              disabled={savingGolden}
              className="h-7 bg-amber-600 px-2.5 text-xs text-white hover:bg-amber-500"
              onClick={handleSaveGolden}
            >
              {savingGolden ? "Saving..." : "Save Clip"}
            </Button>
          </div>
        </PopoverContent>
      </Popover>
    </div>
  );
}
