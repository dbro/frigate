import { useCallback, useMemo, useState } from "react";
import axios from "axios";
import useSWR, { useSWRConfig } from "swr";
import { toast } from "sonner";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { GoldenClipSummary, GoldenClipsResponse } from "@/types/reviewTrust";
import { LuTrash2, LuExternalLink } from "react-icons/lu";

const SCENARIO_COVERAGE_ITEMS: {
  key: string;
  label: string;
  emoji: string;
}[] = [
  { key: "day", label: "Day", emoji: "☀️" },
  { key: "night_ir", label: "Night IR", emoji: "🌙" },
  { key: "rain", label: "Rain", emoji: "🌧️" },
  { key: "wind", label: "Wind", emoji: "💨" },
  { key: "snow", label: "Snow", emoji: "❄️" },
];

function getScenarioCount(
  coverage: Record<string, number> | undefined,
  clips: GoldenClipSummary[],
  key: string,
): number {
  if (coverage) {
    const exact = coverage[key];
    if (typeof exact === "number") {
      return exact;
    }
    const lowerKey = key.toLowerCase();
    for (const [k, v] of Object.entries(coverage)) {
      if (k.toLowerCase().replace(/[\s-]+/g, "_") === lowerKey) {
        return v;
      }
    }
  }

  return clips.filter((clip) =>
    clip.condition_tags.some(
      (tag) => tag.toLowerCase().replace(/[\s-]+/g, "_") === key,
    ),
  ).length;
}

function getKindColorClass(kind: string): string {
  const normalized = kind.toLowerCase();
  if (normalized.includes("fn") || normalized.includes("missed")) {
    return "text-red-400";
  }
  if (normalized.includes("fp") || normalized.includes("false_positive")) {
    return "text-orange-400";
  }
  if (normalized.includes("misclass")) {
    return "text-yellow-400";
  }
  if (normalized.includes("background")) {
    return "text-slate-400";
  }
  return "text-emerald-400";
}

type GoldenClipsDialogProps = {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSelectClip?: (clip: GoldenClipSummary) => void;
};

export default function GoldenClipsDialog({
  open,
  onOpenChange,
  onSelectClip,
}: GoldenClipsDialogProps) {
  const { mutate: globalMutate } = useSWRConfig();
  const [deletingId, setDeletingId] = useState<string | null>(null);

  const { data, isLoading, mutate } = useSWR<GoldenClipsResponse>(
    open ? "review/golden_clips" : null,
    (path: string) => axios.get(path).then((res) => res.data),
    { revalidateOnFocus: false },
  );

  const clips = useMemo(() => data?.clips ?? [], [data]);
  const total = data?.total ?? clips.length;

  const handleDelete = useCallback(
    async (clipId: string) => {
      setDeletingId(clipId);
      try {
        await axios.delete(`review/golden_clips/${encodeURIComponent(clipId)}`);
        await mutate();
        await globalMutate(
          (key) =>
            (Array.isArray(key) && key[0] === "review/overview") ||
            key === "review/golden_clips",
        );
        toast.success(`Deleted golden clip ${clipId}`, {
          position: "top-center",
        });
      } catch {
        toast.error("Failed to delete golden clip", {
          position: "top-center",
        });
      } finally {
        setDeletingId(null);
      }
    },
    [mutate, globalMutate],
  );

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[85dvh] max-w-xl overflow-y-auto">
        <DialogHeader className="border-b border-border pb-3">
          <DialogTitle className="flex items-center gap-2 text-base">
            <span>🏆</span>
            <span>Golden Clips Evaluation Set</span>
            <Badge variant="secondary" className="ml-1 text-xs">
              {total}
            </Badge>
          </DialogTitle>
          <DialogDescription className="text-xs">
            Saved locally in{" "}
            <code className="rounded bg-muted px-1 py-0.5 font-mono text-[11px]">
              exports/golden_clips/
            </code>
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 pt-1">
          <div>
            <div className="mb-2 text-xs font-medium text-muted-foreground">
              Scenario Coverage
            </div>
            <div className="grid grid-cols-5 gap-2 text-center text-xs">
              {SCENARIO_COVERAGE_ITEMS.map((item) => {
                const count = getScenarioCount(data?.coverage, clips, item.key);
                const covered = count > 0;
                return (
                  <div
                    key={item.key}
                    className={cn(
                      "rounded-lg border px-1 py-2",
                      covered
                        ? "border-emerald-500/40 bg-emerald-500/10"
                        : "border-border bg-background opacity-75",
                    )}
                  >
                    <div>
                      {item.emoji} {item.label}
                    </div>
                    <div
                      className={cn(
                        "mt-0.5 font-semibold",
                        covered ? "text-emerald-400" : "text-muted-foreground",
                      )}
                    >
                      {count} {count === 1 ? "clip" : "clips"}{" "}
                      {covered ? "✓" : "○"}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          <div className="space-y-1.5 text-xs">
            <div className="font-medium text-muted-foreground">
              Saved Golden Clips
            </div>

            {isLoading && (
              <div className="rounded-lg border border-border bg-background p-3 text-center text-muted-foreground">
                Loading golden clips...
              </div>
            )}

            {!isLoading && clips.length === 0 && (
              <div className="rounded-lg border border-border bg-background p-4 text-center text-muted-foreground">
                No golden clips saved yet. Use 🏆 Save on any review card to add
                one.
              </div>
            )}

            <div className="max-h-64 space-y-1.5 overflow-y-auto pr-0.5">
              {clips.map((clip) => {
                const tagsLabel = clip.condition_tags.join(", ") || "Untagged";
                const fileLabel =
                  clip.file_path ??
                  clip.export_path ??
                  `exports/golden_clips/${clip.id}.mp4`;

                return (
                  <div
                    key={clip.id}
                    className="flex items-center justify-between gap-2 rounded-lg border border-border bg-background p-2"
                  >
                    <div className="min-w-0 flex-1 space-y-0.5">
                      <div className="flex flex-wrap items-center gap-1.5">
                        <span className="font-semibold text-foreground">
                          {clip.camera}
                        </span>
                        <span className="text-muted-foreground">·</span>
                        <span className="text-muted-foreground">
                          {tagsLabel}
                        </span>
                        <span className="text-muted-foreground">·</span>
                        <span
                          className={cn(
                            "font-medium",
                            getKindColorClass(clip.kind),
                          )}
                        >
                          {clip.kind}
                          {clip.object_label ? ` (${clip.object_label})` : ""}
                        </span>
                      </div>
                      {clip.notes && (
                        <div className="truncate text-[11px] text-muted-foreground">
                          {clip.notes}
                        </div>
                      )}
                      <div className="truncate font-mono text-[11px] text-muted-foreground">
                        {fileLabel}
                      </div>
                    </div>

                    <div className="flex shrink-0 items-center gap-1">
                      {onSelectClip && clip.review_id && (
                        <Button
                          type="button"
                          variant="ghost"
                          size="icon"
                          className="size-7"
                          aria-label={`Open clip ${clip.id}`}
                          onClick={() => {
                            onSelectClip(clip);
                            onOpenChange(false);
                          }}
                        >
                          <LuExternalLink className="size-3.5" />
                        </Button>
                      )}
                      <Button
                        type="button"
                        variant="ghost"
                        size="icon"
                        className="size-7 text-muted-foreground hover:text-destructive"
                        aria-label={`Delete golden clip ${clip.id}`}
                        disabled={deletingId === clip.id}
                        onClick={() => handleDelete(clip.id)}
                      >
                        <LuTrash2 className="size-3.5" />
                      </Button>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
