import { baseUrl } from "@/api/baseUrl";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { CorrectionParams } from "@/hooks/use-corrections";
import {
  KIND_LABELS,
  LongTermGroup,
  LongTermGroupSample,
  LongTermGroupsResponse,
  MaskSuggestion,
  OUTCOME_FILTER_LABELS,
  PRIMARY_COLOR,
  SecondlookVerdict,
} from "@/types/saveAndReview";
import axios from "axios";
import copy from "copy-to-clipboard";
import { useState } from "react";
import useSWR, { useSWRConfig } from "swr";
import useSWRInfinite from "swr/infinite";
import { toast } from "sonner";

const SIZE_LABELS = ["<1%", "1–2.5%", "2.5–5%", "5–10%", ">10%"];
const GRID = { columns: 8, rows: 6 };
const PAGE = 30;

function stillUrl(id: string): string {
  return `${baseUrl}api/secondlook/save_and_review/${id}/still.jpg`;
}

function formatTime(seconds: number): string {
  return new Date(seconds * 1000).toLocaleString();
}

/** "Suppressed", "Not rescued", ...; null for misclass (no outcome). */
export function outcomeText(
  group: Pick<LongTermGroup, "kind" | "outcome">,
): string | null {
  if (group.outcome === null || group.kind === "misclass") return null;
  return OUTCOME_FILTER_LABELS[group.kind][group.outcome];
}

export function groupTitle(group: LongTermGroup): string {
  const parts = [
    `${group.count} ${group.count === 1 ? "decision" : "decisions"}`,
    KIND_LABELS[group.kind],
  ];
  const outcome = outcomeText(group);
  if (outcome) parts.push(outcome);
  parts.push(
    group.kind === "misclass"
      ? `${group.label} → ${group.misclass_label ?? "other class"}`
      : group.label,
  );
  return parts.join(" · ");
}

export function groupDetail(group: LongTermGroup): string {
  const where = group.cells.length
    ? `box size ${SIZE_LABELS[group.size ?? 0]} of the frame`
    : "location unknown (no saved box or frame size)";
  return `${group.camera} · ${where} · ${formatTime(group.first_seen)} to ${formatTime(group.last_seen)}`;
}

/** 8 × 6 map of the frame; shaded cells hold the group's box centres. */
function GroupMap({ group }: { group: LongTermGroup }) {
  const max = Math.max(1, ...group.cell_counts.map(([, , n]) => n));
  return (
    <svg
      data-testid="group-map"
      role="img"
      aria-label={`Grid cells ${group.cells.map(([c, r]) => `column ${c + 1} row ${r + 1}`).join(", ") || "none"}`}
      viewBox={`0 0 ${GRID.columns} ${GRID.rows}`}
      className="h-12 w-16 shrink-0 rounded-sm border bg-muted"
    >
      {group.cell_counts.map(([c, r, n]) => (
        <rect
          key={`${c}-${r}`}
          x={c}
          y={r}
          width={1}
          height={1}
          fill="#F97316"
          fillOpacity={0.35 + 0.65 * (n / max)}
        />
      ))}
      {Array.from({ length: GRID.columns - 1 }, (_, i) => (
        <line
          key={`v${i}`}
          x1={i + 1}
          x2={i + 1}
          y1={0}
          y2={GRID.rows}
          stroke="currentColor"
          strokeOpacity={0.25}
          strokeWidth={0.04}
        />
      ))}
      {Array.from({ length: GRID.rows - 1 }, (_, i) => (
        <line
          key={`h${i}`}
          y1={i + 1}
          y2={i + 1}
          x1={0}
          x2={GRID.columns}
          stroke="currentColor"
          strokeOpacity={0.25}
          strokeWidth={0.04}
        />
      ))}
    </svg>
  );
}

function SampleThumb({
  sample,
  onClick,
}: {
  sample: LongTermGroupSample;
  onClick: () => void;
}) {
  const e = sample.evidence?.version === 1 ? sample.evidence : null;
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={`Sample ${sample.label} at ${formatTime(sample.time)}`}
      className="relative aspect-video w-full overflow-hidden rounded bg-black"
    >
      <img
        src={stillUrl(sample.id)}
        alt=""
        loading="lazy"
        className="absolute inset-0 size-full object-contain"
      />
      {e && sample.box?.length === 4 && (
        <svg
          className="pointer-events-none absolute inset-0 size-full"
          viewBox={`0 0 ${e.width} ${e.height}`}
          preserveAspectRatio="xMidYMid meet"
        >
          <rect
            x={sample.box[0]}
            y={sample.box[1]}
            width={sample.box[2] - sample.box[0]}
            height={sample.box[3] - sample.box[1]}
            fill="none"
            stroke={PRIMARY_COLOR}
            strokeWidth={Math.max(2, e.width / 200)}
          />
        </svg>
      )}
    </button>
  );
}

const BULK: { value: SecondlookVerdict; text: string }[] = [
  { value: "secondlook_right", text: "Secondlook right" },
  { value: "ignore", text: "Ignore" },
];

function GroupCard({
  group,
  params,
  onOpen,
  onChanged,
}: {
  group: LongTermGroup;
  params: CorrectionParams;
  onOpen: (g: LongTermGroup) => void;
  onChanged: () => void;
}) {
  const [bulk, setBulk] = useState<SecondlookVerdict | null>(null);
  const [mask, setMask] = useState(false);
  const canMask = group.kind === "false_positive" && group.cells.length > 0;
  const confirm = async () => {
    if (!bulk) return;
    try {
      const r = await axios.post(
        "secondlook/save_and_review/groups/mark",
        { mark: bulk },
        { params: { ...params, group: group.key } },
      );
      toast.success(
        `${r.data.updated} marked ${bulk === "ignore" ? "Ignore" : "Secondlook right"}`,
      );
      onChanged();
    } catch {
      toast.error("Failed to save verdicts");
    }
    setBulk(null);
  };
  return (
    <section
      data-testid="long-term-group"
      data-count={group.count}
      className="flex flex-col gap-2 rounded-lg border p-2 text-sm"
    >
      <div className="flex items-start gap-2">
        <GroupMap group={group} />
        <div className="flex min-w-0 flex-col gap-0.5">
          <strong className="break-words">{groupTitle(group)}</strong>
          <span className="break-words text-xs text-muted-foreground">
            {groupDetail(group)}
          </span>
          <span className="text-xs">
            {group.unreviewed} of {group.count} without a verdict
          </span>
        </div>
      </div>
      <div className="grid grid-cols-3 gap-1">
        {group.samples.map((sample) => (
          <SampleThumb
            key={sample.id}
            sample={sample}
            onClick={() => onOpen(group)}
          />
        ))}
      </div>
      <div className="flex flex-wrap gap-1">
        <Button size="sm" onClick={() => onOpen(group)}>
          Show all {group.count}
        </Button>
        {group.unreviewed > 0 &&
          BULK.map(({ value, text }) => (
            <Button
              key={value}
              size="sm"
              variant="outline"
              onClick={() => setBulk(value)}
            >
              {text} ({group.unreviewed})
            </Button>
          ))}
        {canMask && (
          <Button size="sm" variant="outline" onClick={() => setMask(true)}>
            Suggest mask
          </Button>
        )}
      </div>
      <AlertDialog
        open={bulk !== null}
        onOpenChange={(open) => {
          if (!open) setBulk(null);
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>
              Mark {group.unreviewed} decisions{" "}
              {bulk === "ignore" ? "Ignore" : "Secondlook right"}?
            </AlertDialogTitle>
            <AlertDialogDescription>
              Applies to the {group.unreviewed} decisions in this group that
              have no verdict yet; the other {group.count - group.unreviewed}{" "}
              keep theirs.
              {bulk === "ignore"
                ? " Ignored decisions are left out of training and are deleted after the retention period for ignored decisions."
                : " Secondlook's answer becomes the training label for each of them."}{" "}
              Each can still be changed one by one.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={confirm}>
              Mark {group.unreviewed}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
      {mask && (
        <MaskDialog
          group={group}
          params={params}
          onClose={() => setMask(false)}
        />
      )}
    </section>
  );
}

function MaskDialog({
  group,
  params,
  onClose,
}: {
  group: LongTermGroup;
  params: CorrectionParams;
  onClose: () => void;
}) {
  const [points, setPoints] = useState<"boxes" | "bottom">("boxes");
  const { data, error } = useSWR<MaskSuggestion>(
    [
      "secondlook/save_and_review/groups/mask",
      { ...params, group: group.key, points },
    ],
    ([path, query]: [string, Record<string, unknown>]) =>
      axios.get(path, { params: query }).then((r) => r.data),
    { revalidateOnFocus: false },
  );
  return (
    <Dialog
      open
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <DialogContent
        data-testid="mask-dialog"
        className="max-h-[95dvh] max-w-3xl overflow-y-auto"
      >
        <DialogHeader>
          <DialogTitle>Suggested object filter mask</DialogTitle>
          <DialogDescription>
            For {group.label} on {group.camera}: {group.count} false positives
            in this group. Nothing is changed in your config; paste the YAML
            yourself.
          </DialogDescription>
        </DialogHeader>
        <ToggleGroup
          type="single"
          size="sm"
          value={points}
          aria-label="Mask covers"
          onValueChange={(value) => {
            if (value === "boxes" || value === "bottom") setPoints(value);
          }}
          className="justify-start"
        >
          <ToggleGroupItem
            value="boxes"
            aria-label="Whole boxes"
            className="text-xs"
          >
            Whole boxes
          </ToggleGroupItem>
          <ToggleGroupItem
            value="bottom"
            aria-label="Bottom centres only"
            className="text-xs"
          >
            Bottom centres only
          </ToggleGroupItem>
        </ToggleGroup>
        {error && (
          <p role="alert" className="text-sm">
            No mask could be suggested for this group.
          </p>
        )}
        {!data && !error && <p className="text-sm">Working out the mask…</p>}
        {data && (
          <>
            <div className="relative w-full overflow-hidden rounded bg-black">
              <img
                data-testid="mask-preview-image"
                src={stillUrl(data.preview_id)}
                alt={`${group.camera} frame with the suggested mask`}
                className="block h-auto w-full"
              />
              <svg
                data-testid="mask-preview"
                role="img"
                aria-label={`Mask polygon with ${data.polygon.length} points over ${data.members} boxes`}
                className="pointer-events-none absolute inset-0 size-full"
                viewBox="0 0 1 1"
                preserveAspectRatio="none"
              >
                {data.boxes.map((b, i) => (
                  <rect
                    key={i}
                    x={b[0]}
                    y={b[1]}
                    width={b[2] - b[0]}
                    height={b[3] - b[1]}
                    fill="none"
                    stroke={PRIMARY_COLOR}
                    strokeOpacity={0.6}
                    strokeWidth={1}
                    vectorEffect="non-scaling-stroke"
                  />
                ))}
                <polygon
                  points={data.polygon.map(([x, y]) => `${x},${y}`).join(" ")}
                  fill="#DC2626"
                  fillOpacity={0.3}
                  stroke="#DC2626"
                  strokeWidth={2}
                  vectorEffect="non-scaling-stroke"
                />
              </svg>
            </div>
            <p className="text-xs text-muted-foreground">
              Red: the mask ({(data.area_fraction * 100).toFixed(2)}% of the
              frame, {Math.round(data.margin * 100)}% margin). Cyan: the{" "}
              {data.members} saved boxes it was made from. Frigate tests only
              the bottom centre of each box against an object filter mask: a{" "}
              {data.label} on {data.camera} whose bottom centre falls inside the
              red area is dropped by the filter and never becomes a detection,
              real ones included. {data.bottom_centres_inside} of {data.members}{" "}
              saved bottom centres fall inside.
              {data.area_fraction > 0.25
                ? " This mask covers a large part of the frame; check the group before using it."
                : ""}
            </p>
            {data.existing_masks.length > 0 && (
              <p className="text-xs">
                This camera already has {data.existing_masks.length}{" "}
                {data.label} filter{" "}
                {data.existing_masks.length === 1 ? "mask" : "masks"} (
                {data.existing_masks.join(", ")}); this one is added beside them
                under its own name.
              </p>
            )}
            <pre
              data-testid="mask-yaml"
              className="overflow-x-auto rounded bg-muted p-2 text-xs"
            >
              {data.yaml}
            </pre>
            <div className="flex items-center gap-2">
              <Button
                size="sm"
                onClick={() => {
                  copy(data.yaml);
                  toast.success("Mask YAML copied");
                }}
              >
                Copy YAML
              </Button>
              <span className="text-xs text-muted-foreground">
                Merge it into cameras → {data.camera} → objects → filters →{" "}
                {data.label} → mask, then restart Frigate.
              </span>
            </div>
          </>
        )}
      </DialogContent>
    </Dialog>
  );
}

export default function CorrectionGroupsView({
  params,
  onOpen,
}: {
  params: CorrectionParams;
  onOpen: (group: LongTermGroup) => void;
}) {
  const { mutate: invalidate } = useSWRConfig();
  const { data, error, size, setSize, isValidating, mutate } =
    useSWRInfinite<LongTermGroupsResponse>(
      (index, previous) =>
        previous && !previous.has_more
          ? null
          : [
              "secondlook/save_and_review/groups",
              { ...params, limit: PAGE, offset: index * PAGE },
            ],
      ([path, query]) => axios.get(path, { params: query }).then((r) => r.data),
      { revalidateOnFocus: false, refreshInterval: 30000 },
    );
  const groups = data?.flatMap((page) => page.groups) ?? [];
  const first = data?.[0];
  const onChanged = () => {
    mutate();
    invalidate(
      (key) => Array.isArray(key) && key[0] === "secondlook/save_and_review",
    );
  };
  return (
    <div
      data-testid="long-term-groups"
      className="min-h-0 flex-1 overflow-y-auto pb-4"
    >
      {error && (
        <p role="alert">
          Groups unavailable. Saved examples are not removed by this error.
        </p>
      )}
      {!data && !error && <p>Grouping saved decisions…</p>}
      {first && (
        <p className="pb-2 text-xs">
          {first.total_groups} groups · {first.total_rows} decisions, most first
        </p>
      )}
      {first && groups.length === 0 && (
        <p>There are no corrections matching these filters</p>
      )}
      <div className="grid auto-rows-max grid-cols-1 content-start items-start gap-3 sm:grid-cols-2 xl:grid-cols-3">
        {groups.map((group) => (
          <GroupCard
            key={group.key}
            group={group}
            params={params}
            onOpen={onOpen}
            onChanged={onChanged}
          />
        ))}
      </div>
      {data?.[data.length - 1]?.has_more && (
        <Button
          className="my-3"
          disabled={isValidating}
          onClick={() => setSize(size + 1)}
        >
          Load more groups
        </Button>
      )}
    </div>
  );
}
