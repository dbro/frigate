import { baseUrl } from "@/api/baseUrl";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { ReviewFilter } from "@/types/review";
import { FrigateConfig } from "@/types/frigateConfig";
import {
  CorrectionGrouping,
  CorrectionOutcome,
  KIND_LABELS,
  LIVE_OUTCOME_BADGES,
  LongTermGroup,
  OUTCOME_FILTER_LABELS,
  LiveOutcome,
  PRIMARY_COLOR,
  SECONDLOOK_COLOR,
  SaveAndReviewResponse,
  SecondlookExample,
  SecondlookExampleMark,
  SecondlookVerdict,
} from "@/types/saveAndReview";
import { correctionParams, CorrectionKind } from "@/hooks/use-corrections";
import CorrectionGroupsView, { groupDetail, groupTitle } from "./CorrectionGroupsView";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { DialogOverlay, DialogPortal } from "@/components/ui/dialog";
import * as DialogPrimitive from "@radix-ui/react-dialog";
import axios from "axios";
import { ChevronLeft, ChevronRight, X } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useSwipeable } from "react-swipeable";
import { TransformComponent, TransformWrapper } from "react-zoom-pan-pinch";
import useSWR, { useSWRConfig } from "swr";
import useSWRInfinite from "swr/infinite";
import { toast } from "sonner";

type OnMark = (id: string, mark: SecondlookExampleMark) => void;
type OnOpen = (id: string) => void;

function formatScore(v: number | null | undefined) {
  return v == null || Number.isNaN(v) ? "—" : v.toFixed(2);
}
function isKind(value: string): value is CorrectionKind {
  return value === "all" || value === "false_positive" || value === "false_negative" || value === "misclass";
}
function exampleTime(ex: SecondlookExample): number {
  return ex.frame_time ?? ex.created_at;
}
function secondlookAnswer(ex: SecondlookExample): string {
  if (ex.kind === "misclass") return ex.misclass_label ?? ex.evidence?.secondary_label ?? "other class";
  if (ex.kind === "false_positive") return `not ${ex.label}`;
  return ex.label;
}
function primaryAnswer(ex: SecondlookExample): string {
  return ex.kind === "false_negative" ? `not ${ex.label} (below threshold)` : ex.label;
}

/**
 * Display-only grouping: consecutive decisions (in the loaded, time-ordered list)
 * with the same camera, kind, primary class and Secondlook class whose frame
 * times are at most gapSeconds apart. Every example keeps its own card and mark.
 */
export function groupByProximity(examples: SecondlookExample[], gapSeconds: number): SecondlookExample[][] {
  const groups: SecondlookExample[][] = [];
  const open = new Map<string, {items: SecondlookExample[]; lastTime: number}>();
  for (const ex of examples) {
    const key = [ex.camera, ex.kind, ex.label, ex.kind === "misclass" ? secondlookAnswer(ex) : ""].join("\u0000");
    const time = exampleTime(ex);
    const current = open.get(key);
    if (current && Math.abs(current.lastTime - time) <= gapSeconds) {
      current.items.push(ex);
      current.lastTime = time;
    } else {
      const items = [ex];
      groups.push(items);
      open.set(key, {items, lastTime: time});
    }
  }
  return groups;
}

const WINDOWS: {value: string; text: string; seconds: number | null}[] = [
  {value: "all", text: "All saved", seconds: null},
  {value: "30d", text: "Last 30 days", seconds: 30 * 86400},
  {value: "7d", text: "Last 7 days", seconds: 7 * 86400},
  {value: "24h", text: "Last 24 hours", seconds: 86400},
];

const GROUPINGS: {value: CorrectionGrouping; text: string}[] = [
  {value: "none", text: "No grouping"},
  {value: "short", text: "Short-term"},
  {value: "long", text: "Long-term"},
];

export default function CorrectionsView({filter, showReviewed, kind, onKind, outcome = "all", onOutcome, onAllHistory, hideEdgeSmall, onHideEdgeSmall}: {
  filter?: ReviewFilter; showReviewed: boolean; kind: CorrectionKind;
  onKind: (kind: CorrectionKind) => void; outcome?: CorrectionOutcome;
  onOutcome?: (outcome: CorrectionOutcome) => void; onAllHistory: () => void;
  hideEdgeSmall: boolean; onHideEdgeSmall: (hide: boolean) => void;
}) {
  const { data: config } = useSWR<FrigateConfig>("config");
  const review = config?.secondlook?.save_and_review.review;
  const gapSeconds = review?.group_gap_seconds ?? 60;
  const [grouping, setGrouping] = useState<CorrectionGrouping>("short");
  // long-term window: the start is fixed when chosen, so the request key is stable
  const [timeWindow, setTimeWindow] = useState<{value: string; since: number | undefined}>({value: "all", since: undefined});
  const [drill, setDrill] = useState<LongTermGroup | null>(null);
  const baseParams = correctionParams(filter, kind, showReviewed, hideEdgeSmall, outcome);
  const longParams = useMemo(() => {
    const after = [baseParams.after, timeWindow.since].filter((v): v is number => v != null);
    return {...baseParams, after: after.length ? Math.max(...after) : undefined};
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [JSON.stringify(baseParams), timeWindow.since]);
  const longKey = JSON.stringify(longParams);
  // a different filter or mode leaves the drilled-in group
  useEffect(() => setDrill(null), [longKey, grouping]);
  const params = grouping === "long" ? (drill ? {...longParams, group: drill.key} : null) : baseParams;
  const { mutate: invalidate } = useSWRConfig();
  const { data, error, size, setSize, isValidating, mutate } = useSWRInfinite<SaveAndReviewResponse>(
    (index, previous) => !params || (previous && !previous.has_more) ? null : ["secondlook/save_and_review", {...params, limit: 50, offset: index * 50}],
    ([path, query]) => axios.get(path, {params: query}).then(r => r.data),
    {revalidateOnFocus: false, refreshInterval: 10000},
  );
  const onMark = useCallback<OnMark>(async (id, mark) => {
    try {
      await axios.post(`secondlook/save_and_review/${id}/mark`, {mark});
      await mutate();
      await invalidate(key => Array.isArray(key) && key[0] === "secondlook/save_and_review");
    } catch { toast.error("Failed to save verdict"); }
  }, [mutate, invalidate]);
  const examples = useMemo(
    () => Array.from(new Map((data?.flatMap(page => page.examples) ?? []).map(ex => [ex.id, ex])).values()),
    [data],
  );
  const groups = useMemo(
    () => grouping === "short" ? groupByProximity(examples, gapSeconds) : examples.map(ex => [ex]),
    [examples, grouping, gapSeconds],
  );
  // Lightbox walks the cards in display order, group members included.
  const order = useMemo(() => groups.flat(), [groups]);
  const [lightbox, setLightbox] = useState<{id: string; index: number} | null>(null);
  const shownIndex = lightbox ? resolveLightboxIndex(order, lightbox) : -1;
  const shown = shownIndex >= 0 ? order[shownIndex] : undefined;
  useEffect(() => {
    if (!lightbox) return;
    // A mark can drop the item from the filtered list: show the one that took its place.
    if (!shown) setLightbox(null);
    else if (shown.id !== lightbox.id || shownIndex !== lightbox.index) setLightbox({id: shown.id, index: shownIndex});
  }, [lightbox, shown, shownIndex]);
  const openLightbox = useCallback<OnOpen>(id => setLightbox({id, index: order.findIndex(ex => ex.id === id)}), [order]);
  const step = useCallback((delta: number) => {
    if (shownIndex < 0) return;
    const index = shownIndex + delta;
    if (index >= 0 && index < order.length) setLightbox({id: order[index].id, index});
  }, [order, shownIndex]);
  const first = data?.[0];
  const last = data?.[data.length - 1];
  const hidden = first?.hidden_edge_small ?? 0;
  const outcomeLabels = kind === "false_positive" || kind === "false_negative" ? OUTCOME_FILTER_LABELS[kind] : null;
  const showList = grouping !== "long" || drill !== null;
  return <div className="flex h-full min-h-0 flex-col overflow-hidden px-2 md:px-3">
    <div className="mb-2 flex shrink-0 flex-wrap items-center gap-2 py-1">
      <ToggleGroup type="single" value={kind} onValueChange={value => {if (isKind(value)) onKind(value);}} className="flex flex-wrap justify-start">
        {Object.entries(KIND_LABELS).map(([value, text]) => <ToggleGroupItem key={value} value={value} aria-label={text}>{text}</ToggleGroupItem>)}
      </ToggleGroup>
      {outcomeLabels && onOutcome && <ToggleGroup data-testid="outcome-filter" type="single" size="sm" value={outcome} aria-label={kind === "false_positive" ? "Suppressed or not" : "Rescued or not"} onValueChange={value => {if (value === "all" || value === "acted" || value === "not_acted") onOutcome(value);}} className="flex flex-wrap justify-start rounded-md border px-1">
        {(["all", "acted", "not_acted"] as const).map(value => <ToggleGroupItem key={value} value={value} aria-label={outcomeLabels[value]} className="text-xs">{outcomeLabels[value]}</ToggleGroupItem>)}
      </ToggleGroup>}
      {showList && <span className="text-xs">{first?.unreviewed ?? "—"} unreviewed · {first?.total ?? "—"} matching</span>}
      {filter?.after || filter?.before ? <Button variant="outline" size="sm" onClick={onAllHistory}>All saved history</Button> : <span className="text-xs">All saved history</span>}
    </div>
    {error && <p role="alert">Corrections unavailable. Saved examples are not removed by this error.</p>}
    <div className="flex shrink-0 flex-wrap items-center gap-x-4 gap-y-2 pb-2 text-xs">
      <div className="flex items-center gap-2">
        <Switch id="sl-hide-edge-small" checked={hideEdgeSmall} onCheckedChange={onHideEdgeSmall} />
        <Label htmlFor="sl-hide-edge-small" className="text-xs">Hide small boxes touching the frame edge</Label>
        <span data-testid="edge-hidden-count">{!showList ? "" : hideEdgeSmall ? `${hidden} hidden (still saved)` : "showing all"}</span>
      </div>
      <div className="flex items-center gap-2">
        <span>Grouping</span>
        <ToggleGroup data-testid="grouping-mode" type="single" size="sm" value={grouping} aria-label="Grouping" onValueChange={value => {if (value === "none" || value === "short" || value === "long") setGrouping(value);}} className="rounded-md border px-1">
          {GROUPINGS.map(({value, text}) => <ToggleGroupItem key={value} value={value} aria-label={text} className="text-xs">{text}</ToggleGroupItem>)}
        </ToggleGroup>
      </div>
      {grouping === "long" && <div className="flex items-center gap-2">
        <span>Time window</span>
        <Select value={timeWindow.value} onValueChange={value => setTimeWindow({value, since: (() => { const s = WINDOWS.find(w => w.value === value)?.seconds; return s ? Math.floor(Date.now() / 1000) - s : undefined; })()})}>
          <SelectTrigger aria-label="Time window" className="h-8 w-36 text-xs"><SelectValue /></SelectTrigger>
          <SelectContent>{WINDOWS.map(w => <SelectItem key={w.value} value={w.value} className="text-xs">{w.text}</SelectItem>)}</SelectContent>
        </Select>
      </div>}
      <span>{grouping === "short" ? `${examples.length} loaded · groups same camera, kind and classes within ${gapSeconds}s; each decision keeps its own verdict`
        : grouping === "long" ? "Groups every saved decision by camera, kind, outcome, class, place on an 8 × 6 grid and box size; touching grid cells merge"
        : `${examples.length} loaded · every decision on its own card`}</span>
    </div>
    {grouping === "long" && !drill && <CorrectionGroupsView params={longParams} onOpen={setDrill} />}
    {drill && <div data-testid="group-drill-in" className="mb-2 flex shrink-0 flex-wrap items-center gap-2 rounded-lg border p-2 text-xs">
      <Button size="sm" variant="outline" onClick={() => setDrill(null)}>Back to groups</Button>
      <span className="font-semibold">{groupTitle(drill)}</span>
      <span>{groupDetail(drill)}</span>
    </div>}
    {showList && <>
      {!data && !error && <p>Loading corrections…</p>}
      {data && examples.length === 0 && <p>There are no corrections matching these filters</p>}
      <div data-testid="corrections-scroll" className="min-h-0 flex-1 overflow-y-auto pb-4">
        <div className="grid auto-rows-max grid-cols-1 content-start items-start gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {groups.map(items => <CorrectionGroup key={items[0].id} items={items} gapSeconds={gapSeconds} onMark={onMark} onOpen={openLightbox} />)}
        </div>
        {last?.has_more && <Button className="my-3" disabled={isValidating} onClick={() => setSize(size + 1)}>Load more examples</Button>}
        {isValidating && <span className="p-2 text-xs">Updating…</span>}
      </div>
    </>}
    {shown && <CorrectionLightbox example={shown} position={shownIndex} count={order.length} onStep={step} onMark={onMark} onClose={() => setLightbox(null)} />}
  </div>;
}

const VERDICTS: {value: SecondlookVerdict; text: string; color: string | null}[] = [
  {value: "primary_right", text: "Primary right", color: PRIMARY_COLOR},
  {value: "secondlook_right", text: "Secondlook right", color: SECONDLOOK_COLOR},
  {value: "ignore", text: "Ignore", color: null},
];

function VerdictButtons({example: ex, onMark}: {example: SecondlookExample; onMark: OnMark}) {
  const answers: Record<SecondlookVerdict, string> = {
    primary_right: primaryAnswer(ex),
    secondlook_right: secondlookAnswer(ex),
    ignore: "exclude from training",
  };
  return <div role="group" aria-label="Verdict" className="grid grid-cols-3 gap-1 pt-1">
    {VERDICTS.map(({value, text, color}) => {
      const selected = ex.mark === value;
      return <Button
        key={value}
        aria-label={text}
        aria-pressed={selected}
        variant="outline"
        className="h-auto min-h-10 flex-col gap-0 whitespace-normal px-1 py-1 text-xs leading-tight"
        style={color ? {borderColor: color, borderWidth: 2, ...(selected ? {backgroundColor: color, color: "#000"} : {})} : undefined}
        data-selected={selected}
        onClick={() => onMark(ex.id, selected ? null : value)}
      >
        <span className="font-semibold">{text}</span>
        <span className="break-words opacity-80">{answers[value]}</span>
      </Button>;
    })}
  </div>;
}

function stillUrl(ex: SecondlookExample): string {
  return `${baseUrl}api/secondlook/save_and_review/${ex.id}/still.jpg`;
}
function secondlookClass(ex: SecondlookExample): string {
  return ex.misclass_label ?? ex.evidence?.secondary_label ?? (ex.sl_score == null ? "No matching detection" : ex.label);
}
function resolveLightboxIndex(order: SecondlookExample[], current: {id: string; index: number}): number {
  const index = order.findIndex(ex => ex.id === current.id);
  if (index >= 0 || order.length === 0) return index;
  return Math.min(current.index, order.length - 1);
}

export function CorrectionCard({example: ex, onMark, onOpen}: {example: SecondlookExample; onMark: OnMark; onOpen?: OnOpen}) {
  const e = ex.evidence;
  return <article data-testid="correction-card" data-mark={ex.mark ?? ""} className="flex shrink-0 flex-col overflow-hidden rounded-lg border bg-background">
    <button type="button" data-lightbox-trigger={ex.id} aria-label={`Open full size: ${KIND_LABELS[ex.kind]}, ${ex.label} on ${ex.camera}`} disabled={!onOpen} onClick={() => onOpen?.(ex.id)} className="relative aspect-video w-full shrink-0 cursor-zoom-in bg-black focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:cursor-default">
      <img src={stillUrl(ex)} alt={`${ex.label} ${KIND_LABELS[ex.kind]}`} className="absolute inset-0 size-full object-contain" loading="lazy" />
      {e && e.version === 1 && <svg role="img" aria-label="Primary and Secondlook boxes" className="pointer-events-none absolute inset-0 size-full" viewBox={`0 0 ${e.width} ${e.height}`} preserveAspectRatio="xMidYMid meet">
        {[{box: e.primary_box, color: PRIMARY_COLOR}, {box: e.sl_box, color: SECONDLOOK_COLOR}].map(({box, color}, index) => box?.length === 4 ? <rect key={index} x={box[0]} y={box[1]} width={box[2]-box[0]} height={box[3]-box[1]} fill="none" stroke={color} strokeWidth={3} /> : null)}
      </svg>}
    </button>
    <div className="flex shrink-0 flex-col gap-1 p-2 text-sm">
      <div className="flex flex-wrap items-center gap-1.5"><strong>{KIND_LABELS[ex.kind]}</strong><LiveOutcomeBadge outcome={ex.live_outcome} /></div>
      <div className="break-words">Primary: {ex.label} · <span>P {formatScore(ex.primary_score)}</span></div>
      <div className="break-words">Secondlook: {secondlookClass(ex)} · <span>SL {formatScore(ex.sl_score)}</span></div>
      <div>Factor <span>× {formatScore(ex.factor)}</span> · {ex.camera}</div>
      <div className="text-xs text-muted-foreground">{e?.version === 1 ? "Cyan: primary · Orange: Secondlook · same analysed frame" : "Overlay unavailable: historical frame alignment not established"}</div>
      <div className="break-words text-xs">{ex.frame_time ? new Date(ex.frame_time * 1000).toLocaleString() : "Time unavailable"}</div>
      <VerdictButtons example={ex} onMark={onMark} />
      {ex.legacy_mark && <div className="text-xs text-muted-foreground">Migrated from earlier “{ex.legacy_mark}” mark</div>}
    </div>
  </article>;
}

function LiveOutcomeBadge({outcome}: {outcome?: LiveOutcome}) {
  if (!outcome || outcome === "none") return null;
  const badge = LIVE_OUTCOME_BADGES[outcome];
  const pill = <span data-testid="live-outcome" data-outcome={outcome} tabIndex={badge.reason ? 0 : undefined} className={badge.alarm ? "rounded-full bg-red-600 px-2 py-0.5 text-xs font-medium text-white" : "rounded-full bg-emerald-600/15 px-2 py-0.5 text-xs font-medium text-emerald-700 dark:text-emerald-300"}>{badge.text}</span>;
  if (!badge.reason) return pill;
  return <Tooltip><TooltipTrigger asChild>{pill}</TooltipTrigger><TooltipContent className="max-w-xs">{badge.reason}</TooltipContent></Tooltip>;
}

function CorrectionGroup({items, gapSeconds, onMark, onOpen}: {items: SecondlookExample[]; gapSeconds: number; onMark: OnMark; onOpen: OnOpen}) {
  const [expanded, setExpanded] = useState(false);
  if (items.length === 1) return <CorrectionCard example={items[0]} onMark={onMark} onOpen={onOpen} />;
  const times = items.map(exampleTime);
  const span = Math.round(Math.max(...times) - Math.min(...times));
  const marked = items.filter(ex => ex.mark !== null).length;
  // members keep their own outcome; surface alarms hidden behind the collapsed card
  const notActed = items.filter(ex => ex.live_outcome === "not_rescued" || ex.live_outcome === "not_suppressed").length;
  return <section className="flex flex-col gap-2 rounded-lg border p-2" data-testid="correction-group">
    <strong>{items.length} separate decisions · {KIND_LABELS[items[0].kind]}</strong>
    <span className="text-xs">{items[0].camera}: primary {items[0].label} · Secondlook {secondlookAnswer(items[0])} · over {span}s (gaps ≤ {gapSeconds}s) · {marked}/{items.length} with a verdict</span>
    {notActed > 0 && <span data-testid="group-not-acted" className="w-fit rounded-full bg-red-600 px-2 py-0.5 text-xs font-medium text-white">{notActed} of {items.length} not acted on live (alert unchanged)</span>}
    {!expanded && <CorrectionCard example={items[0]} onMark={onMark} onOpen={onOpen} />}
    <Button variant="outline" onClick={() => setExpanded(!expanded)}>{expanded ? "Collapse examples" : `Expand ${items.length} examples`}</Button>
    {expanded && items.map(ex => <CorrectionCard key={ex.id} example={ex} onMark={onMark} onOpen={onOpen} />)}
  </section>;
}

/** Is the point on the visible picture of an object-contain image (not its letterbox)? */
function insideContainedImage(img: HTMLImageElement, x: number, y: number): boolean {
  const rect = img.getBoundingClientRect();
  const fit = img.naturalWidth && img.naturalHeight ? Math.min(rect.width / img.naturalWidth, rect.height / img.naturalHeight) : 0;
  const width = img.naturalWidth * fit;
  const height = img.naturalHeight * fit;
  const left = rect.left + (rect.width - width) / 2;
  const top = rect.top + (rect.height - height) / 2;
  return x >= left && x <= left + width && y >= top && y <= top + height;
}

/** Box with its label and score drawn in image pixels; the label sits above (primary) or below (Secondlook). */
function LabelledBox({box, color, text, below, fontSize}: {box: number[]; color: string; text: string; below: boolean; fontSize: number}) {
  const [x1, y1, x2, y2] = box;
  const y = below ? y2 + fontSize * 1.1 : y1 - fontSize * 0.35;
  return <g>
    <rect x={x1} y={y1} width={x2 - x1} height={y2 - y1} fill="none" stroke={color} strokeWidth={Math.max(2, fontSize / 6)} />
    <text x={x1} y={y} fill={color} fontSize={fontSize} fontWeight={600} stroke="#000" strokeWidth={fontSize / 5} paintOrder="stroke">{text}</text>
  </g>;
}

/**
 * Full-size view of one saved item. Radix Dialog gives the focus trap, Esc and
 * focus restore; arrows/swipe step through the loaded, filtered list. Marking
 * keeps the item open (see CorrectionsView for when it leaves the list).
 */
function CorrectionLightbox({example: ex, position, count, onStep, onMark, onClose}: {
  example: SecondlookExample; position: number; count: number;
  onStep: (delta: number) => void; onMark: OnMark; onClose: () => void;
}) {
  const e = ex.evidence?.version === 1 ? ex.evidence : null;
  const opener = useRef<Element | null>(null);
  if (opener.current === null) opener.current = document.activeElement;
  const scale = useRef(1);
  const swipe = useSwipeable({
    onSwipedLeft: () => { if (scale.current <= 1.01) onStep(1); },
    onSwipedRight: () => { if (scale.current <= 1.01) onStep(-1); },
    delta: 40,
  });
  const fontSize = e ? Math.max(14, Math.round(e.width / 40)) : 14;
  // A click on the dark area around the picture (letterbox included) closes; on the picture it does not.
  const closeOnBackdrop = (event: React.MouseEvent<HTMLDivElement>) => {
    if (!(event.target instanceof Element) || event.target.closest("button") || scale.current > 1.01) return;
    const img = event.currentTarget.querySelector<HTMLImageElement>("[data-testid=lightbox-image]");
    if (!img || !insideContainedImage(img, event.clientX, event.clientY)) onClose();
  };
  return <DialogPrimitive.Root open onOpenChange={open => { if (!open) onClose(); }}>
    <DialogPortal>
      <DialogOverlay className="bg-black/90 backdrop-blur-none" />
      <DialogPrimitive.Content
        data-testid="correction-lightbox"
        aria-describedby="correction-lightbox-help"
        className="fixed inset-0 z-50 flex flex-col bg-black text-foreground outline-none md:flex-row"
        onKeyDown={event => {
          if (event.key === "ArrowLeft") { event.preventDefault(); onStep(-1); }
          if (event.key === "ArrowRight") { event.preventDefault(); onStep(1); }
        }}
        onCloseAutoFocus={event => {
          // Return focus to the item now shown if its card is on screen, else to the opener.
          const trigger = document.querySelector<HTMLElement>(`[data-lightbox-trigger="${CSS.escape(ex.id)}"]`) ?? (opener.current instanceof HTMLElement ? opener.current : null);
          if (trigger) { event.preventDefault(); trigger.focus(); }
        }}
      >
        <div data-testid="lightbox-stage" className="relative min-h-0 flex-1" onClick={closeOnBackdrop} {...swipe}>
          <TransformWrapper key={ex.id} minScale={1} maxScale={8} wheel={{ smoothStep: 0.005 }} doubleClick={{ mode: "toggle" }} onTransformed={(_, state) => { scale.current = state.scale; }}>
            <TransformComponent wrapperStyle={{ width: "100%", height: "100%" }} contentStyle={{ width: "100%", height: "100%" }}>
              <div className="relative size-full">
                <img data-testid="lightbox-image" src={stillUrl(ex)} alt={`${ex.label} ${KIND_LABELS[ex.kind]}, full size`} className="absolute inset-0 size-full object-contain" />
                {e && <svg role="img" aria-label={`Primary box ${ex.label} ${formatScore(ex.primary_score)}, Secondlook box ${secondlookClass(ex)} ${formatScore(ex.sl_score)}`} className="pointer-events-none absolute inset-0 size-full" viewBox={`0 0 ${e.width} ${e.height}`} preserveAspectRatio="xMidYMid meet">
                  {e.primary_box?.length === 4 && <LabelledBox box={e.primary_box} color={PRIMARY_COLOR} text={`Primary ${ex.label} ${formatScore(ex.primary_score)}`} below={false} fontSize={fontSize} />}
                  {e.sl_box?.length === 4 && <LabelledBox box={e.sl_box} color={SECONDLOOK_COLOR} text={`Secondlook ${secondlookClass(ex)} ${formatScore(ex.sl_score)}`} below fontSize={fontSize} />}
                </svg>}
              </div>
            </TransformComponent>
          </TransformWrapper>
          {position > 0 && <Button size="icon" variant="secondary" aria-label="Previous item" className="absolute left-2 top-1/2 -translate-y-1/2 rounded-full opacity-80" onClick={() => onStep(-1)}><ChevronLeft /></Button>}
          {position < count - 1 && <Button size="icon" variant="secondary" aria-label="Next item" className="absolute right-2 top-1/2 -translate-y-1/2 rounded-full opacity-80" onClick={() => onStep(1)}><ChevronRight /></Button>}
        </div>
        <aside data-testid="lightbox-panel" className="flex max-h-[45%] shrink-0 flex-col gap-1.5 overflow-y-auto bg-background p-3 text-sm md:max-h-none md:w-80">
          <div className="flex items-start justify-between gap-2">
            <DialogPrimitive.Title className="flex flex-wrap items-center gap-1.5 text-base font-semibold">
              {KIND_LABELS[ex.kind]}<LiveOutcomeBadge outcome={ex.live_outcome} />
            </DialogPrimitive.Title>
            <DialogPrimitive.Close asChild><Button size="icon" variant="ghost" aria-label="Close" className="size-8 shrink-0"><X className="size-4" /></Button></DialogPrimitive.Close>
          </div>
          <span data-testid="lightbox-position" className="text-xs text-muted-foreground">{position + 1} of {count} loaded</span>
          <div className="break-words"><span style={{color: PRIMARY_COLOR}} aria-hidden>■ </span>Primary: {ex.label} · P {formatScore(ex.primary_score)}</div>
          <div className="break-words"><span style={{color: SECONDLOOK_COLOR}} aria-hidden>■ </span>Secondlook: {secondlookClass(ex)} · SL {formatScore(ex.sl_score)}</div>
          <div>Factor × {formatScore(ex.factor)} · {ex.camera}</div>
          <div className="text-xs">{ex.frame_time ? new Date(ex.frame_time * 1000).toLocaleString() : "Time unavailable"}</div>
          {!e && <div className="text-xs text-muted-foreground">Overlay unavailable: historical frame alignment not established</div>}
          <VerdictButtons example={ex} onMark={onMark} />
          <p id="correction-lightbox-help" className="text-xs text-muted-foreground">Arrow keys or swipe: previous/next. Pinch, scroll or double-click to zoom. Esc or a click outside the image closes.</p>
        </aside>
      </DialogPrimitive.Content>
    </DialogPortal>
  </DialogPrimitive.Root>;
}
