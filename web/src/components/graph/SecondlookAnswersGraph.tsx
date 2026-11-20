import { useTheme } from "@/context/theme-provider";
import { Button } from "@/components/ui/button";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { CiCircleAlert } from "react-icons/ci";
import { SecondlookHour, SecondlookHourly } from "@/types/stats";
import { useMemo, useState } from "react";
import Chart from "react-apexcharts";
import { isMobileOnly } from "react-device-detect";
import { MdCircle } from "react-icons/md";
import useSWR from "swr";

type AnswerKey = "agreements" | "no_hit" | "timeouts" | "dropped" | "fail_open";

// lines, not a stack: agreements overlap results and no hit
const LINES = [
  ["agreements", "agreed", "#22C55E"],
  ["no_hit", "no hit", "#5C7CFA"],
  ["timeouts", "timed out", "#F97316"],
  ["dropped", "dropped", "#DC2626"],
  ["fail_open", "failed open", "#A855F7"],
] as const satisfies readonly (readonly [AnswerKey, string, string])[];
const COLORS = LINES.map(([, , color]) => color);

type View = "counts" | "rates";

// numerator per line; every line shares the denominator (tracks_finished)
const NUMERATORS: Record<AnswerKey, string> = {
  agreements:
    "tracked objects where Secondlook gave a factor and it did not change whether they became a detection",
  no_hit:
    "tracked objects where Secondlook analysed the crop and found no box of the same class",
  timeouts:
    "tracked objects whose Secondlook request got no answer before the object ended, nor in the 15 s after",
  dropped:
    "tracked objects whose first Secondlook request was refused because its queue was full, with no later answer",
  fail_open:
    "tracked objects whose answers had no real inference behind them (model refused or unavailable, or inference failed)",
};

function value(hour: SecondlookHour, key: AnswerKey | "tracks_finished") {
  // responses from earlier builds have no answer counts
  return hour[key] ?? 0;
}

/** "1.03%": two decimals, no space; a dash for hours with nothing finished. */
function formatRate(rate: number | null): string {
  return rate === null ? "–" : `${rate.toFixed(2)}%`;
}

function rate(count: number, finished: number): number | null {
  return finished > 0 ? Math.round((10000 * count) / finished) / 100 : null;
}

export function SecondlookAnswersGraph() {
  const { data, error } = useSWR<SecondlookHourly>(
    "secondlook/disagreements/hourly",
    { refreshInterval: 60_000, revalidateOnFocus: false },
  );
  const [view, setView] = useState<View>("rates");
  const { theme, systemTheme } = useTheme();
  const hours = useMemo(() => data?.hours ?? [], [data]);
  const finishedTotal = useMemo(
    () => hours.reduce((sum, h) => sum + value(h, "tracks_finished"), 0),
    [hours],
  );

  const series = useMemo(
    () =>
      LINES.map(([key, name]) => ({
        name,
        data: hours.map((hour) =>
          view === "counts"
            ? value(hour, key)
            : rate(value(hour, key), value(hour, "tracks_finished")),
        ),
      })),
    [hours, view],
  );

  const totals = useMemo(
    () =>
      LINES.map(([key]) => {
        const count = hours.reduce((sum, h) => sum + value(h, key), 0);
        return view === "counts"
          ? count.toString()
          : formatRate(rate(count, finishedTotal));
      }),
    [hours, view, finishedTotal],
  );

  const options = useMemo(() => {
    // no axis titles on phones: there is no room beside the labels (Apex
    // needs the title object, so only the text is left out)
    const axisTitle = (text: string) => ({
      text: isMobileOnly ? undefined : text,
      style: { color: "#6B6B6B" },
    });
    const axisLabel = (val: number) =>
      view === "counts" ? Math.round(val).toString() : formatRate(val);
    return {
      chart: {
        id: "secondlook-answers",
        selection: { enabled: false },
        toolbar: { show: false },
        zoom: { enabled: false },
        animations: { enabled: false },
      },
      colors: COLORS,
      stroke: { width: 1.5, curve: "straight" },
      grid: { show: false },
      legend: { show: false },
      dataLabels: { enabled: false },
      // small dots, so an hour between gaps (or the only hour) still shows
      markers: { size: 2, strokeWidth: 0 },
      tooltip: {
        theme: systemTheme || theme,
        shared: true,
        intersect: false,
        x: {
          formatter: (_value: number, opts?: { dataPointIndex?: number }) => {
            const hour = hours[opts?.dataPointIndex ?? -1];
            return hour
              ? `${hour.local_hour.slice(0, 10)} ${hour.local_hour.slice(11, 16)} · ${value(hour, "tracks_finished")} tracked objects finished`
              : "";
          },
        },
        y: {
          formatter: (
            val: number | null,
            opts?: { seriesIndex?: number; dataPointIndex?: number },
          ) => {
            if (view === "counts") return `${val ?? 0}`;
            const hour = hours[opts?.dataPointIndex ?? -1];
            const key = LINES[opts?.seriesIndex ?? -1]?.[0];
            if (!hour || !key) return formatRate(val);
            return `${formatRate(val)} (${value(hour, key)} of ${value(hour, "tracks_finished")})`;
          },
        },
      },
      xaxis: {
        type: "category",
        categories: hours.map((hour) => hour.local_hour.slice(5, 10)),
        tickAmount: 7,
        labels: {
          rotate: 0,
          hideOverlappingLabels: true,
          style: { colors: "#6B6B6B" },
        },
        axisBorder: { show: false },
        axisTicks: { show: false },
        tooltip: { enabled: false },
      },
      // agreed (usually near 100%) on the left axis, the four no-answer
      // lines (usually a few %) on their own right axis
      yaxis: [
        {
          seriesName: LINES[0][1],
          min: 0,
          forceNiceScale: true,
          title: axisTitle("agreed"),
          labels: { formatter: axisLabel, style: { colors: "#6B6B6B" } },
        },
        {
          seriesName: LINES.slice(1).map(([, name]) => name),
          opposite: true,
          min: 0,
          forceNiceScale: true,
          title: axisTitle("no answer used"),
          labels: { formatter: axisLabel, style: { colors: "#6B6B6B" } },
        },
      ],
    } as ApexCharts.ApexOptions;
  }, [hours, view, theme, systemTheme]);

  if (error) {
    return (
      <p className="text-sm text-muted-foreground">
        Secondlook answers need Secondlook save and review enabled.
      </p>
    );
  }

  return (
    <div className="flex w-full flex-col">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex flex-wrap items-center gap-2.5">
          {LINES.map(([key, name, color], index) => (
            <div key={key} className="flex items-center gap-1">
              <MdCircle className="size-2" style={{ color }} />
              <div className="text-xs text-secondary-foreground">{name}</div>
              <div className="text-xs text-primary">{totals[index]}</div>
            </div>
          ))}
          <div className="text-xs text-muted-foreground">
            of {finishedTotal} finished
          </div>
        </div>
        <div
          className="flex items-center gap-1"
          role="group"
          aria-label="Answer chart values"
        >
          <Popover>
            <PopoverTrigger asChild>
              <button
                className="focus:outline-none"
                aria-label="About the answer rates"
              >
                <CiCircleAlert className="size-5" />
              </button>
            </PopoverTrigger>
            <PopoverContent
              data-testid="answers-info"
              className="w-96 max-w-[90vw] text-xs"
            >
              <p className="mb-1 font-medium">Rates: numerator ÷ denominator</p>
              <p className="mb-2">
                Denominator, the same for every line: tracked objects Secondlook
                was asked about that finished in that hour (each counted once,
                when it ended, or up to 15 s later if its answer was still on
                its way). Hours with none are left as gaps. Numerator, per line:
              </p>
              <ul className="mb-2 list-disc space-y-1 pl-4">
                {LINES.map(([key, name]) => (
                  <li key={key}>
                    <span className="font-medium">{name}:</span>{" "}
                    {NUMERATORS[key]}.
                  </li>
                ))}
              </ul>
              <p className="mb-1">
                No hit, timed out, dropped and failed open do not overlap.
                Agreed can include no-hit objects when a no-hit factor is set.
                Agreed uses the left axis, the other four the right axis.
                Objects with a disagreement are in the chart above.
              </p>
              <p>
                The tooltip shows both: 98.97% (96 of 97) means 96 of 97
                finished objects. The totals beside the names use the whole
                week.
              </p>
            </PopoverContent>
          </Popover>
          {(["counts", "rates"] as const).map((option) => (
            <Button
              key={option}
              size="sm"
              className="h-7 px-2 text-xs"
              variant={view === option ? "select" : "default"}
              aria-pressed={view === option}
              onClick={() => setView(option)}
            >
              {option === "counts" ? "Counts" : "Rates"}
            </Button>
          ))}
        </div>
      </div>
      <Chart
        key={view}
        type="line"
        options={options}
        series={series}
        height={isMobileOnly ? 160 : 180}
      />
    </div>
  );
}
