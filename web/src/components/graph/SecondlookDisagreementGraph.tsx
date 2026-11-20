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

// stack order bottom-up; the two "not" parts (Secondlook disagreed but the
// alert was not changed) use alarm colours, the rest the system palette
const KINDS = [
  ["fp_suppressed", "FP suppressed", "#5C7CFA"],
  ["fp_not_suppressed", "FP not suppressed", "#F97316"],
  ["fn_rescued", "FN rescued", "#74C0FC"],
  ["fn_not_rescued", "FN not rescued", "#DC2626"],
  ["misclass", "misclassifications", "#FAD75C"],
] as const satisfies readonly (readonly [
  keyof SecondlookHour,
  string,
  string,
])[];
const COLORS = KINDS.map(([, , color]) => color);

type View = "counts" | "rates";

// What each rate divides, in plain words; every line shares the denominator.
const RATE_NUMERATORS: Record<(typeof KINDS)[number][0], string> = {
  fp_suppressed:
    "tracked objects the primary detector would have made a detection, where Secondlook's lower score kept them from becoming one",
  fp_not_suppressed:
    "tracked objects where Secondlook's lower score would have kept them from becoming a detection, but its answer came too late and they became one anyway",
  fn_rescued:
    "tracked objects the primary detector scored below the threshold, where Secondlook's higher score made them become a detection",
  fn_not_rescued:
    "tracked objects where Secondlook's higher score would have made them become a detection, but its answer came too late, so they did not",
  misclass:
    "tracked objects where Secondlook saw a different class in the same place (review only; nothing is changed)",
};

/** "1.03%": two decimals, no space; a dash for hours with nothing analysed. */
function formatRate(rate: number | null): string {
  return rate === null ? "–" : `${rate.toFixed(2)}%`;
}

function rate(count: number, analysed: number): number | null {
  // no objects analysed: a gap, not 0%
  return analysed > 0 ? Math.round((10000 * count) / analysed) / 100 : null;
}

/** Axis tick: "2026-11-11T12:00+01:00" -> "11-11" (server-local date). */
function tickDate(localHour: string): string {
  return localHour.slice(5, 10);
}

export function SecondlookDisagreementGraph() {
  const { data, error } = useSWR<SecondlookHourly>(
    "secondlook/disagreements/hourly",
    { refreshInterval: 60_000, revalidateOnFocus: false },
  );
  const [view, setView] = useState<View>("counts");
  const { theme, systemTheme } = useTheme();
  const hours = useMemo(() => data?.hours ?? [], [data]);

  const series = useMemo(
    () =>
      KINDS.map(([kind, name]) => ({
        name,
        data: hours.map((hour) =>
          view === "counts"
            ? hour[kind]
            : rate(hour[kind], hour.objects_analysed),
        ),
      })),
    [hours, view],
  );

  // week totals beside each series name, in the selected unit
  const totals = useMemo(() => {
    const analysed = hours.reduce((sum, h) => sum + h.objects_analysed, 0);
    return KINDS.map(([kind]) => {
      const count = hours.reduce((sum, h) => sum + h[kind], 0);
      return view === "counts"
        ? count.toString()
        : formatRate(rate(count, analysed));
    });
  }, [hours, view]);

  const options = useMemo(
    () =>
      ({
        chart: {
          id: "secondlook-disagreements",
          stacked: true,
          selection: { enabled: false },
          toolbar: { show: false },
          zoom: { enabled: false },
          animations: { enabled: false },
        },
        colors: COLORS,
        grid: { show: false },
        legend: { show: false },
        dataLabels: { enabled: false },
        plotOptions: { bar: { columnWidth: "90%" } },
        states: { active: { filter: { type: "none" } } },
        tooltip: {
          theme: systemTheme || theme,
          shared: true,
          intersect: false,
          x: {
            // the hovered column's own hour, not the category text
            formatter: (_value: number, opts?: { dataPointIndex?: number }) => {
              const hour = hours[opts?.dataPointIndex ?? -1];
              return hour
                ? `${hour.local_hour.slice(0, 10)} ${hour.local_hour.slice(11, 16)} · ${hour.objects_analysed} objects analysed`
                : "";
            },
          },
          y: {
            // rates show their own numerator and denominator: "1.03% (1 of 97)"
            formatter: (
              val: number | null,
              opts?: { seriesIndex?: number; dataPointIndex?: number },
            ) => {
              if (view === "counts") return `${val ?? 0}`;
              const hour = hours[opts?.dataPointIndex ?? -1];
              const kind = KINDS[opts?.seriesIndex ?? -1]?.[0];
              if (!hour || !kind) return formatRate(val);
              return `${formatRate(val)} (${hour[kind]} of ${hour.objects_analysed})`;
            },
          },
        },
        xaxis: {
          type: "category",
          // one dated tick per day; the tooltip carries the hour
          categories: hours.map((hour) => tickDate(hour.local_hour)),
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
        yaxis: {
          show: true,
          min: 0,
          forceNiceScale: true,
          labels: {
            formatter: (val: number) =>
              view === "counts" ? Math.round(val).toString() : formatRate(val),
            style: { colors: "#6B6B6B" },
          },
        },
      }) as ApexCharts.ApexOptions,
    [hours, view, theme, systemTheme],
  );

  if (error) {
    return (
      <p className="text-sm text-muted-foreground">
        Hourly disagreements need Secondlook save and review enabled.
      </p>
    );
  }

  return (
    <div className="flex w-full flex-col">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex flex-wrap items-center gap-2.5">
          {KINDS.map(([kind, name, color], index) => (
            <div key={kind} className="flex items-center gap-1">
              <MdCircle className="size-2" style={{ color }} />
              <div className="text-xs text-secondary-foreground">{name}</div>
              <div className="text-xs text-primary">{totals[index]}</div>
            </div>
          ))}
        </div>
        <div
          className="flex items-center gap-1"
          role="group"
          aria-label="Chart values"
        >
          <Popover>
            <PopoverTrigger asChild>
              <button
                className="focus:outline-none"
                aria-label="About the rates"
              >
                <CiCircleAlert className="size-5" />
              </button>
            </PopoverTrigger>
            <PopoverContent
              data-testid="rates-info"
              className="w-96 max-w-[90vw] text-xs"
            >
              <p className="mb-1 font-medium">Rates: numerator ÷ denominator</p>
              <p className="mb-2">
                Denominator, the same for every line: tracked objects Secondlook
                analysed in that hour (each counted once, in the hour of its
                first Secondlook answer). Hours with none are left as gaps.
                Numerator, per line:
              </p>
              <ul className="mb-2 list-disc space-y-1 pl-4">
                {KINDS.map(([kind, name]) => (
                  <li key={kind}>
                    <span className="font-medium">{name}:</span>{" "}
                    {RATE_NUMERATORS[kind]}.
                  </li>
                ))}
              </ul>
              <p className="mb-1">
                FP and FN numerators are counted in the hour the tracked object
                ended (when the outcome is final), misclassifications in the
                hour Secondlook first saw them, so an object that spans two
                hours can land in a different hour from its analysis.
              </p>
              <p>
                The tooltip shows both: 1.03% (1 of 97) means 1 of 97 analysed
                objects. The totals beside the names use the whole week.
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
        type="bar"
        options={options}
        series={series}
        height={isMobileOnly ? 160 : 180}
      />
    </div>
  );
}
