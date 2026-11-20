import useSWR from "swr";
import { FrigateStats, GpuInfo, GpuStats } from "@/types/stats";
import { startTransition, useEffect, useMemo, useState } from "react";
import { useFrigateStats } from "@/api/ws";
import {
  DetectorCpuThreshold,
  DetectorMemThreshold,
  DetectorTempThreshold,
  GPUMemThreshold,
  GPUUsageThreshold,
  InferenceThreshold,
  SecondlookProcessingThreshold,
} from "@/types/graph";
import { Button } from "@/components/ui/button";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import GPUInfoDialog from "@/components/overlay/GPUInfoDialog";
import { Skeleton } from "@/components/ui/skeleton";
import { ThresholdBarGraph } from "@/components/graph/SystemGraph";
import { SecondlookDisagreementGraph } from "@/components/graph/SecondlookDisagreementGraph";
import { SecondlookAnswersGraph } from "@/components/graph/SecondlookAnswersGraph";
import { cn } from "@/lib/utils";
import { useTranslation } from "react-i18next";
import { CiCircleAlert } from "react-icons/ci";

type GeneralMetricsProps = {
  lastUpdated: number;
  setLastUpdated: (last: number) => void;
  isActive: boolean;
};
export default function GeneralMetrics({
  lastUpdated,
  setLastUpdated,
  isActive,
}: GeneralMetricsProps) {
  // extra info
  const { t } = useTranslation(["views/system"]);
  const [showVainfo, setShowVainfo] = useState(false);

  // stats

  const { data: initialStats, mutate: refreshStats } = useSWR<FrigateStats[]>(
    [
      "stats/history",
      {
        keys: "secondlook,detectors.inference_speed,detectors.temperature,detectors.cpu,detectors.mem,gpu_usages,npu_usages,processes.cpu,processes.mem,service.last_updated",
      },
    ],
    {
      revalidateOnFocus: false,
    },
  );

  const [statsHistory, setStatsHistory] = useState<FrigateStats[]>([]);
  const updatedStats = useFrigateStats();

  useEffect(() => {
    if (initialStats == undefined || initialStats.length == 0) {
      return;
    }

    if (statsHistory.length == 0) {
      startTransition(() => setStatsHistory(initialStats));
      return;
    }

    if (!isActive || !updatedStats) {
      return;
    }

    if (updatedStats.service.last_updated > lastUpdated) {
      setStatsHistory([...statsHistory.slice(1), updatedStats]);
      setLastUpdated(updatedStats.service.last_updated);
    }
  }, [
    initialStats,
    updatedStats,
    statsHistory,
    lastUpdated,
    setLastUpdated,
    isActive,
  ]);

  useEffect(() => {
    if (isActive && statsHistory.length > 0) {
      refreshStats().then((freshStats) => {
        if (freshStats && freshStats.length > 0) {
          setStatsHistory(freshStats);
        }
      });
    }
    // only re-fetch when tab becomes active, not on data changes
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isActive]);

  const [canGetGpuInfo, gpuType] = useMemo<[boolean, GpuInfo]>(() => {
    let vaCount = 0;
    let nvCount = 0;

    statsHistory.length > 0 &&
      Object.values(statsHistory[0]?.gpu_usages ?? {}).forEach((stats) => {
        if (stats.vendor === "nvidia") {
          nvCount += 1;
        } else if (stats.vendor === "intel" || stats.vendor === "amd") {
          vaCount += 1;
        }
      });

    return [vaCount > 0 || nvCount > 0, nvCount > 0 ? "nvinfo" : "vainfo"];
  }, [statsHistory]);

  // timestamps

  const updateTimes = useMemo(
    () => statsHistory.map((stats) => stats.service.last_updated),
    [statsHistory],
  );

  // detectors stats

  const detInferenceTimeSeries = useMemo(() => {
    if (!statsHistory) {
      return [];
    }

    const series: {
      [key: string]: { name: string; data: { x: number; y: number }[] };
    } = {};

    statsHistory.forEach((stats, statsIdx) => {
      if (!stats) {
        return;
      }

      Object.entries(stats.detectors).forEach(([key, stats]) => {
        if (!(key in series)) {
          series[key] = { name: key, data: [] };
        }

        series[key].data.push({ x: statsIdx + 1, y: stats.inference_speed });
      });
    });
    return Object.values(series);
  }, [statsHistory]);

  const detTempSeries = useMemo(() => {
    if (!statsHistory) {
      return undefined;
    }

    const series: {
      [key: string]: { name: string; data: { x: number; y: number }[] };
    } = {};

    statsHistory.forEach((stats, statsIdx) => {
      if (!stats) {
        return;
      }

      Object.entries(stats.detectors).forEach(([key, detectorStats]) => {
        if (detectorStats.temperature === undefined) {
          return;
        }

        if (!(key in series)) {
          series[key] = {
            name: key,
            data: [],
          };
        }

        series[key].data.push({
          x: statsIdx + 1,
          y: Math.round(detectorStats.temperature),
        });
      });
    });

    if (Object.keys(series).length > 0) {
      return Object.values(series);
    }

    return undefined;
  }, [statsHistory]);

  const detCpuSeries = useMemo(() => {
    if (!statsHistory) {
      return [];
    }

    const series: {
      [key: string]: { name: string; data: { x: number; y: string }[] };
    } = {};

    statsHistory.forEach((stats, statsIdx) => {
      if (!stats) {
        return;
      }

      Object.entries(stats.detectors).forEach(([key, detStats]) => {
        if (!(key in series)) {
          series[key] = { name: key, data: [] };
        }

        const data = detStats.cpu;

        if (data != undefined) {
          series[key].data.push({
            x: statsIdx + 1,
            y: data,
          });
        }
      });
    });
    return Object.values(series);
  }, [statsHistory]);

  const detMemSeries = useMemo(() => {
    if (!statsHistory) {
      return [];
    }

    const series: {
      [key: string]: { name: string; data: { x: number; y: string }[] };
    } = {};

    statsHistory.forEach((stats, statsIdx) => {
      if (!stats) {
        return;
      }

      Object.entries(stats.detectors).forEach(([key, detStats]) => {
        if (!(key in series)) {
          series[key] = { name: key, data: [] };
        }

        if (detStats.mem != undefined) {
          series[key].data.push({
            x: statsIdx + 1,
            y: detStats.mem,
          });
        }
      });
    });
    return Object.values(series);
  }, [statsHistory]);

  // gpu stats

  const gpuSeries = useMemo(() => {
    if (!statsHistory) {
      return [];
    }

    const series: {
      [key: string]: { name: string; data: { x: number; y: string }[] };
    } = {};
    let hasValidGpu = false;

    statsHistory.forEach((stats, statsIdx) => {
      if (!stats) {
        return;
      }

      Object.entries(stats.gpu_usages || {}).forEach(([key, stats]) => {
        if (!(key in series)) {
          series[key] = { name: key, data: [] };
        }

        if (stats.gpu) {
          hasValidGpu = true;
          series[key].data.push({ x: statsIdx + 1, y: stats.gpu.slice(0, -1) });
        }
      });
    });

    if (!hasValidGpu) {
      return [];
    }

    return Object.keys(series).length > 0 ? Object.values(series) : [];
  }, [statsHistory]);

  const gpuMemSeries = useMemo(() => {
    if (!statsHistory) {
      return [];
    }

    // Intel doesn't expose VRAM usage, so hide the memory section
    // entirely when every reporting GPU is Intel.
    const firstEntries: GpuStats[] = Object.values(
      statsHistory[0]?.gpu_usages ?? {},
    );
    if (
      firstEntries.length > 0 &&
      firstEntries.every((s) => s.vendor === "intel")
    ) {
      return undefined;
    }

    const series: {
      [key: string]: { name: string; data: { x: number; y: string }[] };
    } = {};
    let hasValidGpu = false;

    statsHistory.forEach((stats, statsIdx) => {
      if (!stats) {
        return;
      }

      Object.entries(stats.gpu_usages || {}).forEach(([key, stats]) => {
        if (stats.vendor === "intel") {
          return;
        }

        if (!(key in series)) {
          series[key] = { name: key, data: [] };
        }

        if (stats.mem) {
          hasValidGpu = true;
          series[key].data.push({ x: statsIdx + 1, y: stats.mem.slice(0, -1) });
        }
      });
    });

    if (!hasValidGpu) {
      return [];
    }

    return Object.values(series);
  }, [statsHistory]);

  const gpuEncSeries = useMemo(() => {
    if (!statsHistory) {
      return [];
    }

    const series: {
      [key: string]: { name: string; data: { x: number; y: string }[] };
    } = {};
    let hasValidGpu = false;

    statsHistory.forEach((stats, statsIdx) => {
      if (!stats) {
        return;
      }

      Object.entries(stats.gpu_usages || {}).forEach(([key, stats]) => {
        if (!(key in series)) {
          series[key] = { name: key, data: [] };
        }

        if (stats.enc) {
          hasValidGpu = true;
          series[key].data.push({ x: statsIdx + 1, y: stats.enc.slice(0, -1) });
        }
      });
    });

    if (!hasValidGpu) {
      return [];
    }

    return Object.keys(series).length > 0 ? Object.values(series) : undefined;
  }, [statsHistory]);

  const gpuComputeSeries = useMemo(() => {
    if (!statsHistory) {
      return [];
    }

    const series: {
      [key: string]: { name: string; data: { x: number; y: string }[] };
    } = {};
    let hasValidGpu = false;

    statsHistory.forEach((stats, statsIdx) => {
      if (!stats) {
        return;
      }

      Object.entries(stats.gpu_usages || {}).forEach(([key, stats]) => {
        if (!(key in series)) {
          series[key] = { name: key, data: [] };
        }

        if (stats.compute) {
          hasValidGpu = true;
          series[key].data.push({
            x: statsIdx + 1,
            y: stats.compute.slice(0, -1),
          });
        }
      });
    });

    if (!hasValidGpu) {
      return [];
    }

    return Object.keys(series).length > 0 ? Object.values(series) : undefined;
  }, [statsHistory]);

  const gpuDecSeries = useMemo(() => {
    if (!statsHistory) {
      return [];
    }

    const series: {
      [key: string]: { name: string; data: { x: number; y: string }[] };
    } = {};
    let hasValidGpu = false;

    statsHistory.forEach((stats, statsIdx) => {
      if (!stats) {
        return;
      }

      Object.entries(stats.gpu_usages || {}).forEach(([key, stats]) => {
        if (!(key in series)) {
          series[key] = { name: key, data: [] };
        }

        if (stats.dec) {
          hasValidGpu = true;
          series[key].data.push({ x: statsIdx + 1, y: stats.dec.slice(0, -1) });
        }
      });
    });

    if (!hasValidGpu) {
      return [];
    }

    return Object.keys(series).length > 0 ? Object.values(series) : undefined;
  }, [statsHistory]);

  const gpuTempSeries = useMemo(() => {
    if (!statsHistory) {
      return [];
    }

    const series: {
      [key: string]: { name: string; data: { x: number; y: number }[] };
    } = {};
    let hasValidGpu = false;

    statsHistory.forEach((stats, statsIdx) => {
      if (!stats) {
        return;
      }

      Object.entries(stats.gpu_usages || {}).forEach(([key, stats]) => {
        if (!(key in series)) {
          series[key] = { name: key, data: [] };
        }

        if (stats?.temp !== undefined) {
          hasValidGpu = true;
          series[key].data.push({ x: statsIdx + 1, y: stats.temp });
        }
      });
    });

    if (!hasValidGpu) {
      return [];
    }

    return Object.keys(series).length > 0 ? Object.values(series) : undefined;
  }, [statsHistory]);

  // npu stats

  const npuSeries = useMemo(() => {
    if (!statsHistory) {
      return [];
    }

    const series: {
      [key: string]: { name: string; data: { x: number; y: number }[] };
    } = {};
    let hasValidNpu = false;

    statsHistory.forEach((stats, statsIdx) => {
      if (!stats) {
        return;
      }

      Object.entries(stats.npu_usages || {}).forEach(([key, stats]) => {
        if (!(key in series)) {
          series[key] = { name: key, data: [] };
        }

        if (stats?.npu) {
          hasValidNpu = true;
          series[key].data.push({ x: statsIdx + 1, y: stats.npu });
        }
      });
    });

    if (!hasValidNpu) {
      return [];
    }

    return Object.keys(series).length > 0 ? Object.values(series) : [];
  }, [statsHistory]);

  const npuTempSeries = useMemo(() => {
    if (!statsHistory) {
      return [];
    }

    const series: {
      [key: string]: { name: string; data: { x: number; y: number }[] };
    } = {};
    let hasValidNpu = false;

    statsHistory.forEach((stats, statsIdx) => {
      if (!stats) {
        return;
      }

      Object.entries(stats.npu_usages || {}).forEach(([key, stats]) => {
        if (!(key in series)) {
          series[key] = { name: key, data: [] };
        }

        if (stats?.temp !== undefined) {
          hasValidNpu = true;
          series[key].data.push({ x: statsIdx + 1, y: stats.temp });
        }
      });
    });

    if (!hasValidNpu) {
      return [];
    }

    return Object.keys(series).length > 0 ? Object.values(series) : undefined;
  }, [statsHistory]);

  // Number of cards the hardware grid renders. Which ones appear depends on
  // the vendor, so the column count follows the count rather than assuming a
  // fixed set is present.
  const hardwareCardCount = useMemo(() => {
    if (!statsHistory[0]?.gpu_usages) {
      return 0;
    }

    const hasNpu = statsHistory[0].npu_usages != undefined;

    return (
      1 + // gpu usage always renders alongside gpu_usages
      (gpuMemSeries ? 1 : 0) +
      (gpuEncSeries?.length ? 1 : 0) +
      (gpuComputeSeries?.length ? 1 : 0) +
      (gpuDecSeries?.length ? 1 : 0) +
      (gpuTempSeries?.length ? 1 : 0) +
      (hasNpu ? 1 : 0) +
      (hasNpu && npuTempSeries?.length ? 1 : 0)
    );
  }, [
    statsHistory,
    gpuMemSeries,
    gpuEncSeries,
    gpuComputeSeries,
    gpuDecSeries,
    gpuTempSeries,
    npuTempSeries,
  ]);

  // other processes stats

  const hardwareType = useMemo(() => {
    const hasGpu = gpuSeries.length > 0;
    const hasNpu = npuSeries.length > 0;

    if (hasGpu && !hasNpu) {
      return "GPUs";
    } else if (!hasGpu && hasNpu) {
      return "NPUs";
    } else {
      return "GPUs / NPUs";
    }
  }, [gpuSeries, npuSeries]);

  const otherProcessCpuSeries = useMemo(() => {
    if (!statsHistory) {
      return [];
    }

    const series: {
      [key: string]: { name: string; data: { x: number; y: string }[] };
    } = {};

    statsHistory.forEach((stats, statsIdx) => {
      if (!stats) {
        return;
      }

      Object.entries(stats.processes).forEach(([key, procStats]) => {
        if (!(key in series)) {
          series[key] = {
            name: t(`general.otherProcesses.series.${key}`),
            data: [],
          };
        }

        if (procStats.cpu != undefined) {
          series[key].data.push({
            x: statsIdx + 1,
            y: procStats.cpu,
          });
        }
      });
    });
    return Object.keys(series).length > 0 ? Object.values(series) : [];
  }, [statsHistory, t]);

  const otherProcessMemSeries = useMemo(() => {
    if (!statsHistory) {
      return [];
    }

    const series: {
      [key: string]: { name: string; data: { x: number; y: string }[] };
    } = {};

    statsHistory.forEach((stats, statsIdx) => {
      if (!stats) {
        return;
      }

      Object.entries(stats.processes).forEach(([key, procStats]) => {
        if (!(key in series)) {
          series[key] = {
            name: t(`general.otherProcesses.series.${key}`),
            data: [],
          };
        }

        if (procStats.mem) {
          series[key].data.push({
            x: statsIdx + 1,
            y: procStats.mem,
          });
        }
      });
    });
    return Object.values(series);
  }, [statsHistory, t]);

  const secondlook = statsHistory.at(-1)?.secondlook;

  return (
    <>
      <GPUInfoDialog
        showGpuInfo={showVainfo}
        gpuType={gpuType}
        setShowGpuInfo={setShowVainfo}
      />

      <div className="scrollbar-container mt-4 flex size-full flex-col overflow-y-auto">
        {secondlook?.enabled && (
          <section
            aria-label="Secondlook disagreements per hour"
            className="mb-4"
          >
            <div className="text-sm font-medium text-muted-foreground">
              Secondlook disagreements per hour, last 7 days
            </div>
            <div className="mt-4 rounded-lg bg-background_alt p-2.5 md:rounded-2xl">
              <div className="mb-2 text-xs text-muted-foreground">
                Objects where Secondlook&apos;s score flips the primary
                detector&apos;s decision. Orange and red: it would have, but the
                alert was not changed (answer too late). Rates = disagreements ÷
                objects analysed that hour.
              </div>
              <SecondlookDisagreementGraph />
            </div>
          </section>
        )}
        {secondlook?.enabled && (
          <section aria-label="Secondlook answers per hour" className="mb-4">
            <div className="text-sm font-medium text-muted-foreground">
              Secondlook answers per hour, last 7 days
            </div>
            <div className="mt-4 rounded-lg bg-background_alt p-2.5 md:rounded-2xl">
              <div className="mb-2 text-xs text-muted-foreground">
                What happened to every tracked object Secondlook was asked
                about, counted when it finished: agreed, or no usable answer (no
                hit, timed out, dropped, failed open). Rates = objects ÷ objects
                finished that hour.
              </div>
              <SecondlookAnswersGraph />
            </div>
          </section>
        )}
        <div className="text-sm font-medium text-muted-foreground">
          {t("general.detector.title")}
        </div>
        <div
          className={cn(
            "mt-4 grid w-full grid-cols-1 gap-2 sm:grid-cols-3",
            detTempSeries && "sm:grid-cols-4",
          )}
        >
          {statsHistory.length != 0 ? (
            <div className="rounded-lg bg-background_alt p-2.5 md:rounded-2xl">
              <div className="mb-5">{t("general.detector.inferenceSpeed")}</div>
              {detInferenceTimeSeries.map((series) => (
                <ThresholdBarGraph
                  key={series.name}
                  graphId={`${series.name}-inference`}
                  name={series.name}
                  unit="ms"
                  threshold={InferenceThreshold}
                  updateTimes={updateTimes}
                  data={[series]}
                  isActive={isActive}
                />
              ))}
            </div>
          ) : (
            <Skeleton className="aspect-video w-full rounded-lg md:rounded-2xl" />
          )}
          {statsHistory.length != 0 && (
            <>
              {detTempSeries && (
                <div className="rounded-lg bg-background_alt p-2.5 md:rounded-2xl">
                  <div className="mb-5">
                    {t("general.detector.temperature")}
                  </div>
                  {detTempSeries.map((series) => (
                    <ThresholdBarGraph
                      key={series.name}
                      graphId={`${series.name}-temp`}
                      name={series.name}
                      unit="°C"
                      threshold={DetectorTempThreshold}
                      updateTimes={updateTimes}
                      data={[series]}
                      isActive={isActive}
                    />
                  ))}
                </div>
              )}
            </>
          )}
          {statsHistory.length != 0 ? (
            <div className="rounded-lg bg-background_alt p-2.5 md:rounded-2xl">
              <div className="mb-5 flex flex-row items-center justify-between">
                {t("general.detector.cpuUsage")}
                <Popover>
                  <PopoverTrigger asChild>
                    <button
                      className="focus:outline-none"
                      aria-label={t("general.detector.cpuUsage")}
                    >
                      <CiCircleAlert
                        className="size-5"
                        aria-label={t("general.detector.cpuUsage")}
                      />
                    </button>
                  </PopoverTrigger>
                  <PopoverContent className="w-80">
                    <div className="space-y-2">
                      {t("general.detector.cpuUsageInformation")}
                    </div>
                  </PopoverContent>
                </Popover>
              </div>
              {detCpuSeries.map((series) => (
                <ThresholdBarGraph
                  key={series.name}
                  graphId={`${series.name}-cpu`}
                  unit="%"
                  name={series.name}
                  threshold={DetectorCpuThreshold}
                  updateTimes={updateTimes}
                  data={[series]}
                  isActive={isActive}
                />
              ))}
            </div>
          ) : (
            <Skeleton className="aspect-video w-full" />
          )}
          {statsHistory.length != 0 ? (
            <div className="rounded-lg bg-background_alt p-2.5 md:rounded-2xl">
              <div className="mb-5">{t("general.detector.memoryUsage")}</div>
              {detMemSeries.map((series) => (
                <ThresholdBarGraph
                  key={series.name}
                  graphId={`${series.name}-mem`}
                  unit="%"
                  name={series.name}
                  threshold={DetectorMemThreshold}
                  updateTimes={updateTimes}
                  data={[series]}
                  isActive={isActive}
                />
              ))}
            </div>
          ) : (
            <Skeleton className="aspect-video w-full" />
          )}
        </div>

        <section aria-label="Secondlook metrics">
          <div className="mt-4 flex items-center gap-2 text-sm font-medium text-muted-foreground">
            Secondlook
            <Popover>
              <PopoverTrigger asChild>
                <button
                  className="focus:outline-none"
                  aria-label="About Secondlook metrics"
                >
                  <CiCircleAlert className="size-5" />
                </button>
              </PopoverTrigger>
              <PopoverContent className="w-80 text-sm">
                Each bar is one stats interval. Processing: the latest worker
                time per analysis (preprocess, inference, association) completed
                in that interval; no bar when nothing completed. Green under 500
                ms, orange to 1 s, red from 1 s, where decisions time out
                (hold_seconds 1.0) and fall back to the primary. Queue: the most
                items at once in the interval, the one in process included, so 1
                means busy without queuing and 0 idle. Orange at{" "}
                {secondlook?.queue_max ?? 8} (queue max), red above it or when
                requests were dropped (see SL analyses dropped/s on Cameras).
                CPU and memory belong to the shared embeddings process below.
                Temperature is device-wide, not per model.
              </PopoverContent>
            </Popover>
          </div>
          {!secondlook?.enabled ? (
            <p className="mt-2 text-sm">
              Secondlook disabled or statistics unavailable.
            </p>
          ) : !secondlook.available ? (
            <p className="mt-2 text-sm">
              No real Secondlook model available.
              {secondlook.model_refusal && (
                <>
                  {" "}
                  Model refused at startup (Secondlook fails open, primary
                  unscaled): {secondlook.model_refusal}
                </>
              )}
            </p>
          ) : !secondlook.samples ? (
            <p className="mt-2 text-sm">Waiting for completed inference.</p>
          ) : (
            <div className="mt-4 grid w-full grid-cols-1 gap-2 sm:grid-cols-3">
              <div className="rounded-lg bg-background_alt p-2.5 md:rounded-2xl">
                <div className="mb-5">Secondlook processing time</div>
                <ThresholdBarGraph
                  graphId="secondlook-processing"
                  name={`secondlook (${secondlook.device ?? "?"})`}
                  unit="ms"
                  threshold={SecondlookProcessingThreshold}
                  updateTimes={updateTimes}
                  isActive={isActive}
                  data={[
                    {
                      name: "processing",
                      // a gap for intervals without a completed analysis
                      data: statsHistory.map((stat, index) => ({
                        x: index + 1,
                        // 2 decimals at most, like detector inference speed
                        y:
                          stat.secondlook?.interval?.completed &&
                          stat.secondlook.processing_ms != null
                            ? Math.round(stat.secondlook.processing_ms * 100) /
                              100
                            : null,
                      })),
                    },
                  ]}
                />
              </div>
              <div className="rounded-lg bg-background_alt p-2.5 md:rounded-2xl">
                <div className="mb-5">Secondlook queue (peak items)</div>
                <ThresholdBarGraph
                  graphId="secondlook-queue"
                  name={`secondlook (max ${secondlook.queue_max ?? "?"})`}
                  unit=""
                  threshold={{
                    warning: secondlook.queue_max ?? 8,
                    error: (secondlook.queue_max ?? 8) + 1,
                  }}
                  updateTimes={updateTimes}
                  isActive={isActive}
                  data={[
                    {
                      name: "peak items",
                      // red whenever requests were dropped in the interval
                      data: statsHistory.map((stat, index) => ({
                        x: index + 1,
                        y: stat.secondlook?.interval?.queue_peak ?? null,
                        ...(stat.secondlook?.interval?.dropped
                          ? { fillColor: "#FA5252" }
                          : {}),
                      })),
                    },
                  ]}
                />
              </div>
            </div>
          )}
        </section>

        {(statsHistory.length == 0 ||
          gpuSeries.length > 0 ||
          npuSeries.length > 0) && (
          <>
            <div className="mt-4 flex items-center justify-between">
              <div className="text-sm font-medium text-muted-foreground">
                {hardwareType}
              </div>
              {canGetGpuInfo && (
                <Button
                  className="cursor-pointer"
                  aria-label={t("general.hardwareInfo.title")}
                  size="sm"
                  onClick={() => setShowVainfo(true)}
                >
                  {t("general.hardwareInfo.title")}
                </Button>
              )}
            </div>
            <div
              className={cn(
                "mt-4 grid grid-cols-1 gap-2 sm:grid-cols-2",
                hardwareCardCount >= 3 && "lg:grid-cols-3",
                hardwareCardCount >= 4 && "xl:grid-cols-4",
                hardwareCardCount >= 5 && "3xl:grid-cols-5",
              )}
            >
              {statsHistory[0]?.gpu_usages && (
                <>
                  {statsHistory.length != 0 ? (
                    <div className="rounded-lg bg-background_alt p-2.5 md:rounded-2xl">
                      <div className="mb-5">
                        {t("general.hardwareInfo.gpuUsage")}
                      </div>
                      {gpuSeries.map((series) => (
                        <ThresholdBarGraph
                          key={series.name}
                          graphId={`${series.name}-gpu`}
                          name={series.name}
                          unit="%"
                          threshold={GPUUsageThreshold}
                          updateTimes={updateTimes}
                          data={[series]}
                          isActive={isActive}
                        />
                      ))}
                    </div>
                  ) : (
                    <Skeleton className="aspect-video w-full" />
                  )}
                  {statsHistory.length != 0 ? (
                    <>
                      {gpuMemSeries && (
                        <div className="rounded-lg bg-background_alt p-2.5 md:rounded-2xl">
                          <div className="mb-5">
                            {t("general.hardwareInfo.gpuMemory")}
                          </div>
                          {gpuMemSeries.map((series) => (
                            <ThresholdBarGraph
                              key={series.name}
                              graphId={`${series.name}-mem`}
                              unit="%"
                              name={series.name}
                              threshold={GPUMemThreshold}
                              updateTimes={updateTimes}
                              data={[series]}
                              isActive={isActive}
                            />
                          ))}
                        </div>
                      )}
                    </>
                  ) : (
                    <Skeleton className="aspect-video w-full" />
                  )}
                  {statsHistory.length != 0 ? (
                    <>
                      {gpuEncSeries && gpuEncSeries?.length != 0 && (
                        <div className="rounded-lg bg-background_alt p-2.5 md:rounded-2xl">
                          <div className="mb-5">
                            {t("general.hardwareInfo.gpuEncoder")}
                          </div>
                          {gpuEncSeries.map((series) => (
                            <ThresholdBarGraph
                              key={series.name}
                              graphId={`${series.name}-enc`}
                              unit="%"
                              name={series.name}
                              threshold={GPUMemThreshold}
                              updateTimes={updateTimes}
                              data={[series]}
                              isActive={isActive}
                            />
                          ))}
                        </div>
                      )}
                    </>
                  ) : (
                    <Skeleton className="aspect-video w-full" />
                  )}
                  {statsHistory.length != 0 ? (
                    <>
                      {gpuComputeSeries && gpuComputeSeries?.length != 0 && (
                        <div className="rounded-lg bg-background_alt p-2.5 md:rounded-2xl">
                          <div className="mb-5">
                            {t("general.hardwareInfo.gpuCompute")}
                          </div>
                          {gpuComputeSeries.map((series) => (
                            <ThresholdBarGraph
                              key={series.name}
                              graphId={`${series.name}-compute`}
                              unit="%"
                              name={series.name}
                              threshold={GPUMemThreshold}
                              updateTimes={updateTimes}
                              data={[series]}
                              isActive={isActive}
                            />
                          ))}
                        </div>
                      )}
                    </>
                  ) : (
                    <Skeleton className="aspect-video w-full" />
                  )}
                  {statsHistory.length != 0 ? (
                    <>
                      {gpuDecSeries && gpuDecSeries?.length != 0 && (
                        <div className="rounded-lg bg-background_alt p-2.5 md:rounded-2xl">
                          <div className="mb-5">
                            {t("general.hardwareInfo.gpuDecoder")}
                          </div>
                          {gpuDecSeries.map((series) => (
                            <ThresholdBarGraph
                              key={series.name}
                              graphId={`${series.name}-dec`}
                              unit="%"
                              name={series.name}
                              threshold={GPUMemThreshold}
                              updateTimes={updateTimes}
                              data={[series]}
                              isActive={isActive}
                            />
                          ))}
                        </div>
                      )}
                    </>
                  ) : (
                    <Skeleton className="aspect-video w-full" />
                  )}
                  {statsHistory.length != 0 ? (
                    <>
                      {gpuTempSeries && gpuTempSeries?.length != 0 && (
                        <div className="rounded-lg bg-background_alt p-2.5 md:rounded-2xl">
                          <div className="mb-5">
                            {t("general.hardwareInfo.gpuTemperature")}
                          </div>
                          {gpuTempSeries.map((series) => (
                            <ThresholdBarGraph
                              key={series.name}
                              graphId={`${series.name}-temp`}
                              name={series.name}
                              unit="°C"
                              threshold={DetectorTempThreshold}
                              updateTimes={updateTimes}
                              data={[series]}
                              isActive={isActive}
                            />
                          ))}
                        </div>
                      )}
                    </>
                  ) : (
                    <Skeleton className="aspect-video w-full" />
                  )}

                  {statsHistory[0]?.npu_usages && (
                    <>
                      {statsHistory.length != 0 ? (
                        <div className="rounded-lg bg-background_alt p-2.5 md:rounded-2xl">
                          <div className="mb-5">
                            {t("general.hardwareInfo.npuUsage")}
                          </div>
                          {npuSeries.map((series) => (
                            <ThresholdBarGraph
                              key={series.name}
                              graphId={`${series.name}-npu`}
                              name={series.name}
                              unit="%"
                              threshold={GPUUsageThreshold}
                              updateTimes={updateTimes}
                              data={[series]}
                              isActive={isActive}
                            />
                          ))}
                        </div>
                      ) : (
                        <Skeleton className="aspect-video w-full" />
                      )}
                      {statsHistory.length != 0 ? (
                        <>
                          {npuTempSeries && npuTempSeries?.length != 0 && (
                            <div className="rounded-lg bg-background_alt p-2.5 md:rounded-2xl">
                              <div className="mb-5">
                                {t("general.hardwareInfo.npuTemperature")}
                              </div>
                              {npuTempSeries.map((series) => (
                                <ThresholdBarGraph
                                  key={series.name}
                                  graphId={`${series.name}-temp`}
                                  name={series.name}
                                  unit="°C"
                                  threshold={DetectorTempThreshold}
                                  updateTimes={updateTimes}
                                  data={[series]}
                                  isActive={isActive}
                                />
                              ))}
                            </div>
                          )}
                        </>
                      ) : (
                        <Skeleton className="aspect-video w-full" />
                      )}
                    </>
                  )}
                </>
              )}
            </div>
          </>
        )}

        <div className="mt-4 text-sm font-medium text-muted-foreground">
          {t("general.otherProcesses.title")}
        </div>
        <div className="mt-4 grid grid-cols-1 gap-2 sm:grid-cols-2">
          {statsHistory.length != 0 ? (
            <div className="rounded-lg bg-background_alt p-2.5 md:rounded-2xl">
              <div className="mb-5">
                {t("general.otherProcesses.processCpuUsage")}
              </div>
              {otherProcessCpuSeries.map((series, index) => (
                <ThresholdBarGraph
                  key={`other-process-cpu-${index}`}
                  graphId={`other-process-cpu-${index}`}
                  unit="%"
                  threshold={DetectorCpuThreshold}
                  updateTimes={updateTimes}
                  data={[series]}
                  isActive={isActive}
                />
              ))}
            </div>
          ) : (
            <Skeleton className="aspect-tall w-full" />
          )}
          {statsHistory.length != 0 ? (
            <div className="rounded-lg bg-background_alt p-2.5 md:rounded-2xl">
              <div className="mb-5">
                {t("general.otherProcesses.processMemoryUsage")}
              </div>
              {otherProcessMemSeries.map((series, index) => (
                <ThresholdBarGraph
                  key={`other-process-mem-${index}`}
                  graphId={`other-process-mem-${index}`}
                  unit="%"
                  threshold={DetectorMemThreshold}
                  updateTimes={updateTimes}
                  data={[series]}
                  isActive={isActive}
                />
              ))}
            </div>
          ) : (
            <Skeleton className="aspect-tall w-full" />
          )}
        </div>
      </div>
    </>
  );
}
