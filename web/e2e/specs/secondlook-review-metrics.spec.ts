import type { Page } from "@playwright/test";
import { test, expect } from "../fixtures/frigate-test";
import { PLACEHOLDER_PNG } from "../helpers/api-mocker";
import { BASE_STATS } from "../fixtures/mock-data/stats";

// Mocked HTTP/WS fixtures (no Frigate backend): these verify the UI contract
// against the API shapes covered by the backend unit tests.

const evidence = (box: number[]) => ({
  version: 1,
  width: 1280,
  height: 720,
  frame_time: 1,
  primary_box: box,
  sl_box: [box[0] + 5, box[1] + 5, box[2] + 5, box[3] + 5],
  crop_region: [0, 0, 640, 640],
  secondary_label: "truck",
  secondary_score: 0.9,
});

test("small-edge toggle defaults on, drives list and tab count, and stays a filter @mobile", async ({
  frigateApp,
}) => {
  await frigateApp.installDefaults({
    config: {
      secondlook: { enabled: true, save_and_review: { enabled: true } },
    },
  });
  const listQueries: URLSearchParams[] = [];
  const rows = [
    { id: "edge-small", hidden: true },
    { id: "centre", hidden: false },
  ].map(({ id, hidden }, i) => ({
    id,
    track_id: id,
    camera: "front_door",
    kind: "misclass",
    label: "car",
    misclass_label: "truck",
    primary_score: 0.8,
    sl_score: 0.9,
    factor: 0.5,
    box: null,
    mark: null,
    frame_time: 1000 * (i + 1),
    created_at: 1000 * (i + 1),
    evidence: evidence(hidden ? [0, 300, 40, 340] : [600, 300, 700, 400]),
    hidden,
  }));
  await frigateApp.page.route(
    "**/api/secondlook/save_and_review**",
    (route) => {
      const url = new URL(route.request().url());
      if (url.pathname.endsWith("/still.jpg"))
        return route.fulfill({
          contentType: "image/png",
          body: PLACEHOLDER_PNG,
        });
      listQueries.push(url.searchParams);
      const hide = url.searchParams.get("hide_edge_small") === "true";
      const visible = rows.filter((r) => !hide || !r.hidden);
      return route.fulfill({
        json: {
          examples: visible,
          total: visible.length,
          unreviewed: visible.length,
          has_more: false,
          hidden_edge_small: rows.length - visible.length,
        },
      });
    },
  );
  await frigateApp.goto("/review");
  await frigateApp.page
    .getByRole("radio", { name: "Corrections", exact: true })
    .click();
  const toggle = frigateApp.page.getByRole("switch", {
    name: "Hide small boxes touching the frame edge",
  });
  await expect(toggle).toHaveAttribute("data-state", "checked");
  await expect(frigateApp.page.getByTestId("edge-hidden-count")).toHaveText(
    "1 hidden (still saved)",
  );
  await expect(frigateApp.page.getByTestId("correction-card")).toHaveCount(1);
  expect(listQueries.every((q) => q.get("hide_edge_small") === "true")).toBe(
    true,
  );
  listQueries.length = 0;
  await toggle.click();
  await expect(frigateApp.page.getByTestId("correction-card")).toHaveCount(2);
  await expect(frigateApp.page.getByTestId("edge-hidden-count")).toHaveText(
    "showing all",
  );
  await expect(
    frigateApp.page.getByText("2 unreviewed · 2 matching"),
  ).toBeVisible();
  // badge (limit=1) and list both dropped the filter
  expect(listQueries.some((q) => q.get("limit") === "1")).toBe(true);
  expect(listQueries.every((q) => !q.has("hide_edge_small"))).toBe(true);
});

test("configured default off keeps every record visible", async ({
  frigateApp,
}) => {
  await frigateApp.installDefaults({
    config: {
      secondlook: {
        enabled: true,
        save_and_review: {
          enabled: true,
          review: {
            hide_edge_small: false,
            group_gap_seconds: 60,
            edge_margin_px: 4,
            small_dim_fraction: 0.05,
          },
        },
      },
    },
  });
  const queries: URLSearchParams[] = [];
  await frigateApp.page.route(
    "**/api/secondlook/save_and_review**",
    (route) => {
      queries.push(new URL(route.request().url()).searchParams);
      return route.fulfill({
        json: { examples: [], total: 0, unreviewed: 0, has_more: false },
      });
    },
  );
  await frigateApp.goto("/review");
  await frigateApp.page
    .getByRole("radio", { name: "Corrections", exact: true })
    .click();
  await expect(
    frigateApp.page.getByRole("switch", {
      name: "Hide small boxes touching the frame edge",
    }),
  ).toHaveAttribute("data-state", "unchecked");
  expect(queries.map((q) => q.get("limit"))).toContain("1");
  expect(queries.every((q) => !q.has("hide_edge_small"))).toBe(true);
});

test("verdict buttons use their box colours, work for misclass, and show migrated marks @mobile", async ({
  frigateApp,
}) => {
  await frigateApp.installDefaults({
    config: {
      secondlook: { enabled: true, save_and_review: { enabled: true } },
    },
  });
  const posted: unknown[] = [];
  let mark: string | null = "secondlook_right";
  await frigateApp.page.route(
    "**/api/secondlook/save_and_review**",
    (route) => {
      const url = new URL(route.request().url());
      if (url.pathname.endsWith("/still.jpg"))
        return route.fulfill({
          contentType: "image/png",
          body: PLACEHOLDER_PNG,
        });
      if (route.request().method() === "POST") {
        posted.push(route.request().postDataJSON());
        mark = route.request().postDataJSON().mark;
        return route.fulfill({ json: { success: true } });
      }
      return route.fulfill({
        json: {
          examples: [
            {
              id: "migrated",
              track_id: "t",
              camera: "front_door",
              kind: "misclass",
              label: "car",
              misclass_label: "truck",
              primary_score: 0.8,
              sl_score: 0.9,
              factor: 0.5,
              box: null,
              mark,
              legacy_mark: "accept",
              frame_time: 1,
              created_at: 1,
              evidence: evidence([600, 300, 700, 400]),
            },
          ],
          total: 1,
          unreviewed: 0,
          has_more: false,
        },
      });
    },
  );
  await frigateApp.goto("/review");
  await frigateApp.page
    .getByRole("radio", { name: "Corrections", exact: true })
    .click();
  const card = frigateApp.page.getByTestId("correction-card");
  await expect(card).toHaveCount(1);
  const rects = card
    .getByRole("img", { name: "Primary and Secondlook boxes" })
    .locator("rect");
  await expect(rects.nth(0)).toHaveAttribute("stroke", "#00ffff");
  await expect(rects.nth(1)).toHaveAttribute("stroke", "#ff9900");
  const primary = card.getByRole("button", {
    name: "Primary right",
    exact: true,
  });
  const secondlook = card.getByRole("button", {
    name: "Secondlook right",
    exact: true,
  });
  const ignore = card.getByRole("button", { name: "Ignore", exact: true });
  await expect(primary).toHaveCSS("border-color", "rgb(0, 255, 255)");
  await expect(secondlook).toHaveCSS("border-color", "rgb(255, 153, 0)");
  await expect(secondlook).toHaveAttribute("aria-pressed", "true");
  await expect(secondlook).toHaveCSS("background-color", "rgb(255, 153, 0)");
  await expect(primary.getByText("car", { exact: true })).toBeVisible();
  await expect(secondlook.getByText("truck", { exact: true })).toBeVisible();
  await expect(ignore.getByText("exclude from training")).toBeVisible();
  await expect(
    card.getByText("Migrated from earlier “accept” mark"),
  ).toBeVisible();
  for (const button of [primary, secondlook, ignore]) {
    const box = await button.boundingBox();
    expect(box?.height ?? 0).toBeGreaterThanOrEqual(36);
  }
  await primary.click();
  await expect(primary).toHaveAttribute("aria-pressed", "true");
  await expect(primary).toHaveCSS("background-color", "rgb(0, 255, 255)");
  await primary.click(); // edit later: clicking the chosen verdict clears it
  expect(posted).toEqual([{ mark: "primary_right" }, { mark: null }]);
});

test("Review cards show whether the live pipeline acted, alarms explain why not @mobile", async ({
  frigateApp,
}, testInfo) => {
  await frigateApp.installDefaults({
    config: {
      secondlook: { enabled: true, save_and_review: { enabled: true } },
    },
  });
  const item = (
    id: string,
    kind: string,
    live_outcome: string,
    minutes: number,
  ) => ({
    id,
    track_id: id,
    camera: "front_door",
    kind,
    label: "person",
    misclass_label: kind === "misclass" ? "dog" : null,
    primary_score: 0.6,
    sl_score: 0.8,
    factor: kind === "false_positive" ? 0.2 : 1.6,
    box: null,
    mark: null,
    frame_time: 1_700_000_000 + minutes * 600,
    created_at: 1_700_000_000 + minutes * 600,
    evidence: evidence([600, 300, 700, 400]),
    live_outcome,
  });
  await frigateApp.page.route("**/api/secondlook/save_and_review**", (route) =>
    new URL(route.request().url()).pathname.endsWith("/still.jpg")
      ? route.fulfill({ contentType: "image/png", body: PLACEHOLDER_PNG })
      : route.fulfill({
          json: {
            examples: [
              item("rescued", "false_negative", "rescued", 5),
              item("not-rescued", "false_negative", "not_rescued", 4),
              item("suppressed", "false_positive", "suppressed", 3),
              item("not-suppressed", "false_positive", "not_suppressed", 2),
              item("misclass", "misclass", "none", 1),
            ],
            total: 5,
            unreviewed: 5,
            has_more: false,
          },
        }),
  );
  await frigateApp.goto("/review");
  await frigateApp.page
    .getByRole("radio", { name: "Corrections", exact: true })
    .click();
  const page = frigateApp.page;
  const badges = page.getByTestId("live-outcome");
  await expect(badges).toHaveCount(4, { timeout: 15_000 });
  await expect(badges).toHaveText([
    "Rescued: alert raised",
    "Not rescued",
    "Suppressed: alert prevented",
    "Not suppressed",
  ]);
  const background = (outcome: string) =>
    page
      .locator(`[data-outcome="${outcome}"]`)
      .evaluate((el) => getComputedStyle(el).backgroundColor);
  for (const alarm of ["not_rescued", "not_suppressed"])
    expect(await background(alarm)).toBe("rgb(220, 38, 38)");
  for (const calm of ["rescued", "suppressed"])
    expect(await background(calm)).not.toBe("rgb(220, 38, 38)");
  // misclass (and old or in-progress items) carry no badge
  await expect(
    page
      .getByTestId("correction-card")
      .filter({ hasText: "dog" })
      .getByTestId("live-outcome"),
  ).toHaveCount(0);
  await page.locator('[data-outcome="not_rescued"]').hover();
  await expect(page.getByRole("tooltip")).toContainText(/too late/);
  await expect(page.getByRole("tooltip")).toContainText(
    /would have reached the threshold/,
  );
  await page.mouse.move(0, 0);
  await page.locator('[data-outcome="not_suppressed"]').focus();
  await expect(
    page.getByRole("tooltip").filter({ hasText: /released anyway/ }),
  ).toContainText(/would have stayed below the threshold/);
  await page.locator('[data-outcome="not_rescued"]').scrollIntoViewIfNeeded();
  await page.screenshot({
    path: `${SCREENS}/review-badges-${testInfo.project.name}.png`,
  });
});

test("a grouped card still flags members the live pipeline did not act on", async ({
  frigateApp,
}) => {
  await frigateApp.installDefaults({
    config: {
      secondlook: { enabled: true, save_and_review: { enabled: true } },
    },
  });
  const member = (id: string, live_outcome: string, seconds: number) => ({
    id,
    track_id: id,
    camera: "front_door",
    kind: "false_negative",
    label: "person",
    misclass_label: null,
    primary_score: 0.5,
    sl_score: 0.7,
    factor: 1.4,
    box: null,
    mark: null,
    frame_time: 1_700_000_000 + seconds,
    created_at: 1_700_000_000 + seconds,
    evidence: evidence([600, 300, 700, 400]),
    live_outcome,
  });
  await frigateApp.page.route("**/api/secondlook/save_and_review**", (route) =>
    new URL(route.request().url()).pathname.endsWith("/still.jpg")
      ? route.fulfill({ contentType: "image/png", body: PLACEHOLDER_PNG })
      : route.fulfill({
          json: {
            examples: [
              member("newest", "rescued", 30),
              member("older", "not_rescued", 0),
            ],
            total: 2,
            unreviewed: 2,
            has_more: false,
          },
        }),
  );
  await frigateApp.goto("/review");
  await frigateApp.page
    .getByRole("radio", { name: "Corrections", exact: true })
    .click();
  const group = frigateApp.page.getByTestId("correction-group");
  await expect(group).toHaveCount(1, { timeout: 15_000 });
  await expect(group.getByTestId("group-not-acted")).toHaveText(
    "1 of 2 not acted on live (alert unchanged)",
  );
  await expect(group.getByTestId("live-outcome")).toHaveText([
    "Rescued: alert raised",
  ]);
});

const slStats = (
  refusal: string | null = null,
  accuracy = { false_positive: 3, false_negative: 2, misclass: 1 },
  interval = {
    processing_ms: 25,
    interval: { completed: 1, dropped: 0, queue_peak: 1 },
  },
) => ({
  ...BASE_STATS,
  cameras: {
    ...BASE_STATS.cameras,
    front_door: {
      ...BASE_STATS.cameras.front_door,
      secondlook_analysed_ps: 3.5,
      secondlook_skipped_ps: 1.25,
    },
    garage: {
      ...BASE_STATS.cameras.garage,
      secondlook_analysed_ps: 1.5,
      secondlook_skipped_ps: 0.75,
    },
  },
  secondlook: {
    enabled: true,
    available: refusal === null,
    samples: refusal === null ? 4 : 0,
    last_updated: Date.now() / 1000,
    inference_ms: refusal === null ? 20 : null,
    processing_ms: refusal === null ? 25 : null,
    queue_ms: refusal === null ? 4 : null,
    queue_items: refusal === null ? 1 : null,
    queue_max: 8,
    device: "CPU",
    model_refusal: refusal,
    accuracy,
    ...(refusal === null ? interval : { interval: null }),
    storage: {
      database_bytes: 3 * 1048576,
      images_bytes: 512 * 1048576,
      image_files: 1234,
      retention: null,
    },
  },
});

const SCREENS = process.env.SECONDLOOK_SCREENS ?? "test-results/secondlook-screens";

// 168 server-local hours ending 2026-11-11 12:00 (+01:00); the last three are
// nothing analysed (a rate gap), 97 objects and 4 objects
const hourly = () => {
  const end = Date.UTC(2026, 10, 11, 11) / 1000;
  const hours = Array.from({ length: 168 }, (_, i) => {
    const start = end - (167 - i) * 3600;
    const local = new Date((start + 3600) * 1000).toISOString();
    return {
      hour_start: start,
      local_hour: `${local.slice(0, 13)}:00+01:00`,
      objects_analysed: 10,
      false_positive: 0,
      false_negative: 0,
      misclass: 0,
      fn_rescued: 0,
      fn_not_rescued: 0,
      fp_suppressed: 0,
      fp_not_suppressed: 0,
      tracks_finished: 10,
      agreements: 10,
      results: 10,
      no_hit: 0,
      timeouts: 0,
      dropped: 0,
      fail_open: 0,
    };
  });
  hours[165] = {
    ...hours[165],
    objects_analysed: 0,
    tracks_finished: 0,
    agreements: 0,
    results: 0,
  };
  hours[166] = {
    ...hours[166],
    objects_analysed: 97,
    false_positive: 1,
    fp_suppressed: 1,
    false_negative: 1,
    fn_not_rescued: 1,
    misclass: 2,
    tracks_finished: 97,
    agreements: 92,
    results: 94,
    no_hit: 3,
    timeouts: 0,
    dropped: 0,
    fail_open: 0,
  };
  hours[166] = { ...hours[166], timeouts: 1, dropped: 1, results: 92 };
  hours[167] = {
    ...hours[167],
    tracks_finished: 4,
    agreements: 2,
    results: 4,
    objects_analysed: 4,
    false_positive: 1,
    fp_not_suppressed: 1,
    false_negative: 1,
    fn_rescued: 1,
  };
  return { timezone: "Europe/Berlin", hours };
};

type ChartConfig = {
  stacked: boolean;
  series: { name: string; data: (number | null)[] }[];
};
declare global {
  interface Window {
    ApexCharts: {
      getChartByID: (id: string) =>
        | {
            w: {
              config: {
                chart: { stacked: boolean };
                series: ChartConfig["series"];
              };
              globals: { colors: string[] };
            };
          }
        | undefined;
    };
  }
}
const chartConfig = (page: Page, id: string) =>
  page.evaluate((chartId) => {
    const w = window.ApexCharts.getChartByID(chartId)?.w;
    return w
      ? { stacked: w.config.chart.stacked, series: w.config.series }
      : null;
  }, id);

// screenshots for review: wait until ApexCharts entry animations have settled
async function settledScreenshot(page: Page, path: string) {
  const shapes = () =>
    page
      .locator(".apexcharts-series path")
      .evaluateAll((paths) => paths.map((p) => p.getAttribute("d")).join("|"));
  await expect(async () => {
    const before = await shapes();
    await new Promise((resolve) => setTimeout(resolve, 400));
    expect(await shapes()).toBe(before);
  }).toPass();
  await page.screenshot({ path });
}

test("Frames / Detections carry SL completed and dropped lines only for SL cameras @mobile", async ({
  frigateApp,
}, testInfo) => {
  await frigateApp.installDefaults();
  frigateApp.ws.statsOverride = slStats();
  await frigateApp.page.route("**/api/stats/history**", (route) =>
    route.fulfill({ json: [slStats(), slStats()] }),
  );
  await frigateApp.goto("/system#cameras");
  const legend = (label: string) =>
    frigateApp.page.locator("div.items-center.gap-1", { hasText: label });
  // overall + front_door + garage; backyard has no Secondlook rates
  await expect(legend("SL analyses completed/s")).toHaveCount(3, {
    timeout: 15_000,
  });
  await expect(legend("SL analyses dropped/s")).toHaveCount(3);
  expect(
    (await legend("SL analyses completed/s").allTextContents()).sort(),
  ).toEqual([
    "SL analyses completed/s1.5",
    "SL analyses completed/s3.5",
    "SL analyses completed/s5",
  ]);
  expect(
    (await legend("SL analyses dropped/s").allTextContents()).sort(),
  ).toEqual([
    "SL analyses dropped/s0.75",
    "SL analyses dropped/s1.25",
    "SL analyses dropped/s2",
  ]);
  // the separate Secondlook analyses chart was replaced, not duplicated
  await expect(frigateApp.page.getByText("Secondlook analyses")).toHaveCount(0);
  await expect(
    frigateApp.page.locator("div.items-center.gap-1", {
      hasText: /^camera[\d.]+$/,
    }),
  ).toHaveCount(4);
  const viewport = frigateApp.page.viewportSize()!;
  await frigateApp.page.setViewportSize({
    width: viewport.width,
    height: 2600,
  });
  await expect(legend("SL analyses completed/s").last()).toBeInViewport();
  await settledScreenshot(
    frigateApp.page,
    `${SCREENS}/cameras-${testInfo.project.name}.png`,
  );
});

test("General: disagreements on top, Secondlook below Detectors in detector style @mobile", async ({
  frigateApp,
}, testInfo) => {
  await frigateApp.installDefaults();
  const acc = { false_positive: 3, false_negative: 2, misclass: 1 };
  const at = (
    processing_ms: number,
    completed: number,
    peak: number,
    dropped = 0,
  ) =>
    slStats(null, acc, {
      processing_ms,
      interval: { completed, dropped, queue_peak: peak },
    });
  // busy, idle (stale processing_ms), slow + queue at max, too slow + drops
  const history = [
    slStats(null, { false_positive: 1, false_negative: 0, misclass: 0 }),
    slStats(null, { false_positive: 2, false_negative: 1, misclass: 1 }),
    at(25, 2, 1),
    at(25, 0, 0),
    at(700, 3, 8),
    at(1200.4567, 4, 8, 2),
  ];
  frigateApp.ws.statsOverride = history.at(-1)!;
  await frigateApp.page.route("**/api/stats/history**", (route) =>
    route.fulfill({ json: history }),
  );
  await frigateApp.page.route(
    "**/api/secondlook/disagreements/hourly",
    (route) => route.fulfill({ json: hourly() }),
  );
  await frigateApp.goto("/system#general");
  const accuracy = frigateApp.page.getByRole("region", {
    name: "Secondlook disagreements per hour",
  });
  await expect(accuracy.locator(".apexcharts-canvas")).toHaveCount(1, {
    timeout: 15_000,
  });
  await expect(frigateApp.page.getByText(/accuracy/i)).toHaveCount(0);
  const section = frigateApp.page.getByRole("region", {
    name: "Secondlook metrics",
  });
  await expect(section.locator(".apexcharts-canvas")).toHaveCount(2);
  await expect(section.getByText("secondlook (max 8)")).toBeVisible();
  // last four intervals: busy, idle, slow/at max, too slow/dropped
  const bars = (graph: string) =>
    section
      .locator(`#apexcharts${graph} .apexcharts-series path`)
      .evaluateAll((paths) =>
        paths
          .slice(-4)
          .map((p) => [p.getAttribute("fill"), p.getAttribute("val")]),
      );
  const [green, orange, red] = [
    "rgba(33,121,48,0.85)",
    "rgba(255,153,102,0.85)",
    "rgba(250,82,82,0.85)",
  ];
  await expect
    .poll(() => bars("secondlook-processing"))
    .toEqual([
      [green, "25"],
      ["none", null], // gap: nothing completed although processing_ms is stale
      [orange, "700"],
      [red, "1200.46"], // 2 decimals, like detector inference speed
    ]);
  await expect
    .poll(() => bars("secondlook-queue"))
    .toEqual([
      [green, "1"], // busy without queuing
      [green, "0"], // idle is 0, not a gap
      [orange, "8"],
      [red, "8"], // red from drops in the interval, not from the count
    ]);
  await expect(section.getByText("secondlook (CPU)1200.46ms")).toBeVisible();
  await expect(section.getByText(/queue_ms|queue time/i)).toHaveCount(0);
  const detectors = frigateApp.page.getByText("Detectors", { exact: true });
  const y = async (l: typeof section) => (await l.boundingBox())!.y;
  expect(await y(accuracy)).toBeLessThan(await y(detectors));
  expect(await y(detectors)).toBeLessThan(await y(section));
  // same card/chart shape as the detector inference chart
  const card = (l: typeof section) => l.locator(".apexcharts-canvas").first();
  const detectorChart = frigateApp.page
    .locator("div.rounded-lg", { hasText: "Detector Inference Speed" })
    .locator(".apexcharts-canvas")
    .first();
  expect((await card(section).boundingBox())!.height).toBeCloseTo(
    (await detectorChart.boundingBox())!.height,
    0,
  );
  const viewport = frigateApp.page.viewportSize()!;
  await frigateApp.page.setViewportSize({
    width: viewport.width,
    height: 2000,
  });
  await expect(section).toBeInViewport();
  await settledScreenshot(
    frigateApp.page,
    `${SCREENS}/general-${testInfo.project.name}.png`,
  );
});

test("General disagreements: stacked counts, rates toggle with gaps, tooltip @mobile", async ({
  frigateApp,
}, testInfo) => {
  await frigateApp.installDefaults();
  frigateApp.ws.statsOverride = slStats();
  await frigateApp.page.route("**/api/stats/history**", (route) =>
    route.fulfill({ json: [slStats(), slStats()] }),
  );
  await frigateApp.page.route(
    "**/api/secondlook/disagreements/hourly",
    (route) => route.fulfill({ json: hourly() }),
  );
  await frigateApp.goto("/system#general");
  const page = frigateApp.page;
  const region = page.getByRole("region", {
    name: "Secondlook disagreements per hour",
  });
  await expect(
    region.getByText("Secondlook disagreements per hour, last 7 days"),
  ).toBeVisible({ timeout: 15_000 });
  await expect(region.getByText(/primary detector/)).toBeVisible();
  const counts = region.getByRole("button", { name: "Counts", exact: true });
  const rates = region.getByRole("button", { name: "Rates", exact: true });
  await expect(counts).toHaveAttribute("aria-pressed", "true");
  const legend = (label: string) =>
    region.locator("div.items-center.gap-1", {
      hasText: new RegExp(`^${label}`),
    });
  const names = [
    "FP suppressed",
    "FP not suppressed",
    "FN rescued",
    "FN not rescued",
    "misclassifications",
  ];

  // default: stacked counts, 168 hourly points per series, integer totals,
  // in stack order with alarm colours on the two "not" parts
  await expect
    .poll(() => chartConfig(page, "secondlook-disagreements"))
    .toMatchObject({ stacked: true });
  const countsConfig = (await chartConfig(page, "secondlook-disagreements"))!;
  expect(countsConfig.series.map((s) => s.name)).toEqual(names);
  expect(countsConfig.series.map((s) => s.data.length)).toEqual([
    168, 168, 168, 168, 168,
  ]);
  expect(countsConfig.series.map((s) => s.data[166])).toEqual([1, 0, 0, 1, 2]);
  expect(countsConfig.series.map((s) => s.data[167])).toEqual([0, 1, 1, 0, 0]);
  expect(countsConfig.series.map((s) => s.data[165])).toEqual([0, 0, 0, 0, 0]);
  expect(
    await page.evaluate(
      () =>
        window.ApexCharts.getChartByID("secondlook-disagreements")!.w.globals
          .colors,
    ),
  ).toEqual(["#5C7CFA", "#F97316", "#74C0FC", "#DC2626", "#FAD75C"]);
  // the alarm parts are drawn in alarm colours
  const fills = (series: number) =>
    region
      .locator(`.apexcharts-series[rel="${series + 1}"] path[val="1"]`)
      .first()
      .getAttribute("fill");
  expect(await fills(1)).toMatch(/^#F97316$|rgba\(249,\s*115,\s*22/i);
  expect(await fills(3)).toMatch(/^#DC2626$|rgba\(220,\s*38,\s*38/i);
  for (const [name, total] of [
    ["FP suppressed", "1"],
    ["FP not suppressed", "1"],
    ["FN rescued", "1"],
    ["FN not rescued", "1"],
    ["misclassifications", "2"],
  ])
    await expect(legend(name)).toHaveText(name + total);

  const tooltip = region.locator(".apexcharts-tooltip");
  // hour 166 (misclass on top, non-zero in both views) anchors the column
  // pitch (bars use 90% of it); other hours are hovered by offset from it
  const hoverHour = async (index: number) => {
    const bar = (await region
      .locator('.apexcharts-series path[j="166"]:not([val="0"])')
      .last()
      .boundingBox())!;
    const pitch = bar.width / 0.9;
    await page.mouse.move(
      bar.x + bar.width / 2 + (index - 166) * pitch,
      bar.y + bar.height / 2,
    );
  };
  const values = tooltip.locator(".apexcharts-tooltip-text-y-value");
  const labels = tooltip.locator(".apexcharts-tooltip-text-y-label");

  await hoverHour(166);
  await expect(tooltip).toContainText("97 objects analysed");
  await expect(tooltip).toContainText("2026-11-11 11:00");
  await expect(labels).toHaveText(names.map((name) => `${name}:`));
  await expect(values).toHaveText(["1", "0", "0", "1", "2"]);
  await settledScreenshot(
    page,
    `${SCREENS}/general-counts-${testInfo.project.name}.png`,
  );

  // rates: disagreements / objects analysed that hour, gaps where 0 analysed
  await rates.click();
  await expect(rates).toHaveAttribute("aria-pressed", "true");
  await expect
    .poll(async () =>
      (await chartConfig(page, "secondlook-disagreements"))?.series.map(
        (s) => s.data[166],
      ),
    )
    .toEqual([1.03, 0, 0, 1.03, 2.06]);
  const ratesConfig = (await chartConfig(page, "secondlook-disagreements"))!;
  expect(ratesConfig.series.map((s) => s.data[165])).toEqual([
    null,
    null,
    null,
    null,
    null,
  ]);
  expect(ratesConfig.series.map((s) => s.data[167])).toEqual([0, 25, 25, 0, 0]);
  // week rates: 1 or 2 of 1751 objects
  await expect(legend("FP suppressed")).toHaveText("FP suppressed0.06%");
  await expect(legend("misclassifications")).toHaveText(
    "misclassifications0.11%",
  );
  await hoverHour(166);
  await expect(tooltip).toContainText("97 objects analysed");
  // each point carries its own numerator and denominator
  await expect(values).toHaveText([
    "1.03% (1 of 97)",
    "0.00% (0 of 97)",
    "0.00% (0 of 97)",
    "1.03% (1 of 97)",
    "2.06% (2 of 97)",
  ]);
  for (const text of await values.allTextContents())
    expect(text).toMatch(/^\d+\.\d{2}% \(\d+ of \d+\)$/);
  await settledScreenshot(
    page,
    `${SCREENS}/rates-tooltip-${testInfo.project.name}.png`,
  );
  await hoverHour(167);
  await expect(values).toHaveText([
    "0.00% (0 of 4)",
    "25.00% (1 of 4)",
    "25.00% (1 of 4)",
    "0.00% (0 of 4)",
    "0.00% (0 of 4)",
  ]);
  await hoverHour(165);
  await expect(tooltip).toContainText("0 objects analysed");
  await expect(values).toHaveText([
    "– (0 of 0)",
    "– (0 of 0)",
    "– (0 of 0)",
    "– (0 of 0)",
    "– (0 of 0)",
  ]);
  for (const label of await region
    .locator(".apexcharts-yaxis-label tspan")
    .allTextContents())
    expect(label).toMatch(/^\d+\.\d{2}%$/);
  await page.mouse.move(0, 0);
  await settledScreenshot(
    page,
    `${SCREENS}/general-rates-${testInfo.project.name}.png`,
  );
  // info box: numerator and denominator for every line, in plain words
  await region.getByRole("button", { name: "About the rates" }).click();
  const info = page.getByTestId("rates-info");
  await expect(info).toContainText("Rates: numerator ÷ denominator");
  await expect(info).toContainText(
    "Denominator, the same for every line: tracked objects Secondlook analysed in that hour",
  );
  for (const name of names) await expect(info).toContainText(`${name}:`);
  await expect(info).toContainText("1.03% (1 of 97)");
  // let the open animation finish before the screenshot
  await expect(info).toHaveCSS("opacity", "1");
  await page.screenshot({
    path: `${SCREENS}/rates-info-${testInfo.project.name}.png`,
  });
  await page.keyboard.press("Escape");
  await counts.click();
  await expect(legend("misclassifications")).toHaveText("misclassifications2");
});

test("General answers: agreed and no-answer rates with numerator and denominator @mobile", async ({
  frigateApp,
}, testInfo) => {
  await frigateApp.installDefaults();
  frigateApp.ws.statsOverride = slStats();
  await frigateApp.page.route("**/api/stats/history**", (route) =>
    route.fulfill({ json: [slStats(), slStats()] }),
  );
  await frigateApp.page.route(
    "**/api/secondlook/disagreements/hourly",
    (route) => route.fulfill({ json: hourly() }),
  );
  await frigateApp.goto("/system#general");
  const page = frigateApp.page;
  const region = page.getByRole("region", {
    name: "Secondlook answers per hour",
  });
  await expect(
    region.getByText("Secondlook answers per hour, last 7 days"),
  ).toBeVisible({ timeout: 15_000 });
  // below the disagreements chart, above the detectors
  const disagreements = page.getByRole("region", {
    name: "Secondlook disagreements per hour",
  });
  const y = async (l: typeof region) => (await l.boundingBox())!.y;
  expect(await y(disagreements)).toBeLessThan(await y(region));
  const names = ["agreed", "no hit", "timed out", "dropped", "failed open"];
  const legend = (label: string) =>
    region.locator("div.items-center.gap-1", {
      hasText: new RegExp(`^${label}`),
    });
  const rates = region.getByRole("button", { name: "Rates", exact: true });
  const counts = region.getByRole("button", { name: "Counts", exact: true });
  await expect(rates).toHaveAttribute("aria-pressed", "true");
  await expect
    .poll(async () =>
      (await chartConfig(page, "secondlook-answers"))?.series.map(
        (s) => s.data[166],
      ),
    )
    .toEqual([94.85, 3.09, 1.03, 1.03, 0]);
  const ratesConfig = (await chartConfig(page, "secondlook-answers"))!;
  expect(ratesConfig.series.map((s) => s.name)).toEqual(names);
  expect(ratesConfig.series.map((s) => s.data.length)).toEqual([
    168, 168, 168, 168, 168,
  ]);
  expect(ratesConfig.series.map((s) => s.data[165])).toEqual([
    null,
    null,
    null,
    null,
    null,
  ]);
  expect(ratesConfig.series.map((s) => s.data[167])).toEqual([50, 0, 0, 0, 0]);
  // week: 1744 agreed of 1751 finished
  await expect(legend("agreed")).toHaveText("agreed99.60%");
  await expect(legend("no hit")).toHaveText("no hit0.17%");
  await expect(legend("timed out")).toHaveText("timed out0.06%");
  await expect(region.getByText("of 1751 finished")).toBeVisible();

  // tooltip: each point with its numerator and denominator
  // the plot area moves with the axis label widths: measure on each hover
  const hoverHour = async (index: number) => {
    const plot = (await region.locator(".apexcharts-grid").boundingBox())!;
    await page.mouse.move(
      plot.x + (plot.width * index) / 167,
      plot.y + plot.height / 2,
    );
  };
  const tooltip = region.locator(".apexcharts-tooltip");
  const values = tooltip.locator(".apexcharts-tooltip-text-y-value");
  await hoverHour(166);
  await expect(tooltip).toContainText("97 tracked objects finished");
  await expect(values).toHaveText([
    "94.85% (92 of 97)",
    "3.09% (3 of 97)",
    "1.03% (1 of 97)",
    "1.03% (1 of 97)",
    "0.00% (0 of 97)",
  ]);
  for (const text of await values.allTextContents())
    expect(text).toMatch(/^\d+\.\d{2}% \(\d+ of \d+\)$/);
  await settledScreenshot(
    page,
    `${SCREENS}/answers-tooltip-${testInfo.project.name}.png`,
  );
  await hoverHour(165);
  await expect(values).toHaveText([
    "– (0 of 0)",
    "– (0 of 0)",
    "– (0 of 0)",
    "– (0 of 0)",
    "– (0 of 0)",
  ]);
  for (const label of await region
    .locator(".apexcharts-yaxis-label tspan")
    .allTextContents())
    expect(label).toMatch(/^\d+\.\d{2}%$/);

  await counts.click();
  await expect
    .poll(async () =>
      (await chartConfig(page, "secondlook-answers"))?.series.map(
        (s) => s.data[166],
      ),
    )
    .toEqual([92, 3, 1, 1, 0]);
  await expect(legend("agreed")).toHaveText("agreed1744");
  await hoverHour(166);
  await expect(values).toHaveText(["92", "3", "1", "1", "0"]);
  await page.mouse.move(0, 0);

  await region.getByRole("button", { name: "About the answer rates" }).click();
  const info = page.getByTestId("answers-info");
  await expect(info).toContainText("Rates: numerator ÷ denominator");
  await expect(info).toContainText(
    "Denominator, the same for every line: tracked objects Secondlook was asked about that finished in that hour",
  );
  for (const name of names) await expect(info).toContainText(`${name}:`);
  await expect(info).toContainText("became a detection");
  await expect(info).not.toContainText(/\bgate\b/);
  await expect(info).toHaveCSS("opacity", "1");
  await page.screenshot({
    path: `${SCREENS}/answers-info-${testInfo.project.name}.png`,
  });
});

test("General answers from an earlier build's hourly data show zeros", async ({
  frigateApp,
}) => {
  await frigateApp.installDefaults();
  frigateApp.ws.statsOverride = slStats();
  await frigateApp.page.route("**/api/stats/history**", (route) =>
    route.fulfill({ json: [slStats()] }),
  );
  const old = hourly();
  const added = [
    "tracks_finished",
    "agreements",
    "results",
    "no_hit",
    "timeouts",
    "dropped",
    "fail_open",
  ];
  for (const hour of old.hours)
    for (const key of added) delete (hour as Record<string, unknown>)[key];
  await frigateApp.page.route(
    "**/api/secondlook/disagreements/hourly",
    (route) => route.fulfill({ json: old }),
  );
  await frigateApp.goto("/system#general");
  const region = frigateApp.page.getByRole("region", {
    name: "Secondlook answers per hour",
  });
  await expect(region.getByText("of 0 finished")).toBeVisible({
    timeout: 15_000,
  });
  await expect
    .poll(async () =>
      (await chartConfig(frigateApp.page, "secondlook-answers"))?.series.map(
        (s) => s.data[166],
      ),
    )
    .toEqual([null, null, null, null, null]);
});

test("General disagreements explain when saving is off", async ({
  frigateApp,
}) => {
  await frigateApp.installDefaults();
  frigateApp.ws.statsOverride = slStats();
  await frigateApp.page.route("**/api/stats/history**", (route) =>
    route.fulfill({ json: [slStats()] }),
  );
  await frigateApp.page.route(
    "**/api/secondlook/disagreements/hourly",
    (route) => route.fulfill({ status: 404, json: { success: false } }),
  );
  await frigateApp.goto("/system#general");
  const region = frigateApp.page.getByRole("region", {
    name: "Secondlook disagreements per hour",
  });
  await expect(
    region.getByText(/need Secondlook save and review enabled/),
  ).toBeVisible({ timeout: 15_000 });
  await expect(region.locator(".apexcharts-canvas")).toHaveCount(0);
});

test("storage metrics show disk used by Secondlook saved data", async ({
  frigateApp,
}) => {
  await frigateApp.installDefaults({
    config: {
      secondlook: { enabled: true, save_and_review: { enabled: true } },
    },
  });
  // the shared stats fixture has no /dev/shm entry, which the stock storage
  // page requires
  const base = slStats();
  const stats = {
    ...base,
    secondlook: {
      ...base.secondlook,
      storage: {
        ...base.secondlook.storage,
        retention: {
          last_run: 1_700_000_000,
          deleted_rows: 12,
          deleted_files: 11,
          missing_files: 1,
          deleted_orphans: 2,
        },
      },
    },
  };
  await frigateApp.page.route("**/api/stats", (route) =>
    route.fulfill({
      json: {
        ...stats,
        service: {
          ...stats.service,
          storage: {
            ...stats.service.storage,
            "/dev/shm": {
              free: 400000000,
              total: 500000000,
              used: 100000000,
              mount_type: "tmpfs",
            },
          },
        },
      },
    }),
  );
  await frigateApp.page.route("**/api/recordings/storage", (route) =>
    route.fulfill({
      json: { front_door: { bandwidth: 1, usage: 100, usage_percent: 10 } },
    }),
  );
  await frigateApp.goto("/system#storage");
  const section = frigateApp.page.getByRole("region", {
    name: "Secondlook saved data storage",
  });
  await expect(section).toBeVisible({ timeout: 15_000 });
  await expect(section.getByText("3.00 MiB")).toBeVisible();
  await expect(section.getByText("512.00 MiB")).toBeVisible();
  await expect(section.getByText("515.00 MiB")).toBeVisible();
  await expect(section.getByText(/1234 JPEG files/)).toBeVisible();
  await expect(
    section.getByText(
      /Cleanup last ran .+: deleted 12 decisions, 11 images, 2 orphan images/,
    ),
  ).toBeVisible();
});

test("general metrics explain a refused Secondlook model", async ({
  frigateApp,
}) => {
  const refusal = "model type 'yolonas' unsupported";
  await frigateApp.installDefaults();
  frigateApp.ws.statsOverride = slStats(refusal);
  await frigateApp.page.route("**/api/stats/history**", (route) =>
    route.fulfill({ json: [slStats(refusal)] }),
  );
  await frigateApp.goto("/system#general");
  const section = frigateApp.page.getByRole("region", {
    name: "Secondlook metrics",
  });
  await expect(section.getByText(/Model refused at startup/)).toBeVisible();
  await expect(section.getByText(new RegExp(refusal))).toBeVisible();
});
