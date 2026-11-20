import type { Page, Route } from "@playwright/test";
import { existsSync, readFileSync } from "node:fs";
import { test, expect } from "../fixtures/frigate-test";
import { PLACEHOLDER_PNG } from "../helpers/api-mocker";
import { grantClipboardPermissions, readClipboard } from "../helpers/clipboard";

// Mocked HTTP fixtures (no Frigate backend): the API shapes are the ones
// covered by frigate/test/test_secondlook_clusters.py.

const SCREENS = process.env.SECONDLOOK_SCREENS ?? "test-results/secondlook-screens";
// Optional real camera frame for the screenshots; a generated still is used otherwise.
const FRAME = process.env.SECONDLOOK_FRAME ?? "";
const still = existsSync(FRAME)
  ? { contentType: "image/jpeg", body: readFileSync(FRAME) }
  : { contentType: "image/png", body: PLACEHOLDER_PNG };

const evidence = (box: number[]) => ({
  version: 1,
  width: 1280,
  height: 720,
  frame_time: 1,
  primary_box: box,
  sl_box: null,
  crop_region: [0, 0, 640, 640],
  secondary_label: "person",
  secondary_score: 0.1,
});

type Row = Record<string, unknown>;
const example = (
  id: string,
  kind: string,
  event_id: string | null,
  i: number,
  box = [350, 210, 460, 360],
): Row => ({
  id,
  track_id: id,
  camera: "front_door",
  kind,
  label: "person",
  misclass_label: kind === "misclass" ? "dog" : null,
  primary_score: 0.74,
  sl_score: 0.12,
  factor: kind === "false_negative" ? 1.6 : 0.2,
  box,
  sl_box: null,
  mark: null,
  event_id,
  frame_time: 1_763_500_000 + i * 3600,
  created_at: 1_763_500_000 + i * 3600,
  evidence: evidence(box),
  live_outcome: "none",
});

const sample = (
  id: string,
  i: number,
  box: number[],
  kind = "false_positive",
) => ({
  id,
  time: 1_763_500_000 + i * 3600,
  kind,
  label: "person",
  misclass_label: kind === "misclass" ? "dog" : null,
  camera: "front_door",
  primary_score: 0.74,
  sl_score: 0.12,
  factor: 0.2,
  mark: null,
  box,
  evidence: evidence(box),
});

const KEY_FP = JSON.stringify({
  camera: "front_door",
  cells: [
    [2, 2],
    [3, 2],
  ],
  kind: "false_positive",
  label: "person",
  misclass_label: null,
  outcome: "acted",
  size: 1,
});
const GROUPS = [
  {
    key: KEY_FP,
    camera: "front_door",
    kind: "false_positive",
    outcome: "acted",
    label: "person",
    misclass_label: null,
    size: 1,
    cells: [
      [2, 2],
      [3, 2],
    ],
    cell_counts: [
      [2, 2, 17],
      [3, 2, 6],
    ],
    count: 23,
    unreviewed: 20,
    first_seen: 1_763_500_000,
    last_seen: 1_763_590_000,
    extent: [0.27, 0.29, 0.37, 0.5],
    samples: Array.from({ length: 6 }, (_, i) =>
      sample(`fp-${i}`, i, [
        345 + i * 3,
        205 + i * 2,
        460 + i * 3,
        360 + i * 2,
      ]),
    ),
  },
  {
    key: JSON.stringify({
      camera: "front_door",
      cells: [[5, 4]],
      kind: "false_negative",
      label: "person",
      misclass_label: null,
      outcome: "not_acted",
      size: 3,
    }),
    camera: "front_door",
    kind: "false_negative",
    outcome: "not_acted",
    label: "person",
    misclass_label: null,
    size: 3,
    cells: [[5, 4]],
    cell_counts: [[5, 4, 9]],
    count: 9,
    unreviewed: 9,
    first_seen: 1_763_400_000,
    last_seen: 1_763_580_000,
    extent: [0.6, 0.55, 0.78, 0.95],
    samples: Array.from({ length: 3 }, (_, i) =>
      sample(`fn-${i}`, i, [800, 420, 980, 690], "false_negative"),
    ),
  },
  {
    key: JSON.stringify({
      camera: "front_door",
      cells: [],
      kind: "misclass",
      label: "person",
      misclass_label: "dog",
      outcome: null,
      size: null,
    }),
    camera: "front_door",
    kind: "misclass",
    outcome: null,
    label: "person",
    misclass_label: "dog",
    size: null,
    cells: [],
    cell_counts: [],
    count: 4,
    unreviewed: 4,
    first_seen: 1_763_300_000,
    last_seen: 1_763_310_000,
    extent: null,
    samples: [sample("mc-0", 0, [0, 0, 1, 1], "misclass")],
  },
];

const MASK = {
  camera: "front_door",
  label: "person",
  name: "secondlook_person_3fa2c1",
  friendly_name: "Secondlook suggestion: person false positives (23)",
  members: 23,
  points: "boxes",
  margin: 0.02,
  polygon: [
    [0.25, 0.258],
    [0.385, 0.258],
    [0.385, 0.525],
    [0.25, 0.525],
  ],
  coordinates: "0.250,0.258,0.385,0.258,0.385,0.525,0.250,0.525",
  area_fraction: 0.036,
  bottom_centres_inside: 23,
  existing_masks: [],
  yaml: 'cameras:\n  front_door:\n    objects:\n      filters:\n        person:\n          mask:\n            secondlook_person_3fa2c1:\n              friendly_name: "Secondlook suggestion: person false positives (23)"\n              enabled: true\n              coordinates: "0.250,0.258,0.385,0.258,0.385,0.525,0.250,0.525"\n',
  boxes: Array.from({ length: 6 }, (_, i) => [
    (345 + i * 3) / 1280,
    (205 + i * 2) / 720,
    (460 + i * 3) / 1280,
    (360 + i * 2) / 720,
  ]),
  preview_id: "fp-0",
};

/** All Corrections routes; records every request's path and query. */
async function mockCorrections(page: Page, rows: Row[], requests: URL[]) {
  await page.route("**/api/secondlook/save_and_review**", (route: Route) => {
    const url = new URL(route.request().url());
    requests.push(url);
    if (url.pathname.endsWith("/still.jpg")) return route.fulfill(still);
    if (url.pathname.endsWith("/groups/mask"))
      return route.fulfill({ json: MASK });
    if (url.pathname.endsWith("/groups/mark"))
      return route.fulfill({ json: { success: true, updated: 20 } });
    if (url.pathname.endsWith("/groups"))
      return route.fulfill({
        json: {
          groups: GROUPS,
          total_groups: 3,
          total_rows: 36,
          has_more: false,
          grid: { columns: 8, rows: 6 },
          size_labels: ["<1%", "1–2.5%", "2.5–5%", "5–10%", ">10%"],
        },
      });
    const q = url.searchParams;
    let shown = rows;
    if (q.get("kind")) shown = shown.filter((r) => r.kind === q.get("kind"));
    const outcome = q.get("outcome");
    if (outcome)
      shown = shown.filter((r) => {
        const acted =
          r.kind === "false_negative"
            ? r.event_id !== null
            : r.event_id === null;
        return r.kind !== "misclass" && (outcome === "acted") === acted;
      });
    if (q.get("group"))
      shown = shown.filter((r) => String(r.id).startsWith("fp"));
    return route.fulfill({
      json: {
        examples: shown,
        total: shown.length,
        unreviewed: shown.length,
        has_more: false,
      },
    });
  });
}

const ROWS = [
  example("fp-0", "false_positive", null, 0),
  example("fp-1", "false_positive", null, 0.01),
  example("fp-2", "false_positive", "e2", 2),
  example("fn-0", "false_negative", "e3", 3, [800, 420, 980, 690]),
  example("fn-1", "false_negative", null, 4, [800, 420, 980, 690]),
  example("mc-0", "misclass", null, 5),
];

async function openCorrections(page: Page) {
  await page.getByRole("radio", { name: "Corrections", exact: true }).click();
  await expect(page.getByTestId("grouping-mode")).toBeVisible({
    timeout: 15_000,
  });
}

test("FP/FN outcome filter: plain labels, server-side param, hidden for All and misclass @mobile", async ({
  frigateApp,
}, testInfo) => {
  await frigateApp.installDefaults({
    config: {
      secondlook: { enabled: true, save_and_review: { enabled: true } },
    },
  });
  const requests: URL[] = [];
  await mockCorrections(frigateApp.page, ROWS, requests);
  await frigateApp.goto("/review");
  const page = frigateApp.page;
  await openCorrections(page);
  const outcome = page.getByTestId("outcome-filter");
  await expect(outcome).toHaveCount(0);
  await page
    .getByRole("radio", { name: "False positive", exact: true })
    .click();
  await expect(outcome.getByRole("radio")).toHaveText([
    "All",
    "Suppressed",
    "Not suppressed",
  ]);
  const last = () =>
    requests.filter((u) => u.pathname.endsWith("/save_and_review")).at(-1)!;
  await outcome
    .getByRole("radio", { name: "Not suppressed", exact: true })
    .click();
  await expect.poll(() => last().searchParams.get("outcome")).toBe("not_acted");
  expect(last().searchParams.get("kind")).toBe("false_positive");
  await expect(page.getByTestId("correction-card")).toHaveCount(1);
  await expect(
    page
      .getByText("0 unreviewed · 1 matching")
      .or(page.getByText("1 unreviewed · 1 matching")),
  ).toBeVisible();
  await page.screenshot({
    path: `${SCREENS}/filter-fp-not-suppressed-${testInfo.project.name}.png`,
  });
  await page
    .getByRole("radio", { name: "False negative", exact: true })
    .click();
  await expect(outcome.getByRole("radio")).toHaveText([
    "All",
    "Rescued",
    "Not rescued",
  ]);
  await outcome.getByRole("radio", { name: "Rescued", exact: true }).click();
  await expect.poll(() => last().searchParams.get("outcome")).toBe("acted");
  await expect(page.getByTestId("correction-card")).toHaveCount(1);
  await page.screenshot({
    path: `${SCREENS}/filter-fn-rescued-${testInfo.project.name}.png`,
  });
  await page.getByRole("radio", { name: "Misclass", exact: true }).click();
  await expect(outcome).toHaveCount(0);
  await expect.poll(() => last().searchParams.get("kind")).toBe("misclass");
  expect(last().searchParams.get("outcome")).toBeNull();
});

test("grouping toggle: no grouping, short-term and long-term with drill-in and bulk verdict @mobile", async ({
  frigateApp,
}, testInfo) => {
  await frigateApp.installDefaults({
    config: {
      secondlook: { enabled: true, save_and_review: { enabled: true } },
    },
  });
  const requests: URL[] = [];
  await mockCorrections(frigateApp.page, ROWS, requests);
  await frigateApp.goto("/review");
  const page = frigateApp.page;
  await openCorrections(page);
  const mode = page.getByTestId("grouping-mode");
  await expect(mode.getByRole("radio")).toHaveText([
    "No grouping",
    "Short-term",
    "Long-term",
  ]);
  await expect(mode.getByRole("radio", { name: "Short-term" })).toHaveAttribute(
    "data-state",
    "on",
  );
  // fp-0 and fp-1 are 36 s apart: one short-term group
  await expect(page.getByTestId("correction-group")).toHaveCount(1);
  await page.screenshot({
    path: `${SCREENS}/grouping-short-${testInfo.project.name}.png`,
  });
  await mode.getByRole("radio", { name: "No grouping" }).click();
  await expect(page.getByTestId("correction-group")).toHaveCount(0);
  await expect(page.getByTestId("correction-card")).toHaveCount(6);
  await page.screenshot({
    path: `${SCREENS}/grouping-none-${testInfo.project.name}.png`,
  });

  await mode.getByRole("radio", { name: "Long-term" }).click();
  const groups = page.getByTestId("long-term-group");
  await expect(groups).toHaveCount(3);
  await expect(groups.first()).toContainText(
    "23 decisions · False positive · Suppressed · person",
  );
  await expect(groups.nth(1)).toContainText(
    "9 decisions · False negative · Not rescued · person",
  );
  await expect(groups.nth(2)).toContainText("person → dog");
  await expect(groups.nth(2)).toContainText("location unknown");
  await expect(
    groups.first().getByRole("button", { name: /^Sample/ }),
  ).toHaveCount(6);
  await expect(
    groups.first().getByRole("button", { name: "Suggest mask" }),
  ).toBeVisible();
  await expect(
    groups.nth(1).getByRole("button", { name: "Suggest mask" }),
  ).toHaveCount(0);
  await expect(
    page.getByText("3 groups · 36 decisions, most first"),
  ).toBeVisible();
  const groupReq = requests
    .filter((u) => u.pathname.endsWith("/groups"))
    .at(-1)!;
  expect(groupReq.searchParams.get("reviewed")).toBe("false");
  await page.screenshot({
    path: `${SCREENS}/grouping-long-${testInfo.project.name}.png`,
  });

  // time window narrows the request
  await page.getByRole("combobox", { name: "Time window" }).click();
  await page.getByRole("option", { name: "Last 7 days" }).click();
  await expect
    .poll(() =>
      Number(
        requests
          .filter((u) => u.pathname.endsWith("/groups"))
          .at(-1)!
          .searchParams.get("after"),
      ),
    )
    .toBeGreaterThan(Date.now() / 1000 - 7 * 86400 - 60);

  // bulk verdict: confirm, then POST with the group key
  await groups.first().getByRole("button", { name: "Ignore (20)" }).click();
  await expect(page.getByRole("alertdialog")).toContainText(
    "the other 3 keep theirs",
  );
  await page.screenshot({
    path: `${SCREENS}/grouping-long-bulk-confirm-${testInfo.project.name}.png`,
  });
  const markReq = page.waitForRequest((r) => r.url().includes("/groups/mark"));
  await page.getByRole("button", { name: "Mark 20" }).click();
  const mark = await markReq;
  expect(mark.postDataJSON()).toEqual({ mark: "ignore" });
  expect(new URL(mark.url()).searchParams.get("group")).toBe(KEY_FP);

  // drill-in lists the members through the same list API
  await groups.first().getByRole("button", { name: "Show all 23" }).click();
  await expect(page.getByTestId("group-drill-in")).toContainText(
    "23 decisions · False positive · Suppressed · person",
  );
  await expect
    .poll(() =>
      requests
        .filter((u) => u.pathname.endsWith("/save_and_review"))
        .at(-1)!
        .searchParams.get("group"),
    )
    .toBe(KEY_FP);
  await expect(page.getByTestId("correction-card")).toHaveCount(3);
  await page.screenshot({
    path: `${SCREENS}/grouping-long-drill-in-${testInfo.project.name}.png`,
  });
  await page.getByRole("button", { name: "Back to groups" }).click();
  await expect(groups).toHaveCount(3);
});

test("suggest mask: preview polygon over a frame, YAML and copy @mobile", async ({
  frigateApp,
  context,
}, testInfo) => {
  await grantClipboardPermissions(context);
  await frigateApp.installDefaults({
    config: {
      secondlook: { enabled: true, save_and_review: { enabled: true } },
    },
  });
  const requests: URL[] = [];
  await mockCorrections(frigateApp.page, ROWS, requests);
  await frigateApp.goto("/review");
  const page = frigateApp.page;
  await openCorrections(page);
  await page
    .getByTestId("grouping-mode")
    .getByRole("radio", { name: "Long-term" })
    .click();
  await page
    .getByTestId("long-term-group")
    .first()
    .getByRole("button", { name: "Suggest mask" })
    .click();
  const dialog = page.getByTestId("mask-dialog");
  await expect(dialog.getByTestId("mask-yaml")).toContainText(
    "cameras:\n  front_door:\n    objects:\n      filters:\n        person:\n          mask:",
  );
  await expect(
    dialog.getByTestId("mask-preview").locator("polygon"),
  ).toHaveAttribute("points", "0.25,0.258 0.385,0.258 0.385,0.525 0.25,0.525");
  await expect(dialog.getByTestId("mask-preview").locator("rect")).toHaveCount(
    6,
  );
  await expect(dialog).toContainText("3.60% of the frame");
  await expect(dialog).toContainText(
    "23 of 23 saved bottom centres fall inside",
  );
  await expect(dialog).toContainText("Nothing is changed in your config");
  const req = requests
    .filter((u) => u.pathname.endsWith("/groups/mask"))
    .at(-1)!;
  expect(req.searchParams.get("group")).toBe(KEY_FP);
  expect(req.searchParams.get("points")).toBe("boxes");
  await expect(dialog.getByTestId("mask-preview-image")).toHaveJSProperty(
    "complete",
    true,
  );
  await expect(dialog).toHaveCSS("opacity", "1");
  await page.screenshot({
    path: `${SCREENS}/mask-suggestion-${testInfo.project.name}.png`,
  });
  await dialog
    .getByRole("button", { name: "Copy YAML" })
    .scrollIntoViewIfNeeded();
  await page.screenshot({
    path: `${SCREENS}/mask-suggestion-yaml-${testInfo.project.name}.png`,
  });
  await dialog.getByRole("button", { name: "Copy YAML" }).click();
  // the clipboard drops the final newline; the YAML is otherwise exact
  expect((await readClipboard(page)).trimEnd()).toBe(MASK.yaml.trimEnd());
  await dialog.getByRole("radio", { name: "Bottom centres only" }).click();
  await expect
    .poll(() =>
      requests
        .filter((u) => u.pathname.endsWith("/groups/mask"))
        .at(-1)!
        .searchParams.get("points"),
    )
    .toBe("bottom");
});
