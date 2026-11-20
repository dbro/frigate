import { test, expect } from "../fixtures/frigate-test";
import { PLACEHOLDER_PNG } from "../helpers/api-mocker";

test.describe("Secondlook save and review", () => {
  test("enabled shows Corrections and uses renamed list/still/mark API", async ({
    frigateApp,
  }) => {
    // Existing saved data remains reviewable even while live score adjustment is off.
    await frigateApp.installDefaults({
      config: {
        secondlook: { enabled: false, save_and_review: { enabled: true } },
      },
    });
    const example = {
      id: "saved-one",
      track_id: "track-one",
      camera: "front_door",
      kind: "false_positive",
      label: "person",
      primary_score: 0.9,
      sl_score: 0.1,
      factor: 0.0,
      box: null,
      mark: null,
    };
    await frigateApp.page.route(
      "**/api/secondlook/save_and_review**",
      async (route) => {
        if (route.request().url().endsWith("/still.jpg")) {
          return route.fulfill({
            contentType: "image/png",
            body: PLACEHOLDER_PNG,
          });
        }
        if (route.request().method() === "POST") {
          return route.fulfill({
            json: {
              success: true,
              example: { ...example, mark: "secondlook_right" },
            },
          });
        }
        return route.fulfill({ json: { examples: [example], unreviewed: 1 } });
      },
    );
    await frigateApp.goto("/review");
    const tab = frigateApp.page.getByRole("radio", {
      name: "Corrections",
      exact: true,
    });
    await expect(tab).toBeVisible();
    const listRequest = frigateApp.page.waitForRequest((r) =>
      r.url().includes("/api/secondlook/save_and_review?"),
    );
    const stillRequest = frigateApp.page.waitForRequest((r) =>
      r.url().endsWith("/api/secondlook/save_and_review/saved-one/still.jpg"),
    );
    await tab.click();
    await listRequest;
    await stillRequest;
    await expect(
      frigateApp.page.getByText("P 0.90", { exact: true }),
    ).toBeVisible();
    await expect(
      frigateApp.page.getByText("SL 0.10", { exact: true }),
    ).toBeVisible();
    await expect(
      frigateApp.page.getByText("× 0.00", { exact: true }),
    ).toBeVisible();
    const still = frigateApp.page.getByRole("img", {
      name: "person False positive",
      exact: true,
    });
    await expect(still).toHaveCount(1);
    await still.hover();
    await still.click();
    await expect(frigateApp.page.locator("video")).toHaveCount(0);
    const markRequest = frigateApp.page.waitForRequest(
      (r) =>
        r.method() === "POST" &&
        r.url().endsWith("/api/secondlook/save_and_review/saved-one/mark"),
    );
    await frigateApp.page
      .getByRole("button", { name: "Secondlook right", exact: true })
      .click();
    expect((await markRequest).postDataJSON()).toEqual({
      mark: "secondlook_right",
    });
  });

  test("disabled hides Corrections and refuses restored Corrections navigation", async ({
    frigateApp,
  }) => {
    await frigateApp.installDefaults({
      config: {
        secondlook: { enabled: true, save_and_review: { enabled: false } },
      },
    });
    const featureRequests: string[] = [];
    frigateApp.page.on("request", (r) => {
      if (r.url().includes("/secondlook/save_and_review"))
        featureRequests.push(r.url());
    });
    // React Router restores overlay state from browser history.
    await frigateApp.page.addInitScript(() => {
      window.history.replaceState(
        { ...window.history.state, usr: { severity: "correction" } },
        "",
      );
    });
    await frigateApp.goto("/review");
    await expect(
      frigateApp.page.getByRole("radio", { name: "Alerts", exact: true }),
    ).toHaveAttribute("data-state", "on");
    await expect(
      frigateApp.page.getByRole("radio", { name: "Corrections", exact: true }),
    ).toHaveCount(0);
    await expect(
      frigateApp.page.getByRole("button", { name: "Ignore", exact: true }),
    ).toHaveCount(0);
    expect(featureRequests).toEqual([]);
  });
});

test("all-history pagination, counts, long misclass cards and independent marks", async ({
  frigateApp,
}) => {
  await frigateApp.installDefaults({
    config: {
      secondlook: { enabled: true, save_and_review: { enabled: true } },
    },
  });
  function initialMark():
    | "primary_right"
    | "secondlook_right"
    | "ignore"
    | null {
    return null;
  }
  const rows = Array.from({ length: 221 }, (_, i) => ({
    id: `row-${i}`,
    track_id: `track-${i}`,
    camera: "front_door",
    kind: "misclass",
    label: "car",
    misclass_label: "truck",
    primary_score: 0.9,
    sl_score: 0.8,
    factor: 0.8,
    box: null,
    mark: initialMark(),
    // 1000 s apart: outside the 60 s grouping gap, so every row is its own card
    frame_time: i * 1000 + 1,
    evidence: null,
    group_key: null,
  }));
  await frigateApp.page.route(
    "**/api/secondlook/save_and_review**",
    async (route) => {
      const url = new URL(route.request().url());
      if (url.pathname.endsWith("/still.jpg"))
        return route.fulfill({
          contentType: "image/png",
          body: PLACEHOLDER_PNG,
        });
      if (url.pathname.endsWith("/mark")) {
        const row = rows.find((r) => url.pathname.includes(`/${r.id}/`));
        if (row) row.mark = route.request().postDataJSON().mark;
        return route.fulfill({ json: { success: true } });
      }
      const after = Number(url.searchParams.get("after") ?? 0);
      const before = Number(url.searchParams.get("before") ?? Infinity);
      const matching = rows.filter(
        (r) => r.frame_time >= after && r.frame_time <= before,
      );
      const visible = matching.filter(
        (r) => url.searchParams.get("reviewed") !== "false" || !r.mark,
      );
      const offset = Number(url.searchParams.get("offset") ?? 0),
        limit = Number(url.searchParams.get("limit") ?? 50);
      return route.fulfill({
        json: {
          examples: visible.slice(offset, offset + limit),
          total: visible.length,
          unreviewed: matching.filter((r) => !r.mark).length,
          has_more: offset + limit < visible.length,
        },
      });
    },
  );
  await frigateApp.goto("/review");
  await frigateApp.page
    .getByRole("radio", { name: "Corrections", exact: true })
    .click();
  await expect(
    frigateApp.page.getByText("221 unreviewed · 221 matching"),
  ).toBeVisible();
  await expect(
    frigateApp.page.getByText("All saved history", { exact: true }).last(),
  ).toBeVisible();
  await expect(frigateApp.page.getByTestId("correction-card")).toHaveCount(50);
  for (let i = 0; i < 4; i++) {
    await frigateApp.page
      .getByRole("button", { name: "Load more examples" })
      .click();
    await expect(frigateApp.page.getByTestId("correction-card")).toHaveCount(
      Math.min(221, 100 + i * 50),
    );
  }
  const last = frigateApp.page.getByTestId("correction-card").last();
  await last.scrollIntoViewIfNeeded();
  await expect(last.getByText("Primary: car", { exact: false })).toBeVisible();
  await expect(
    last.getByText("Secondlook: truck", { exact: false }),
  ).toBeVisible();
  await expect(last.getByRole("button", { name: "Ignore" })).toBeVisible();
  const scroller = frigateApp.page.getByTestId("corrections-scroll");
  expect(
    await scroller.evaluate((el) => el.scrollHeight > el.clientHeight),
  ).toBeTruthy();
  const firstHeight = await frigateApp.page
    .getByTestId("correction-card")
    .first()
    .evaluate((el) => el.clientHeight);
  expect(firstHeight).toBeGreaterThan(150);
  await last.getByRole("button", { name: "Ignore" }).click();
  await expect(
    frigateApp.page.getByText("220 unreviewed · 220 matching"),
  ).toBeVisible();
  expect(rows.filter((r) => r.mark === "ignore")).toHaveLength(1);
  if (frigateApp.isMobile) {
    await frigateApp.page
      .getByRole("button", { name: "Filters", exact: true })
      .click();
    await frigateApp.page
      .getByRole("button", { name: "Calendar", exact: true })
      .click();
  } else {
    await frigateApp.page
      .getByRole("button", { name: "Select a date to filter by" })
      .click();
  }
  await frigateApp.page
    .getByRole("gridcell")
    .getByRole("button")
    .first()
    .click();
  await frigateApp.page.keyboard.press("Escape");
  await expect(
    frigateApp.page.getByText("0 unreviewed · 0 matching"),
  ).toBeVisible();
  await frigateApp.page
    .getByRole("button", { name: "All saved history" })
    .click();
  await expect(
    frigateApp.page.getByText("220 unreviewed · 220 matching"),
  ).toBeVisible();
});

test("proximity groups expand losslessly and never share verdicts @mobile", async ({
  frigateApp,
}) => {
  await frigateApp.installDefaults({
    config: {
      secondlook: { enabled: true, save_and_review: { enabled: true } },
    },
  });
  const marks = new Map<string, string>();
  const row = (id: string, frame_time: number, misclass_label = "truck") => ({
    id,
    track_id: id,
    camera: "front_door",
    kind: "misclass",
    label: "car",
    misclass_label,
    primary_score: 0.8,
    sl_score: 0.9,
    factor: 0,
    box: null,
    // different saved frames: the old exact-bytes key would never group them
    group_key: `exact-${id}`,
    frame_time,
    created_at: frame_time,
    mark: null,
  });
  await frigateApp.page.route(
    "**/api/secondlook/save_and_review**",
    async (route) => {
      const url = new URL(route.request().url());
      if (url.pathname.endsWith("/still.jpg"))
        return route.fulfill({
          contentType: "image/png",
          body: PLACEHOLDER_PNG,
        });
      if (route.request().method() === "POST") {
        marks.set(
          url.pathname.split("/").at(-2) ?? "",
          route.request().postDataJSON().mark,
        );
        return route.fulfill({ json: { success: true } });
      }
      // newest first, as the API orders them
      const rows = [
        row("one", 10_000),
        row("two", 9_970),
        row("three", 9_930),
        row("bus", 9_920, "bus"),
        row("later-loop", 5_000),
      ].filter((r) => !marks.has(r.id));
      return route.fulfill({
        json: {
          examples: rows,
          total: rows.length,
          unreviewed: rows.length,
          has_more: false,
        },
      });
    },
  );
  await frigateApp.goto("/review");
  await frigateApp.page
    .getByRole("radio", { name: "Corrections", exact: true })
    .click();
  const group = frigateApp.page.getByTestId("correction-group");
  await expect(group).toHaveCount(1);
  await expect(
    group.getByText("3 separate decisions · Misclass"),
  ).toBeVisible();
  await expect(group.getByText(/over 70s \(gaps ≤ 60s\)/)).toBeVisible();
  // collapsed: one representative card + the two ungrouped rows
  await expect(frigateApp.page.getByTestId("correction-card")).toHaveCount(3);
  await group.getByRole("button", { name: "Expand 3 examples" }).click();
  await expect(group.getByTestId("correction-card")).toHaveCount(3);
  await expect(frigateApp.page.getByTestId("correction-card")).toHaveCount(5);
  await group
    .getByTestId("correction-card")
    .nth(1)
    .getByRole("button", { name: "Primary right", exact: true })
    .click();
  await expect(
    frigateApp.page.getByText("4 unreviewed · 4 matching"),
  ).toBeVisible();
  expect([...marks.entries()]).toEqual([["two", "primary_right"]]);
  await frigateApp.page
    .getByRole("radio", { name: "No grouping", exact: true })
    .click();
  await expect(frigateApp.page.getByTestId("correction-group")).toHaveCount(0);
  await expect(frigateApp.page.getByTestId("correction-card")).toHaveCount(4);
});

test("FN overlays use their own non-square frame and historical evidence stays explicitly unavailable", async ({
  frigateApp,
}) => {
  await frigateApp.installDefaults({
    config: {
      secondlook: { enabled: true, save_and_review: { enabled: true } },
    },
  });
  await frigateApp.page.route(
    "**/api/secondlook/save_and_review**",
    (route) => {
      if (new URL(route.request().url()).pathname.endsWith("/still.jpg"))
        return route.fulfill({
          contentType: "image/svg+xml",
          body: '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="480"><rect width="640" height="480" fill="gray"/></svg>',
        });
      return route.fulfill({
        json: {
          total: 2,
          unreviewed: 2,
          has_more: false,
          examples: [true, false].map((aligned) => ({
            id: aligned ? "aligned" : "historical",
            camera: "front_door",
            kind: "false_negative",
            label: "person",
            primary_score: 0.5,
            sl_score: 0.9,
            factor: 1.8,
            mark: null,
            evidence: aligned
              ? {
                  version: 1,
                  width: 640,
                  height: 480,
                  primary_box: [10, 20, 110, 220],
                  sl_box: [15, 25, 115, 225],
                  secondary_label: "person",
                }
              : null,
          })),
        },
      });
    },
  );
  await frigateApp.goto("/review");
  await frigateApp.page
    .getByRole("radio", { name: "Corrections", exact: true })
    .click();
  const card = frigateApp.page.getByTestId("correction-card").first();
  const overlay = card.getByRole("img", {
    name: "Primary and Secondlook boxes",
  });
  await expect(overlay).toHaveAttribute("viewBox", "0 0 640 480");
  await expect(overlay.locator("rect")).toHaveCount(2);
  await expect(overlay.locator("rect").first()).toHaveAttribute("x", "10");
  await expect(overlay.locator("rect").nth(1)).toHaveAttribute("x", "15");
  await expect(
    frigateApp.page.getByText(
      "Overlay unavailable: historical frame alignment not established",
    ),
  ).toBeVisible();
  const dimensions = await card.locator("img").evaluate((image) => ({
    width: image.naturalWidth,
    height: image.naturalHeight,
  }));
  expect(dimensions).toEqual({ width: 640, height: 480 });
});
