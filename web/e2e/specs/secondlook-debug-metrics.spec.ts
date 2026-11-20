import { test, expect } from "../fixtures/frigate-test";
import { PLACEHOLDER_PNG } from "../helpers/api-mocker";
import { BASE_STATS } from "../fixtures/mock-data/stats";

test("Debug isolates secondary boxes on their analysed still, including disabled and empty states @mobile", async ({
  frigateApp,
}) => {
  await frigateApp.installDefaults();
  await frigateApp.page.route("**/api/front_door/latest.webp?**", (route) =>
    route.fulfill({ contentType: "image/png", body: PLACEHOLDER_PNG }),
  );
  let state: "disabled" | "no_result" | "ready" = "disabled";
  const currentFrameRequests: string[] = [];
  frigateApp.page.on("request", (request) => {
    if (request.url().includes("/latest."))
      currentFrameRequests.push(request.url());
  });
  await frigateApp.page.route("**/api/front_door/secondlook", (route) =>
    route.fulfill({
      json: {
        state,
        frame:
          state === "ready"
            ? {
                version: 1,
                jpeg: PLACEHOLDER_PNG.toString("base64"),
                frame_time: Date.now() / 1000 - 30,
                width: 640,
                height: 480,
                primary_box: [100, 80, 200, 240],
                sl_box: [110, 90, 210, 250],
                label: "car",
                secondary_label: "truck",
                primary_score: 0.8,
                secondary_score: 0.95,
              }
            : null,
      },
    }),
  );
  await frigateApp.goto("/?debug=true#front_door");
  await frigateApp.page
    .getByRole("switch", { name: "Secondlook analysed frame" })
    .click();
  const inset = frigateApp.page.getByRole("region", {
    name: "Secondlook analysed frame",
  });
  await expect(
    inset.getByText("Secondlook disabled for this camera."),
  ).toBeVisible();
  state = "no_result";
  await expect(inset.getByText("No analysed result yet.")).toBeVisible();
  state = "ready";
  await expect(
    inset.getByText("Primary: car 0.800 · Secondlook: truck 0.950"),
  ).toBeVisible();
  await expect(inset.getByText(/stale; not current video/)).toBeVisible();
  await expect(inset.getByAltText("Secondlook analysed frame")).toBeVisible();
  await expect(inset.locator("svg rect")).toHaveCount(2);
  expect(
    currentFrameRequests.some((url) => url.includes("/api/front_door/latest.")),
  ).toBe(true);
  expect(
    currentFrameRequests.every(
      (url) => !new URL(url).searchParams.has("secondlook"),
    ),
  ).toBe(true);
});

for (const mode of ["disabled", "unavailable", "waiting", "sampled"]) {
  test(`General Secondlook metrics honestly show ${mode}`, async ({
    frigateApp,
  }) => {
    await frigateApp.installDefaults();
    frigateApp.ws.statsOverride = {
      ...BASE_STATS,
      secondlook: {
        enabled: mode !== "disabled",
        available: mode !== "unavailable",
        samples: mode === "sampled" ? 1 : 0,
        inference_ms: mode === "sampled" ? 12 : null,
        processing_ms: mode === "sampled" ? 15 : null,
        queue_ms: mode === "sampled" ? 2 : null,
        last_updated: Date.now() / 1000,
      },
    };
    await frigateApp.page.route("**/api/stats/history**", (route) =>
      route.fulfill({
        json: [
          {
            ...BASE_STATS,
            secondlook: {
              enabled: mode !== "disabled",
              available: mode !== "unavailable",
              samples: mode === "sampled" ? 1 : 0,
              inference_ms: mode === "sampled" ? 12 : null,
              processing_ms: mode === "sampled" ? 15 : null,
              queue_ms: mode === "sampled" ? 2 : null,
              last_updated: Date.now() / 1000,
            },
          },
        ],
      }),
    );
    await frigateApp.goto("/system#general");
    const section = frigateApp.page.getByRole("region", {
      name: "Secondlook metrics",
    });
    await section
      .getByRole("button", { name: "About Secondlook metrics" })
      .click();
    await expect(
      frigateApp.page.getByText(/shared embeddings process/),
    ).toBeVisible();
    await frigateApp.page.keyboard.press("Escape");
    if (mode === "disabled")
      await expect(
        section.getByText("Secondlook disabled or statistics unavailable."),
      ).toBeVisible();
    await expect(
      frigateApp.page.getByRole("region", {
        name: "Secondlook disagreements per hour",
      }),
    ).toHaveCount(mode === "disabled" ? 0 : 1);
    if (mode === "unavailable")
      await expect(
        section.getByText("No real Secondlook model available."),
      ).toBeVisible();
    if (mode === "waiting")
      await expect(
        section.getByText("Waiting for completed inference."),
      ).toBeVisible();
    if (mode === "sampled") {
      // processing time + peak items; inference is a subset of processing
      await expect(
        section.getByText("Secondlook processing time"),
      ).toBeVisible();
      await expect(
        section.getByText("Secondlook queue (peak items)"),
      ).toBeVisible();
      await expect(section.locator(".apexcharts-canvas")).toHaveCount(2);
      await expect(section.getByText(/inference/i)).toHaveCount(0);
    } else await expect(section.locator(".apexcharts-canvas")).toHaveCount(0);
  });
}
