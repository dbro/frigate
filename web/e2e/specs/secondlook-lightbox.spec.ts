import type { Page } from "@playwright/test";
import { test, expect, type FrigateApp } from "../fixtures/frigate-test";
import { PLACEHOLDER_PNG } from "../helpers/api-mocker";

// Mocked HTTP fixtures (no Frigate backend): the Corrections lightbox.

const SCREENS = process.env.SECONDLOOK_SCREENS ?? "test-results/secondlook-screens";

type Row = {
  id: string;
  kind: string;
  live_outcome: string;
  mark: string | null;
  frame_time: number;
  primary_score: number;
  sl_score: number;
};

const row = (
  id: string,
  kind: string,
  live_outcome: string,
  minutes: number,
  primary_score: number,
  sl_score: number,
): Row => ({
  id,
  kind,
  live_outcome,
  mark: null,
  frame_time: 1_700_000_000 + minutes * 600,
  primary_score,
  sl_score,
});

async function setup(frigateApp: FrigateApp) {
  await frigateApp.installDefaults({
    config: {
      secondlook: { enabled: true, save_and_review: { enabled: true } },
    },
  });
  // newest first, 10 min apart: three separate cards
  const rows = [
    row("a", "false_negative", "not_rescued", 3, 0.42, 0.81),
    row("b", "false_positive", "suppressed", 2, 0.77, 0.12),
    row("c", "false_negative", "rescued", 1, 0.55, 0.9),
  ];
  const marks: { id: string; mark: string | null }[] = [];
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
        const id = url.pathname.split("/").at(-2) ?? "";
        const mark = route.request().postDataJSON().mark;
        marks.push({ id, mark });
        const target = rows.find((r) => r.id === id);
        if (target) target.mark = mark;
        return route.fulfill({ json: { success: true } });
      }
      const visible = rows.filter(
        (r) => url.searchParams.get("reviewed") !== "false" || !r.mark,
      );
      return route.fulfill({
        json: {
          examples: visible.map((r) => ({
            ...r,
            track_id: r.id,
            camera: "front_door",
            label: "person",
            misclass_label: null,
            factor: r.sl_score / r.primary_score,
            box: null,
            created_at: r.frame_time,
            evidence: {
              version: 1,
              width: 1280,
              height: 720,
              frame_time: r.frame_time,
              primary_box: [500, 200, 700, 600],
              sl_box: [510, 210, 710, 610],
              crop_region: [0, 0, 640, 640],
              secondary_label: r.kind === "false_positive" ? "truck" : "person",
              secondary_score: r.sl_score,
            },
          })),
          total: visible.length,
          unreviewed: rows.filter((r) => !r.mark).length,
          has_more: false,
        },
      });
    },
  );
  await frigateApp.goto("/review");
  await frigateApp.page
    .getByRole("radio", { name: "Corrections", exact: true })
    .click();
  await expect(frigateApp.page.getByTestId("correction-card")).toHaveCount(3, {
    timeout: 15_000,
  });
  return { rows, marks };
}

const lightbox = (page: Page) => page.getByTestId("correction-lightbox");
const trigger = (page: Page, id: string) =>
  page.locator(`[data-lightbox-trigger="${id}"]`);

async function open(page: Page, id: string) {
  await trigger(page, id).click();
  await expect(lightbox(page)).toBeVisible();
}

/** Click a corner of the dark stage, away from the 1x1 picture in its centre. */
async function clickBackdrop(page: Page) {
  const box = await page.getByTestId("lightbox-stage").boundingBox();
  if (!box) throw new Error("no stage");
  await page.mouse.click(box.x + 8, box.y + box.height - 8);
}

test("lightbox opens full size with boxes, scores and badge, steps and closes @mobile", async ({
  frigateApp,
}, testInfo) => {
  const page = frigateApp.page;
  await setup(frigateApp);
  await open(page, "a");
  const box = lightbox(page);
  await expect(box).toHaveAttribute("role", "dialog");
  await expect(box.getByTestId("lightbox-image")).toHaveAttribute(
    "src",
    /\/api\/secondlook\/save_and_review\/a\/still\.jpg$/,
  );
  const overlay = box.getByRole("img", { name: /Primary box/ });
  await expect(overlay.locator("rect")).toHaveCount(2);
  await expect(overlay.locator("rect").first()).toHaveAttribute(
    "stroke",
    "#00ffff",
  );
  await expect(overlay.locator("rect").nth(1)).toHaveAttribute(
    "stroke",
    "#ff9900",
  );
  await expect(overlay.locator("text")).toHaveText([
    "Primary person 0.42",
    "Secondlook person 0.81",
  ]);
  await expect(box.getByTestId("live-outcome")).toHaveText("Not rescued");
  await expect(box.getByTestId("lightbox-position")).toHaveText(
    "1 of 3 loaded",
  );
  await expect(
    box.getByRole("button", { name: "Primary right" }),
  ).toBeVisible();
  // fills the screen on mobile
  const full = await box.boundingBox();
  const viewport = page.viewportSize();
  expect(full?.width).toBe(viewport?.width);
  expect(full?.height).toBe(viewport?.height);
  await page.screenshot({
    path: `${SCREENS}/lightbox-${testInfo.project.name}.png`,
  });

  // keyboard and buttons step through the loaded list
  await page.keyboard.press("ArrowRight");
  await expect(box.getByTestId("lightbox-position")).toHaveText(
    "2 of 3 loaded",
  );
  await expect(box.getByTestId("live-outcome")).toHaveText(
    "Suppressed: alert prevented",
  );
  await expect(overlay.locator("text").nth(1)).toHaveText(
    "Secondlook truck 0.12",
  );
  await box.getByRole("button", { name: "Next item" }).click();
  await expect(box.getByTestId("lightbox-position")).toHaveText(
    "3 of 3 loaded",
  );
  await expect(box.getByRole("button", { name: "Next item" })).toHaveCount(0);
  await page.keyboard.press("ArrowRight"); // stays on the last one
  await expect(box.getByTestId("lightbox-position")).toHaveText(
    "3 of 3 loaded",
  );
  await page.keyboard.press("ArrowLeft");
  await expect(box.getByTestId("lightbox-position")).toHaveText(
    "2 of 3 loaded",
  );
  await box.getByRole("button", { name: "Previous item" }).click();
  await expect(box.getByTestId("lightbox-position")).toHaveText(
    "1 of 3 loaded",
  );

  // focus stays inside (trap)
  for (let i = 0; i < 12; i++) await page.keyboard.press("Tab");
  expect(await box.evaluate((el) => el.contains(document.activeElement))).toBe(
    true,
  );

  // Esc closes and focus returns to the card shown last
  await page.keyboard.press("ArrowRight");
  await page.keyboard.press("Escape");
  await expect(lightbox(page)).toHaveCount(0);
  await expect(trigger(page, "b")).toBeFocused();

  // close button
  await open(page, "c");
  await lightbox(page).getByRole("button", { name: "Close" }).click();
  await expect(lightbox(page)).toHaveCount(0);
  await expect(trigger(page, "c")).toBeFocused();

  // a click on the picture keeps it open; a click outside closes
  await open(page, "a");
  const picture = await lightbox(page)
    .getByTestId("lightbox-stage")
    .boundingBox();
  if (!picture) throw new Error("no stage");
  await page.mouse.click(
    picture.x + picture.width / 2,
    picture.y + picture.height / 2,
  );
  await expect(lightbox(page)).toBeVisible();
  await clickBackdrop(page);
  await expect(lightbox(page)).toHaveCount(0);
});

test("marking inside the lightbox: stays on the item, or shows the next when it leaves the list @mobile", async ({
  frigateApp,
}, testInfo) => {
  const page = frigateApp.page;
  const { marks } = await setup(frigateApp);
  await open(page, "a");
  const box = lightbox(page);
  // Show Reviewed off (default): the marked item leaves the list
  await box.getByRole("button", { name: "Secondlook right" }).click();
  await expect
    .poll(() => marks)
    .toEqual([{ id: "a", mark: "secondlook_right" }]);
  await expect(box.getByTestId("lightbox-position")).toHaveText(
    "1 of 2 loaded",
  );
  await expect(box.getByTestId("live-outcome")).toHaveText(
    "Suppressed: alert prevented",
  );
  await expect(
    box.getByRole("button", { name: "Primary right" }),
  ).toHaveAttribute("aria-pressed", "false");
  await page.screenshot({
    path: `${SCREENS}/lightbox-after-mark-${testInfo.project.name}.png`,
  });
  // last item marked: the previous one takes its place
  await page.keyboard.press("ArrowRight");
  await expect(box.getByTestId("lightbox-position")).toHaveText(
    "2 of 2 loaded",
  );
  await box.getByRole("button", { name: "Ignore" }).click();
  await expect(box.getByTestId("lightbox-position")).toHaveText(
    "1 of 1 loaded",
  );
  await expect(box.getByTestId("live-outcome")).toHaveText(
    "Suppressed: alert prevented",
  );
  await page.keyboard.press("Escape");
  await expect(lightbox(page)).toHaveCount(0);
});

test("with Show Reviewed on, marking keeps the same item open and shows the verdict", async ({
  frigateApp,
}) => {
  test.skip(frigateApp.isMobile, "Show Reviewed switch is in the desktop bar");
  const page = frigateApp.page;
  const { marks } = await setup(frigateApp);
  await page.getByRole("switch", { name: /show reviewed/i }).click();
  await open(page, "b");
  const box = lightbox(page);
  await box.getByRole("button", { name: "Primary right" }).click();
  await expect.poll(() => marks).toEqual([{ id: "b", mark: "primary_right" }]);
  await expect(
    box.getByRole("button", { name: "Primary right" }),
  ).toHaveAttribute("aria-pressed", "true");
  await expect(box.getByTestId("lightbox-position")).toHaveText(
    "2 of 3 loaded",
  );
  // clicking the selected verdict again clears it, still in place
  await box.getByRole("button", { name: "Primary right" }).click();
  await expect.poll(() => marks.at(-1)).toEqual({ id: "b", mark: null });
  await expect(
    box.getByRole("button", { name: "Primary right" }),
  ).toHaveAttribute("aria-pressed", "false");
});

test("mobile: swipe moves between items and pinch zooms @mobile", async ({
  frigateApp,
}) => {
  test.skip(!frigateApp.isMobile, "touch gestures");
  const page = frigateApp.page;
  await setup(frigateApp);
  await open(page, "a");
  const box = lightbox(page);
  const stage = await box.getByTestId("lightbox-stage").boundingBox();
  if (!stage) throw new Error("no stage");
  const cdp = await page.context().newCDPSession(page);
  const y = stage.y + stage.height / 2;
  const touch = async (type: string, points: { x: number; y: number }[]) =>
    cdp.send("Input.dispatchTouchEvent", {
      type,
      touchPoints: points.map((p, id) => ({ ...p, id })),
    });
  const swipe = async (from: number, to: number) => {
    await touch("touchStart", [{ x: from, y }]);
    for (let i = 1; i <= 8; i++)
      await touch("touchMove", [{ x: from + ((to - from) * i) / 8, y }]);
    await touch("touchEnd", []);
  };
  await swipe(stage.x + stage.width - 40, stage.x + 40);
  await expect(box.getByTestId("lightbox-position")).toHaveText(
    "2 of 3 loaded",
  );
  await swipe(stage.x + 40, stage.x + stage.width - 40);
  await expect(box.getByTestId("lightbox-position")).toHaveText(
    "1 of 3 loaded",
  );

  // two-finger pinch outwards scales the picture
  const cx = stage.x + stage.width / 2;
  await touch("touchStart", [
    { x: cx - 20, y },
    { x: cx + 20, y },
  ]);
  for (let i = 1; i <= 10; i++)
    await touch("touchMove", [
      { x: cx - 20 - i * 10, y },
      { x: cx + 20 + i * 10, y },
    ]);
  await touch("touchEnd", []);
  await expect
    .poll(() =>
      box
        .locator(".react-transform-component")
        .evaluate((el) => new DOMMatrix(getComputedStyle(el).transform).a),
    )
    .toBeGreaterThan(1.5);
  await page.screenshot({ path: `${SCREENS}/lightbox-pinch-mobile.png` });
  // a swipe while zoomed pans instead of changing the item
  await swipe(cx + 60, cx - 60);
  await expect(box.getByTestId("lightbox-position")).toHaveText(
    "1 of 3 loaded",
  );
});
