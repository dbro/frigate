"""Corrections: outcome filter, long-term groups, bulk verdicts, mask suggestion."""

import json
import sqlite3
import tempfile
import time
import unittest

import numpy as np
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

from frigate.api.auth import get_allowed_cameras_for_filter
from frigate.api.save_and_review import router
from frigate.config import FrigateConfig
from frigate.config.classification import SecondlookSaveAndReviewConfig
from frigate.secondlook import clusters
from frigate.secondlook.save_and_review import _SCHEMA, configure_save_and_review
from frigate.test.test_save_and_review import make_config
from frigate.util.object import is_object_filtered

W, H = 640, 480  # make_config detect size for "cam"


def row(
    i,
    *,
    kind="false_positive",
    event_id=None,
    box=(100, 100, 140, 160),
    evidence=True,
    label="person",
    t=None,
    mark=None,
    camera="cam",
    misclass=None,
):
    ev = (
        {"version": 1, "width": W, "height": H, "primary_box": list(box)}
        if evidence
        else None
    )
    return {
        "id": f"r{i:06d}",
        "track_id": f"t{i}",
        "camera": camera,
        "kind": kind,
        "label": label,
        "misclass_label": misclass,
        "primary_score": 0.8,
        "sl_score": 0.1,
        "factor": 0.2,
        "box": json.dumps(list(box)) if box else None,
        "frame_time": 1_000_000.0 + i if t is None else t,
        "still_path": "/nonexistent.jpg",
        "event_id": event_id,
        "has_clip": 0,
        "zones": "[]",
        "mark": mark,
        "created_at": 1_000_000.0 + i,
        "evidence": json.dumps(ev) if ev else None,
        "group_key": "v2:test",
    }


def insert(db, rows):
    cols = list(rows[0])
    with sqlite3.connect(db) as conn:
        conn.executemany(
            f"INSERT INTO examples ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
            [tuple(r[c] for c in cols) for r in rows],
        )


class Base(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.config = make_config(True, self.temp.name)
        self.store = configure_save_and_review(self.config.secondlook.save_and_review)
        self.db = self.store.db_path
        app = FastAPI()
        app.frigate_config = self.config
        app.dependency_overrides[get_allowed_cameras_for_filter] = lambda: ["cam"]
        app.include_router(router)
        self.client = TestClient(app)

    def tearDown(self):
        self.client.close()
        configure_save_and_review(SecondlookSaveAndReviewConfig(enabled=False))
        self.temp.cleanup()

    def list(self, **params):
        r = self.client.get("/secondlook/save_and_review", params=params)
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()

    def groups(self, **params):
        r = self.client.get("/secondlook/save_and_review/groups", params=params)
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()


class TestOutcomeFilter(Base):
    def setUp(self):
        super().setUp()
        rows = []
        # 3 FN rescued, 2 FN not rescued, 4 FP suppressed, 1 FP not suppressed, 2 misclass
        for i in range(3):
            rows.append(row(len(rows), kind="false_negative", event_id=f"e{i}"))
        for _ in range(2):
            rows.append(row(len(rows), kind="false_negative"))
        for _ in range(4):
            rows.append(row(len(rows), kind="false_positive"))
        rows.append(row(len(rows), kind="false_positive", event_id="e9"))
        for i in range(2):
            rows.append(
                row(len(rows), kind="misclass", misclass="car", event_id=f"m{i}")
            )
        insert(self.db, rows)

    def test_outcome_per_kind_with_counts_and_paging(self):
        cases = [
            ("false_negative", "acted", 3),
            ("false_negative", "not_acted", 2),
            ("false_positive", "acted", 4),
            ("false_positive", "not_acted", 1),
            ("misclass", "acted", 0),
            (None, "acted", 7),
            (None, "not_acted", 3),
        ]
        for kind, outcome, n in cases:
            with self.subTest(kind=kind, outcome=outcome):
                params = {"outcome": outcome, **({"kind": kind} if kind else {})}
                body = self.list(**params)
                self.assertEqual(body["total"], n)
                self.assertEqual(body["unreviewed"], n)
                self.assertEqual(len(body["examples"]), n)
        page = self.list(kind="false_positive", outcome="acted", limit=3)
        self.assertEqual(
            (len(page["examples"]), page["total"], page["has_more"]), (3, 4, True)
        )
        rest = self.list(kind="false_positive", outcome="acted", limit=3, offset=3)
        self.assertEqual((len(rest["examples"]), rest["has_more"]), (1, False))
        self.assertTrue(
            all(e["event_id"] is None for e in page["examples"] + rest["examples"])
        )
        self.assertEqual(self.list(kind="false_negative")["total"], 5)
        self.assertEqual(
            self.client.get(
                "/secondlook/save_and_review", params={"outcome": "x"}
            ).status_code,
            400,
        )

    def test_old_rows_without_new_columns_work(self):
        """A DB from before evidence/group_key: outcome needs only event_id."""
        old = f"{self.temp.name}/old.db"
        with sqlite3.connect(old) as conn:
            conn.executescript(_SCHEMA.split("-- hourly")[0])
            conn.execute(
                "INSERT INTO examples (id, track_id, camera, kind, label, still_path, event_id, created_at, box) "
                "VALUES ('a', 't', 'cam', 'false_negative', 'person', '/x.jpg', 't', 1, '[1,2,30,40]'), "
                "('b', 'u', 'cam', 'false_negative', 'person', '/x.jpg', NULL, 2, NULL)"
            )
        from frigate.secondlook.save_and_review import SaveAndReviewStore

        store = SaveAndReviewStore(old, f"{self.temp.name}/img")
        self.assertEqual([r.id for r in store.list(outcome="acted")], ["a"])
        self.assertEqual([r.id for r in store.list(outcome="not_acted")], ["b"])
        groups = store.groups(frame_sizes={"cam": (W, H)})["groups"]
        self.assertEqual(
            sorted((g["cells"], g["count"]) for g in groups), [([], 1), ([[0, 0]], 1)]
        )


class TestLongTermGroups(Base):
    def test_bins_merge_touching_cells_and_split_the_rest(self):
        cw, ch = W / 8, H / 6
        rows = [
            # two centres either side of the column 2/3 edge, same size bucket
            row(0, box=(3 * cw - 30, 100, 3 * cw - 2, 140)),
            row(1, box=(3 * cw + 2, 100, 3 * cw + 30, 140)),
            # diagonal neighbour (touching corner) also merges
            row(2, box=(4 * cw + 2, ch + 2, 4 * cw + 30, ch + 42)),
            # far away cell: its own group
            row(3, box=(10, 400, 40, 440)),
            # same place, bigger size bucket: its own group
            row(4, box=(3 * cw - 100, 60, 3 * cw + 100, 200)),
            # same place, not suppressed: its own group
            row(5, box=(3 * cw - 30, 100, 3 * cw - 2, 140), event_id="x"),
            # no evidence: frame size from the camera's detect resolution
            row(6, box=(3 * cw + 4, 100, 3 * cw + 32, 140), evidence=False),
            # no box at all: location unknown
            row(7, box=None, evidence=False),
        ]
        insert(self.db, rows)
        body = self.groups()
        by_ids = {}
        for g in body["groups"]:
            members = self.list(group=g["key"], limit=50)
            self.assertEqual(members["total"], g["count"])
            by_ids[tuple(sorted(e["id"] for e in members["examples"]))] = g
        self.assertEqual(
            sorted(by_ids),
            sorted(
                [
                    ("r000000", "r000001", "r000002", "r000006"),
                    ("r000003",),
                    ("r000004",),
                    ("r000005",),
                    ("r000007",),
                ]
            ),
        )
        big = by_ids[("r000000", "r000001", "r000002", "r000006")]
        self.assertEqual(big["count"], 4)
        self.assertEqual(sorted(big["cells"]), [[2, 1], [3, 1], [4, 1]])
        self.assertEqual(big["size"], 0)
        self.assertEqual(big["outcome"], "acted")
        self.assertEqual(len(big["samples"]), 4)
        self.assertEqual(body["groups"][0]["key"], big["key"])  # sorted by count
        self.assertEqual(by_ids[("r000004",)]["size"], 3)
        self.assertEqual(by_ids[("r000007",)]["cells"], [])
        self.assertEqual(body["total_rows"], 8)

    def test_size_buckets_and_grid_edges(self):
        frame = W * H
        sizes = [0.005, 0.01, 0.02, 0.025, 0.049, 0.05, 0.09, 0.1, 0.5]
        rows = []
        for i, frac in enumerate(sizes):
            side = (frac * frame) ** 0.5
            rows.append(row(i, box=(0, 0, side, side), t=float(i)))
        insert(self.db, rows)
        groups = self.store.groups(frame_sizes={})["groups"]
        size_of = {}
        for g in groups:
            for r in self.store.list(group=json.loads(g["key"]), frame_sizes={}):
                size_of[r.id] = g["size"]
        self.assertEqual(
            [size_of[f"r{i:06d}"] for i in range(len(sizes))],
            [0, 1, 1, 2, 2, 3, 3, 4, 4],
        )

    def test_filters_combine_with_groups_and_bulk_verdict(self):
        rows = [
            row(i, kind="false_negative", event_id="e" if i % 2 else None)
            for i in range(10)
        ]
        rows += [
            row(10 + i, kind="false_positive", mark="primary_right" if i == 0 else None)
            for i in range(5)
        ]
        insert(self.db, rows)
        self.assertEqual(
            self.groups(kind="false_negative", outcome="acted")["total_rows"], 5
        )
        self.assertEqual(self.groups(kind="false_negative")["total_groups"], 2)
        self.assertEqual(self.groups(reviewed=False)["total_rows"], 14)
        self.assertEqual(self.groups(after=1_000_012)["total_rows"], 3)
        fp = self.groups(kind="false_positive")["groups"]
        self.assertEqual((len(fp), fp[0]["count"], fp[0]["unreviewed"]), (1, 5, 4))
        r = self.client.post(
            "/secondlook/save_and_review/groups/mark",
            params={"group": fp[0]["key"], "kind": "false_positive"},
            json={"mark": "ignore"},
        )
        self.assertEqual(r.json(), {"success": True, "updated": 4})
        marks = {
            e["id"]: e["mark"] for e in self.list(kind="false_positive")["examples"]
        }
        self.assertEqual(marks["r000010"], "primary_right")  # existing verdict kept
        self.assertEqual(sum(m == "ignore" for m in marks.values()), 4)
        self.assertEqual(self.list(kind="false_negative", reviewed=False)["total"], 10)
        bad = self.client.get(
            "/secondlook/save_and_review", params={"group": '{"camera":"cam"}'}
        )
        self.assertEqual(bad.status_code, 400)

    def test_fifty_thousand_rows(self):
        rng = np.random.default_rng(7)
        rows = []
        kinds = ["false_positive", "false_negative", "misclass"]
        for i in range(50_000):
            k = kinds[int(rng.integers(0, 3))]
            if rng.random() < 0.3:  # a recurring hot spot
                x, y, w, h = 300 + rng.normal(0, 8), 200 + rng.normal(0, 8), 40, 80
            else:
                x, y = rng.uniform(0, W - 60), rng.uniform(0, H - 60)
                w, h = rng.uniform(5, 200), rng.uniform(5, 200)
            box = (int(x), int(y), int(min(W, x + w)), int(min(H, y + h)))
            rows.append(
                row(
                    i,
                    kind=k,
                    box=box,
                    event_id="e" if rng.random() < 0.5 else None,
                    misclass="car" if k == "misclass" else None,
                    label=["person", "car", "dog"][int(rng.integers(0, 3))],
                    mark=None if rng.random() < 0.8 else "ignore",
                )
            )
        insert(self.db, rows)
        timings = {}
        for name, params in [
            ("all", {}),
            ("fp_acted", {"kind": "false_positive", "outcome": "acted"}),
            ("unreviewed_30d", {"reviewed": False, "after": 1_000_000.0 + 20_000}),
        ]:
            start = time.perf_counter()
            body = self.groups(**params, limit=200)
            timings[name] = time.perf_counter() - start
            expected = self.list(**params, limit=1)["total"]
            self.assertEqual(body["total_rows"], expected)
        top = self.groups(kind="false_positive", limit=1)["groups"][0]
        start = time.perf_counter()
        drill = self.list(group=top["key"], kind="false_positive", limit=50)
        timings["drill_in"] = time.perf_counter() - start
        self.assertEqual(drill["total"], top["count"])
        start = time.perf_counter()
        mask = self.client.get(
            "/secondlook/save_and_review/groups/mask",
            params={"group": top["key"], "kind": "false_positive"},
        ).json()
        timings["mask"] = time.perf_counter() - start
        self.assertEqual(mask["members"], top["count"])
        print(
            "\n50k rows timings (s):",
            {k: round(v, 3) for k, v in timings.items()},
            "groups:",
            self.groups()["total_groups"],
            "top FP group:",
            top["count"],
        )
        self.assertLess(max(timings.values()), 5.0)


class TestMaskSuggestion(Base):
    def test_yaml_loads_in_frigate_and_filters_every_member(self):
        boxes = [(300 + d, 200 + d, 340 + d, 280 + d) for d in (-6, 0, 5, 9)]
        rows = [row(i, box=b) for i, b in enumerate(boxes)]
        rows.append(row(10, box=(600, 400, 640, 480)))  # far away, other group
        insert(self.db, rows)
        group = next(g for g in self.groups()["groups"] if g["count"] == 4)
        for points in ("boxes", "bottom"):
            with self.subTest(points=points):
                body = self.client.get(
                    "/secondlook/save_and_review/groups/mask",
                    params={"group": group["key"], "points": points},
                ).json()
                self.assertEqual(
                    (body["members"], body["bottom_centres_inside"]), (4, 4)
                )
                snippet = yaml.safe_load(body["yaml"])
                mask = snippet["cameras"]["cam"]["objects"]["filters"]["person"]["mask"]
                self.assertEqual(list(mask), [body["name"]])
                self.assertEqual(mask[body["name"]]["coordinates"], body["coordinates"])
                raw = make_config(True, self.temp.name).model_dump(exclude_unset=True)
                raw["cameras"]["cam"]["objects"] = snippet["cameras"]["cam"]["objects"]
                config = FrigateConfig(**raw)
                filters = config.cameras["cam"].objects.filters
                self.assertIsNotNone(filters["person"].rasterized_mask)
                for b in boxes:
                    obj = (
                        "person",
                        0.9,
                        list(b),
                        (b[2] - b[0]) * (b[3] - b[1]),
                        0.5,
                        None,
                    )
                    self.assertTrue(is_object_filtered(obj, ["person"], filters), b)
                far = ("person", 0.9, [600, 400, 640, 480], 3200, 0.5, None)
                self.assertFalse(is_object_filtered(far, ["person"], filters))
                if points == "boxes":
                    for x1, y1, x2, y2 in boxes:  # union covered: every corner inside
                        for x, y in ((x1, y1), (x2, y1), (x1, y2), (x2, y2)):
                            self.assertTrue(
                                clusters.point_in_polygon(x / W, y / H, body["polygon"])
                            )

    def test_only_located_false_positive_groups(self):
        insert(
            self.db, [row(0, kind="false_negative"), row(1, box=None, evidence=False)]
        )
        for g in self.groups()["groups"]:
            r = self.client.get(
                "/secondlook/save_and_review/groups/mask", params={"group": g["key"]}
            )
            self.assertEqual(r.status_code, 400)

    def test_coordinate_format_avoids_frigate_pixel_detection(self):
        self.assertEqual(clusters.format_coordinate(1.0), "1.0")
        self.assertEqual(clusters.format_coordinate(0.99999), "1.0")
        self.assertEqual(clusters.format_coordinate(1.2), "1.0")
        self.assertEqual(clusters.format_coordinate(-0.1), "0.000")
        self.assertEqual(clusters.format_coordinate(0.5), "0.500")
        poly = clusters.mask_polygon([(0.95, 0.9, 1.0, 1.0)], margin=0.05)
        coords = ",".join(
            f"{clusters.format_coordinate(x)},{clusters.format_coordinate(y)}"
            for x, y in poly
        )
        self.assertFalse(any(v > "1.0" for v in coords.split(",")), coords)
        # why: 0.18.2 reads "1.000" as a pixel value and cannot parse it
        raw = make_config(True, self.temp.name).model_dump(exclude_unset=True)
        raw["cameras"]["cam"]["objects"] = {
            "filters": {
                "person": {
                    "mask": {"m": {"coordinates": "0.5,0.5,1.000,0.5,1.000,1.000"}}
                }
            }
        }
        with self.assertRaises(ValueError):
            FrigateConfig(**raw)


if __name__ == "__main__":
    unittest.main()
