"""Save-and-review config, persistence, and enable/disable boundaries."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from frigate.api.auth import get_allowed_cameras_for_filter
from frigate.api.save_and_review import router
from frigate.config import FrigateConfig
from frigate.config.classification import (
    SecondlookConfig,
    SecondlookSaveAndReviewConfig,
)
from frigate.secondlook.save_and_review import (
    configure_save_and_review,
    get_save_and_review_store,
)
from frigate.secondlook.save_and_review_deposit import save_track_for_review
from frigate.track.tracked_object import TrackedObject


def make_config(enabled: bool, directory: str, live: bool = True) -> FrigateConfig:
    return FrigateConfig(
        mqtt={"enabled": False},
        secondlook={
            "enabled": live,
            "save_and_review": {
                "enabled": enabled,
                "database": {"path": f"{directory}/saved/secondlook_examples.db"},
                "images": {"path": f"{directory}/saved/images"},
            },
        },
        cameras={
            "cam": {
                "ffmpeg": {"inputs": [{"path": "/dev/null", "roles": ["detect"]}]},
                "detect": {"width": 640, "height": 480, "fps": 5},
                "secondlook": {"enabled": live},
            }
        },
    )


def make_track(config: FrigateConfig, score: float = 0.9, factor: float = 0.0):
    data = {
        "id": "1000.0-abcdef",
        "label": "person",
        "frame_time": 1000.0,
        "start_time": 1000.0,
        "score": score,
        "box": [100, 100, 200, 300],
        "area": 20000,
        "ratio": 0.5,
        "region": [0, 0, 640, 480],
        "centroid": [150, 200],
        "position_changes": 1,
        "motionless_count": 0,
        "attributes": [],
    }
    obj = TrackedObject(
        config.model,
        config.cameras["cam"],
        config.ui,
        {},
        {**data, "score_history": [score] * 10},
    )
    obj.apply_secondlook(
        {"factor": factor, "primary_score": score, "sl_score": score * factor}
    )
    data["frame_time"] = 1000.2
    obj.update(1000.2, data, False)
    obj.capture_save_and_review_frame(np.zeros((720, 640), dtype=np.uint8))
    return obj


class TestSaveAndReviewConfig(unittest.TestCase):
    def test_nested_defaults_preserve_disk_locations(self):
        config = SecondlookConfig()
        self.assertTrue(config.save_and_review.enabled)
        self.assertEqual(
            config.save_and_review.database.path, "/config/secondlook_examples.db"
        )
        self.assertEqual(
            config.save_and_review.images.path,
            "/media/frigate/clips/secondlook_examples",
        )
        self.assertFalse(config.enabled)

    def test_old_unpublished_config_keys_are_rejected(self):
        for key, value in (
            ("examples_enabled", False),
            ("examples_db", "a"),
            ("examples_dir", "b"),
        ):
            with self.subTest(key=key), self.assertRaises(ValidationError):
                SecondlookConfig(**{key: value})

    def test_disabled_initialization_does_not_create_paths(self):
        with tempfile.TemporaryDirectory() as td:
            config = make_config(False, td)
            self.assertIsNone(
                configure_save_and_review(config.secondlook.save_and_review)
            )
            self.assertIsNone(get_save_and_review_store())
            self.assertEqual(list(Path(td).iterdir()), [])

    def test_live_qa_available_without_saving(self):
        with tempfile.TemporaryDirectory() as td:
            config = make_config(False, td)
            configure_save_and_review(config.secondlook.save_and_review)
            suppressed = make_track(config)
            promoted = make_track(config, score=0.5, factor=2.0)
            self.assertTrue(suppressed.false_positive)
            self.assertFalse(promoted.false_positive)
            self.assertEqual(suppressed.secondlook_factor, 0.0)
            self.assertIsNone(suppressed.save_and_review_jpeg)
            self.assertFalse(suppressed.save_and_review_needs_frame)
            self.assertIsNone(save_track_for_review(suppressed, config.secondlook))
            self.assertEqual(list(Path(td).iterdir()), [])

    def test_enabled_saves_and_disabled_preserves_existing_records(self):
        with tempfile.TemporaryDirectory() as td:
            config = make_config(True, td)
            store = configure_save_and_review(config.secondlook.save_and_review)
            obj = make_track(config)
            example_id = save_track_for_review(obj, config.secondlook)
            self.assertIsNotNone(example_id)
            record = store.set_mark(example_id, "secondlook_right")
            jpeg = Path(record.still_path).read_bytes()
            config.secondlook.save_and_review.enabled = False
            self.assertIsNone(save_track_for_review(obj, config.secondlook))
            self.assertEqual(len(store.list()), 1)
            configure_save_and_review(config.secondlook.save_and_review)
            config.secondlook.save_and_review.enabled = True
            reopened = configure_save_and_review(config.secondlook.save_and_review)
            self.assertEqual(reopened.get(example_id).mark, "secondlook_right")
            self.assertEqual(Path(record.still_path).read_bytes(), jpeg)

    def test_review_existing_data_does_not_require_live_adjustment(self):
        with tempfile.TemporaryDirectory() as td:
            config = make_config(True, td, live=False)
            self.assertIsNotNone(
                configure_save_and_review(config.secondlook.save_and_review)
            )
            self.assertFalse(config.secondlook.enabled)


class TestSaveAndReviewAPI(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.config = make_config(True, self.temp.name)
        self.store = configure_save_and_review(self.config.secondlook.save_and_review)
        self.example_id = save_track_for_review(
            make_track(self.config), self.config.secondlook
        )
        self.app = FastAPI()
        self.app.frigate_config = self.config
        self.app.dependency_overrides[get_allowed_cameras_for_filter] = lambda: ["cam"]
        self.app.include_router(router)
        self.client = TestClient(self.app)

    def tearDown(self):
        self.client.close()
        configure_save_and_review(SecondlookSaveAndReviewConfig(enabled=False))
        self.temp.cleanup()

    def test_enabled_list_still_and_marks(self):
        base = f"/secondlook/save_and_review/{self.example_id}"
        self.assertEqual(
            self.client.get("/secondlook/save_and_review").json()["examples"][0]["id"],
            self.example_id,
        )
        self.assertEqual(self.client.get(base).status_code, 200)
        self.assertEqual(
            self.client.get(f"{base}/still.jpg").headers["content-type"], "image/jpeg"
        )
        self.assertEqual(
            self.client.post(f"{base}/mark", json={"mark": "secondlook_right"}).status_code, 200
        )
        self.assertEqual(self.store.get(self.example_id).mark, "secondlook_right")

    def test_disabled_blocks_all_access_even_with_existing_store(self):
        self.config.secondlook.save_and_review.enabled = False
        base = f"/secondlook/save_and_review/{self.example_id}"
        for path in ("/secondlook/save_and_review", base, f"{base}/still.jpg"):
            with self.subTest(path=path):
                self.assertEqual(self.client.get(path).status_code, 404)
        self.assertEqual(
            self.client.post(f"{base}/mark", json={"mark": "secondlook_right"}).status_code, 404
        )
        self.assertIsNone(self.store.get(self.example_id).mark)
        self.assertEqual(len(self.store.list()), 1)
