"""Release-port tests for the embedding IPC boundary, not full Frigate startup."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from frigate.embeddings.maintainer import EmbeddingMaintainer
from frigate.comms.event_metadata_updater import EventMetadataTypeEnum


class TestSecondlookReleaseIPC(unittest.TestCase):
    def test_saved_crop_and_misclass_evidence_survives_ipc(self):
        evidence = {
            "type": "secondlook", "object_id": "track",
            "sl_box": [1, 2, 30, 40], "sl_score": 0.2,
            "misclass_label": "car", "misclass_score": 0.9,
            "misclass_box": [2, 3, 31, 41], "primary_score": 0.8,
            "factor": 0.25, "frame_time": 1234.5, "model_id": "model-sha",
        }
        publisher = Mock()
        processor = Mock()
        processor.drain_results.return_value = [evidence]
        maintainer = SimpleNamespace(
            realtime_processors=[processor], event_metadata_publisher=publisher
        )
        EmbeddingMaintainer._process_deferred_results(maintainer)
        payload, topic = publisher.publish.call_args.args
        self.assertEqual(topic, EventMetadataTypeEnum.secondlook.value)
        self.assertEqual(payload[0], "track")
        for key, value in evidence.items():
            if key not in ("type", "object_id"):
                self.assertEqual(payload[1][key], value, key)


if __name__ == "__main__":
    unittest.main()
