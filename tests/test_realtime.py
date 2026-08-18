from __future__ import annotations

from PIL import Image

from gasket_inspection.realtime import FolderMonitor, StableFileTracker


def test_file_must_be_unchanged_for_required_checks(tmp_path) -> None:
    path = tmp_path / "P001__front.jpg"
    path.write_bytes(b"first")
    tracker = StableFileTracker(required_checks=3)
    assert tracker.is_stable(path) is False
    assert tracker.is_stable(path) is False
    assert tracker.is_stable(path) is True

    path.write_bytes(b"changed-and-larger")
    assert tracker.is_stable(path) is False


class DummyPredictor:
    def predict_bytes(self, sample_id, **images):
        return {
            "schema_version": 1,
            "sample_id": sample_id,
            "decision": {"status": "OK"},
            "latency_ms": {"inference": 1.0},
        }


def monitor_config(tmp_path, stable_checks=3):
    return {
        "_project_root": str(tmp_path),
        "input": {"mode": "paired_files"},
        "realtime": {
            "inbox_dir": "inbox",
            "results_dir": "results",
            "state_db": "state/inspection.sqlite3",
            "poll_interval_s": 0.001,
            "stable_checks": stable_checks,
            "pair_timeout_s": 1.0,
            "retry_errors": False,
            "retry_backoff_s": 0.01,
            "max_retry_attempts": 2,
            "paired_filename_regex": (
                r"^(?P<id>[A-Za-z0-9_-]+)__(?P<view>front|side)\."
                r"(?P<ext>jpg|jpeg|png)$"
            ),
            "combined_filename_regex": (
                r"^(?P<id>[A-Za-z0-9_-]+)__combined\.(?P<ext>jpg|jpeg|png)$"
            ),
        },
    }


def test_run_current_files_processes_stable_pair(tmp_path) -> None:
    cfg = monitor_config(tmp_path)
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    Image.new("RGB", (16, 16), "yellow").save(inbox / "P001__front.jpg")
    Image.new("RGB", (16, 16), "yellow").save(inbox / "P001__side.jpg")
    monitor = FolderMonitor(cfg, DummyPredictor())
    try:
        assert monitor.run_current_files() == 1
        assert (tmp_path / "results" / "P001.json").is_file()
    finally:
        monitor.close()


def test_duplicate_view_is_not_processed(tmp_path) -> None:
    cfg = monitor_config(tmp_path, stable_checks=1)
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    image = Image.new("RGB", (16, 16), "yellow")
    image.save(inbox / "P002__front.jpg")
    image.save(inbox / "P002__front.png")
    image.save(inbox / "P002__side.jpg")
    monitor = FolderMonitor(cfg, DummyPredictor())
    try:
        assert monitor.scan_once() == 0
        assert not (tmp_path / "results" / "P002.json").exists()
    finally:
        monitor.close()
