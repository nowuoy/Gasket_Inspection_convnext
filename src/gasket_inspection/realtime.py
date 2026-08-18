from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import resolve_project_path
from .predictor import Predictor


class InputCollisionError(RuntimeError):
    pass


class SingleInstanceLock:
    """동일 state DB에 watcher가 둘 이상 붙는 것을 막습니다."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("a+b")
        if path.stat().st_size == 0:
            self.handle.write(b"0")
            self.handle.flush()
        self.handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.handle.close()
            raise RuntimeError(
                f"같은 state DB를 사용하는 watcher가 이미 실행 중입니다: {path}"
            ) from exc

    def close(self) -> None:
        if self.handle.closed:
            return
        self.handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        finally:
            self.handle.close()


@dataclass(frozen=True)
class ReadyItem:
    sample_id: str
    front_path: Path | None = None
    side_path: Path | None = None
    combined_path: Path | None = None

    def input_paths(self) -> list[tuple[str, Path]]:
        if self.combined_path is not None:
            return [("combined", self.combined_path)]
        if self.front_path is None or self.side_path is None:
            raise ValueError(f"{self.sample_id}: 두 view가 완성되지 않았습니다.")
        return [("front", self.front_path), ("side", self.side_path)]


class StableFileTracker:
    def __init__(self, required_checks: int) -> None:
        if required_checks < 1:
            raise ValueError("stable_checks는 1 이상이어야 합니다.")
        self.required_checks = required_checks
        self._state: dict[Path, tuple[int, int, int]] = {}

    def is_stable(self, path: Path) -> bool:
        try:
            stat = path.stat()
        except FileNotFoundError:
            self._state.pop(path, None)
            return False
        if stat.st_size <= 0:
            return False
        signature = (stat.st_size, stat.st_mtime_ns)
        previous = self._state.get(path)
        count = previous[2] + 1 if previous and previous[:2] == signature else 1
        self._state[path] = (signature[0], signature[1], count)
        return count >= self.required_checks


class InspectionStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS inspections (
                sample_id TEXT PRIMARY KEY,
                fingerprint TEXT NOT NULL,
                processing_state TEXT NOT NULL,
                decision_status TEXT,
                result_path TEXT,
                error_message TEXT,
                updated_at TEXT NOT NULL
            )
            """
        )
        # 비정상 종료 중 RUNNING이었던 항목은 재시작 시 다시 처리할 수 있게 합니다.
        self.connection.execute(
            "UPDATE inspections SET processing_state='PENDING' WHERE processing_state='RUNNING'"
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def should_process(self, sample_id: str, fingerprint: str, retry_errors: bool) -> bool:
        row = self.connection.execute(
            "SELECT fingerprint, processing_state FROM inspections WHERE sample_id=?",
            (sample_id,),
        ).fetchone()
        if row is None:
            return True
        previous_fingerprint, state = row
        if previous_fingerprint != fingerprint:
            raise InputCollisionError(
                f"동일 sample_id에 다른 이미지가 들어왔습니다: {sample_id}. 새 ID로 저장하세요."
            )
        if state == "DONE":
            return False
        if state == "ERROR" and not retry_errors:
            return False
        return True

    def claim(self, sample_id: str, fingerprint: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.connection.execute(
            """
            INSERT INTO inspections(sample_id, fingerprint, processing_state, updated_at)
            VALUES (?, ?, 'RUNNING', ?)
            ON CONFLICT(sample_id) DO UPDATE SET
              processing_state='RUNNING', error_message=NULL, updated_at=excluded.updated_at
            """,
            (sample_id, fingerprint, now),
        )
        self.connection.commit()

    def finish(self, sample_id: str, decision_status: str, result_path: Path) -> None:
        self.connection.execute(
            """
            UPDATE inspections
            SET processing_state='DONE', decision_status=?, result_path=?, updated_at=?
            WHERE sample_id=?
            """,
            (decision_status, str(result_path), datetime.now(timezone.utc).isoformat(), sample_id),
        )
        self.connection.commit()

    def fail(self, sample_id: str, message: str, result_path: Path | None) -> None:
        self.connection.execute(
            """
            UPDATE inspections
            SET processing_state='ERROR', decision_status='ERROR', result_path=?,
                error_message=?, updated_at=?
            WHERE sample_id=?
            """,
            (
                str(result_path) if result_path is not None else None,
                message,
                datetime.now(timezone.utc).isoformat(),
                sample_id,
            ),
        )
        self.connection.commit()


class AtomicResultWriter:
    def __init__(self, results_dir: Path) -> None:
        self.results_dir = results_dir
        results_dir.mkdir(parents=True, exist_ok=True)

    def write(self, sample_id: str, payload: dict[str, Any]) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", sample_id):
            raise ValueError(f"안전하지 않은 sample_id입니다: {sample_id}")
        final_path = self.results_dir / f"{sample_id}.json"
        temporary_path = self.results_dir / f".{sample_id}.json.tmp"
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, final_path)
        return final_path


class FolderMonitor:
    def __init__(
        self,
        cfg: dict[str, Any],
        predictor: Predictor,
        *,
        instance_lock: SingleInstanceLock | None = None,
    ) -> None:
        realtime_cfg = cfg["realtime"]
        self.cfg = cfg
        self.predictor = predictor
        self.input_mode = cfg["input"]["mode"]
        self.inbox = resolve_project_path(cfg, realtime_cfg["inbox_dir"])
        self.inbox.mkdir(parents=True, exist_ok=True)
        self.poll_interval = float(realtime_cfg.get("poll_interval_s", 0.2))
        self.pair_timeout = float(realtime_cfg.get("pair_timeout_s", 10.0))
        self.retry_errors = bool(realtime_cfg.get("retry_errors", False))
        self.retry_backoff = float(realtime_cfg.get("retry_backoff_s", 5.0))
        self.max_retry_attempts = int(realtime_cfg.get("max_retry_attempts", 3))
        regex_key = (
            "paired_filename_regex" if self.input_mode == "paired_files" else "combined_filename_regex"
        )
        self.filename_pattern = re.compile(realtime_cfg[regex_key], flags=re.IGNORECASE)
        self.stability = StableFileTracker(int(realtime_cfg.get("stable_checks", 3)))
        state_db = resolve_project_path(cfg, realtime_cfg["state_db"])
        self.instance_lock = instance_lock or SingleInstanceLock(
            state_db.with_suffix(state_db.suffix + ".lock")
        )
        self.store = InspectionStore(state_db)
        self.writer = AtomicResultWriter(resolve_project_path(cfg, realtime_cfg["results_dir"]))
        self.first_seen: dict[str, float] = {}
        self.timeout_warned: set[str] = set()
        self.handled_signatures: dict[
            str, tuple[tuple[str, str, int, int, int, int], ...]
        ] = {}
        self.retry_after: dict[str, float] = {}
        self.retry_attempts: dict[str, int] = {}
        self.input_retry_attempts: dict[str, int] = {}

    def close(self) -> None:
        self.store.close()
        self.instance_lock.close()

    def _discover(self) -> list[ReadyItem]:
        if self.input_mode == "combined_image":
            candidates: dict[str, list[Path]] = {}
            for path in sorted(self.inbox.iterdir()):
                if not path.is_file():
                    continue
                match = self.filename_pattern.fullmatch(path.name)
                if match:
                    candidates.setdefault(match.group("id").upper(), []).append(path)
            ready: list[ReadyItem] = []
            for sample_id, paths in candidates.items():
                if len(paths) != 1:
                    print(
                        f"[ERROR] {sample_id}: combined 파일이 둘 이상이므로 처리하지 않습니다: {paths}",
                        file=sys.stderr,
                    )
                    continue
                path = paths[0]
                if self.stability.is_stable(path):
                    ready.append(ReadyItem(sample_id=sample_id, combined_path=path))
            return ready

        pairs: dict[str, dict[str, Path]] = {}
        invalid_ids: set[str] = set()
        for path in sorted(self.inbox.iterdir()):
            if not path.is_file():
                continue
            match = self.filename_pattern.fullmatch(path.name)
            if not match:
                continue
            sample_id = match.group("id").upper()
            view = match.group("view").lower()
            current = pairs.setdefault(sample_id, {})
            if view in current and current[view] != path:
                print(f"[ERROR] {sample_id}: {view} 파일이 둘 이상입니다.", file=sys.stderr)
                invalid_ids.add(sample_id)
                continue
            current[view] = path
            self.first_seen.setdefault(sample_id, time.monotonic())

        ready = []
        now = time.monotonic()
        for sample_id, views in pairs.items():
            if sample_id in invalid_ids:
                continue
            if {"front", "side"}.issubset(views):
                front_stable = self.stability.is_stable(views["front"])
                side_stable = self.stability.is_stable(views["side"])
                if front_stable and side_stable:
                    ready.append(
                        ReadyItem(
                            sample_id=sample_id,
                            front_path=views["front"],
                            side_path=views["side"],
                        )
                    )
                continue
            if now - self.first_seen[sample_id] >= self.pair_timeout and sample_id not in self.timeout_warned:
                missing = "side" if "front" in views else "front"
                print(
                    f"[WARN] {sample_id}: {missing} 이미지가 {self.pair_timeout:.1f}초 안에 오지 않았습니다. 대기합니다.",
                    file=sys.stderr,
                )
                self.timeout_warned.add(sample_id)
        return ready

    @staticmethod
    def _stat_signature(
        item: ReadyItem,
    ) -> tuple[tuple[str, str, int, int, int, int], ...]:
        signature = []
        for view, path in item.input_paths():
            stat = path.stat()
            signature.append(
                (view, str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino)
            )
        return tuple(signature)

    @staticmethod
    def _input_metadata(
        item: ReadyItem,
    ) -> tuple[list[dict[str, Any]], str, dict[str, bytes]]:
        records: list[dict[str, Any]] = []
        snapshots: dict[str, bytes] = {}
        fingerprint_builder = hashlib.sha256()
        for view, path in item.input_paths():
            before = path.stat()
            file_bytes = path.read_bytes()
            file_hash = hashlib.sha256(file_bytes).hexdigest()
            stat = path.stat()
            if (before.st_size, before.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
                raise OSError(f"hash 계산 중 파일이 변경되었습니다: {path}")
            records.append(
                {
                    "view": view,
                    "path": str(path),
                    "sha256": file_hash,
                    "size_bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
            fingerprint_builder.update(view.encode("utf-8"))
            fingerprint_builder.update(file_hash.encode("ascii"))
            snapshots[view] = file_bytes
        return records, fingerprint_builder.hexdigest(), snapshots

    def _schedule_retry(
        self,
        sample_id: str,
        stat_signature: tuple[tuple[str, str, int, int, int, int], ...],
    ) -> None:
        attempt = self.retry_attempts.get(sample_id, 0) + 1
        self.retry_attempts[sample_id] = attempt
        if self.retry_errors and attempt < self.max_retry_attempts:
            delay = self.retry_backoff * (2 ** (attempt - 1))
            self.retry_after[sample_id] = time.monotonic() + delay
            print(
                f"[WARN] {sample_id}: {delay:.1f}초 후 재시도 "
                f"({attempt}/{self.max_retry_attempts})",
                file=sys.stderr,
            )
            return
        self.handled_signatures[sample_id] = stat_signature

    def _schedule_input_retry(self, sample_id: str) -> None:
        attempt = min(self.input_retry_attempts.get(sample_id, 0) + 1, self.max_retry_attempts)
        self.input_retry_attempts[sample_id] = attempt
        delay = self.retry_backoff * (2 ** (attempt - 1))
        self.retry_after[sample_id] = time.monotonic() + delay

    def _process(self, item: ReadyItem) -> bool:
        item_start = time.perf_counter()
        if time.monotonic() < self.retry_after.get(item.sample_id, 0.0):
            return False
        try:
            stat_signature = self._stat_signature(item)
            if self.handled_signatures.get(item.sample_id) == stat_signature:
                self.first_seen.pop(item.sample_id, None)
                return False
            input_records, fingerprint, snapshots = self._input_metadata(item)
            if not self.store.should_process(item.sample_id, fingerprint, self.retry_errors):
                self.handled_signatures[item.sample_id] = stat_signature
                self.first_seen.pop(item.sample_id, None)
                return False
            self.store.claim(item.sample_id, fingerprint)
            self.input_retry_attempts.pop(item.sample_id, None)
            self.retry_after.pop(item.sample_id, None)
        except InputCollisionError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            self.handled_signatures[item.sample_id] = stat_signature
            return False
        except (FileNotFoundError, OSError) as exc:
            # 저장 프로그램의 rename과 scan이 겹친 경우 다음 scan에서 재발견합니다.
            self._schedule_input_retry(item.sample_id)
            print(
                f"[WARN] {item.sample_id}: 입력 파일 상태 확인 실패, backoff 후 재시도합니다: {exc}",
                file=sys.stderr,
            )
            return False
        except sqlite3.Error as exc:
            print(f"[ERROR] {item.sample_id}: 상태 DB 접근 실패, 재시도합니다: {exc}", file=sys.stderr)
            return False

        try:
            result = self.predictor.predict_bytes(
                item.sample_id,
                front_bytes=snapshots.get("front"),
                side_bytes=snapshots.get("side"),
                combined_bytes=snapshots.get("combined"),
            )
            result["inputs"] = input_records
            result.setdefault("latency_ms", {})["watcher_to_prediction"] = (
                time.perf_counter() - item_start
            ) * 1000.0
        except Exception as exc:
            error_result = {
                "schema_version": 1,
                "sample_id": item.sample_id,
                "processed_at": datetime.now(timezone.utc).isoformat(),
                "inputs": input_records,
                "decision": {"status": "ERROR"},
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
            error_path: Path | None = None
            try:
                error_path = self.writer.write(item.sample_id, error_result)
            except Exception as write_exc:
                print(
                    f"[ERROR] {item.sample_id}: 오류 결과 JSON 저장 실패: {write_exc}",
                    file=sys.stderr,
                )
            try:
                self.store.fail(item.sample_id, str(exc), error_path)
            except Exception as store_exc:
                print(
                    f"[ERROR] {item.sample_id}: 오류 상태 DB 저장 실패: {store_exc}",
                    file=sys.stderr,
                )
            try:
                print(
                    json.dumps(error_result, ensure_ascii=False, allow_nan=False),
                    file=sys.stderr,
                    flush=True,
                )
            except (BrokenPipeError, OSError, UnicodeError):
                pass
            self._schedule_retry(item.sample_id, stat_signature)
            self.first_seen.pop(item.sample_id, None)
            self.timeout_warned.discard(item.sample_id)
            return False

        try:
            result_path = self.writer.write(item.sample_id, result)
        except Exception as exc:
            print(f"[ERROR] {item.sample_id}: 정상 결과 JSON 저장 실패: {exc}", file=sys.stderr)
            try:
                self.store.fail(item.sample_id, f"result write failed: {exc}", None)
            except Exception as store_exc:
                print(f"[ERROR] {item.sample_id}: DB 오류 상태 저장 실패: {store_exc}", file=sys.stderr)
            self._schedule_retry(item.sample_id, stat_signature)
            self.first_seen.pop(item.sample_id, None)
            return False

        db_finished = False
        for attempt in range(1, self.max_retry_attempts + 1):
            try:
                self.store.finish(item.sample_id, result["decision"]["status"], result_path)
                db_finished = True
                break
            except sqlite3.Error as exc:
                if attempt >= self.max_retry_attempts:
                    print(
                        f"[ERROR] {item.sample_id}: 정상 JSON은 저장됐지만 DB 확정 실패: {exc}",
                        file=sys.stderr,
                    )
                    break
                delay = self.retry_backoff * (2 ** (attempt - 1))
                print(
                    f"[WARN] {item.sample_id}: DB 확정을 {delay:.1f}초 후 재시도 "
                    f"({attempt}/{self.max_retry_attempts})",
                    file=sys.stderr,
                )
                time.sleep(delay)

        if not db_finished:
            # 정상 JSON은 이미 원자적으로 저장되었습니다. 재추론/ERROR 덮어쓰기를 막습니다.
            self.handled_signatures[item.sample_id] = stat_signature
            self.first_seen.pop(item.sample_id, None)
            self.timeout_warned.discard(item.sample_id)
            return True

        try:
            print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
        except (BrokenPipeError, OSError, UnicodeError) as exc:
            # stdout 소비자 오류는 이미 확정된 검사 결과를 바꾸지 않습니다.
            print(f"[WARN] {item.sample_id}: stdout 출력 실패: {exc}", file=sys.stderr)
        self.handled_signatures[item.sample_id] = stat_signature
        self.retry_after.pop(item.sample_id, None)
        self.retry_attempts.pop(item.sample_id, None)
        self.input_retry_attempts.pop(item.sample_id, None)
        self.first_seen.pop(item.sample_id, None)
        self.timeout_warned.discard(item.sample_id)
        return True

    def scan_once(self) -> int:
        processed = 0
        for item in self._discover():
            processed += int(self._process(item))
        return processed

    def run_forever(self) -> None:
        print(f"감시 시작: {self.inbox} (mode={self.input_mode})")
        while True:
            try:
                self.scan_once()
            except Exception as exc:
                # 일시적인 폴더/DB 오류 한 번으로 장시간 watcher가 종료되지 않게 합니다.
                print(f"[ERROR] scan 실패, 다음 주기에 재시도합니다: {exc}", file=sys.stderr)
            time.sleep(self.poll_interval)

    def run_current_files(self) -> int:
        # 안정성 검사 횟수를 충족할 만큼 현재 폴더를 반복 조회합니다.
        total = 0
        checks = int(self.cfg["realtime"].get("stable_checks", 3)) + 1
        for _ in range(checks):
            total += self.scan_once()
            time.sleep(self.poll_interval)
        return total
