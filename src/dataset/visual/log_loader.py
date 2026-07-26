from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set


LOG_FILE_RE = re.compile(r"(?P<log_type>.+)_rank(?P<rank>\d+)\.jsonl$")


@dataclass
class LogRecord:
    log_type: str
    rank: int
    ts: Optional[str]
    ts_unix: Optional[int]
    line_no: int
    seq: int
    payload: Dict[str, Any]


@dataclass
class LogBundle:
    log_dir: Path
    by_type: Dict[str, List[LogRecord]]

    @property
    def available_types(self) -> List[str]:
        return sorted(self.by_type.keys())

    @property
    def available_ranks(self) -> List[int]:
        ranks: Set[int] = set()
        for records in self.by_type.values():
            for record in records:
                ranks.add(record.rank)
        return sorted(ranks)

    def records(self, log_type: str, ranks: Optional[Set[int]] = None) -> List[LogRecord]:
        records = self.by_type.get(log_type, [])
        if not ranks:
            return records
        return [record for record in records if record.rank in ranks]


def _parse_ts(ts: Any) -> Optional[int]:
    if not isinstance(ts, str) or not ts:
        return None
    try:
        dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        return int(dt.timestamp())
    except Exception:
        return None


def _extract_payload(raw_obj: Dict[str, Any]) -> Dict[str, Any]:
    payload = raw_obj.get("payload")
    if isinstance(payload, dict):
        return payload
    return {k: v for k, v in raw_obj.items() if k not in {"ts", "rank", "pid", "payload"}}


def _record_sort_key(record: LogRecord) -> tuple:
    ts = record.ts_unix
    if ts is None:
        ts = 10**12
    return (ts, record.rank, record.seq)


def load_log_bundle(log_dir: Path) -> LogBundle:
    if not log_dir.exists():
        raise FileNotFoundError(f"Log directory does not exist: {log_dir}")
    if not log_dir.is_dir():
        raise NotADirectoryError(f"Log path is not a directory: {log_dir}")

    by_type: Dict[str, List[LogRecord]] = {}
    seq_by_type: Dict[str, int] = {}

    for path in sorted(log_dir.glob("*_rank*.jsonl")):
        match = LOG_FILE_RE.match(path.name)
        if not match:
            continue

        log_type = match.group("log_type")
        rank = int(match.group("rank"))
        records = by_type.setdefault(log_type, [])
        seq = seq_by_type.setdefault(log_type, 0)

        with path.open("r", encoding="utf-8") as fp:
            for line_no, line in enumerate(fp, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw_obj = json.loads(line)
                except Exception:
                    continue
                if not isinstance(raw_obj, dict):
                    continue
                payload = _extract_payload(raw_obj)
                if not isinstance(payload, dict):
                    continue
                ts = raw_obj.get("ts")
                record = LogRecord(
                    log_type=log_type,
                    rank=rank,
                    ts=ts if isinstance(ts, str) else None,
                    ts_unix=_parse_ts(ts),
                    line_no=line_no,
                    seq=seq,
                    payload=payload,
                )
                records.append(record)
                seq += 1

        seq_by_type[log_type] = seq

    for records in by_type.values():
        records.sort(key=_record_sort_key)

    return LogBundle(log_dir=log_dir, by_type=by_type)


def build_dir_signature(log_dir: Path) -> str:
    """
    Build a lightweight signature for cache invalidation.
    Signature format: file_count:total_size:max_mtime_ns
    """
    file_count = 0
    total_size = 0
    max_mtime_ns = 0
    for path in sorted(log_dir.glob("*_rank*.jsonl")):
        if not path.is_file():
            continue
        stat = path.stat()
        file_count += 1
        total_size += int(stat.st_size)
        max_mtime_ns = max(max_mtime_ns, int(stat.st_mtime_ns))
    return f"{file_count}:{total_size}:{max_mtime_ns}"


def parse_rank_query(raw: str, available_ranks: Iterable[int]) -> Optional[Set[int]]:
    text = (raw or "all").strip().lower()
    if not text or text == "all":
        return None

    available = set(int(rank) for rank in available_ranks)
    wanted: Set[int] = set()
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            a, b = token.split("-", 1)
            try:
                start = int(a)
                end = int(b)
            except Exception:
                continue
            lo, hi = min(start, end), max(start, end)
            for rank in range(lo, hi + 1):
                if rank in available:
                    wanted.add(rank)
            continue
        try:
            rank = int(token)
        except Exception:
            continue
        if rank in available:
            wanted.add(rank)

    if not wanted:
        return None
    return wanted
