from __future__ import annotations

import logging
from pathlib import Path
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterator


def configure_run_logging(
    run_dir: Path,
    *,
    level: int = logging.INFO,
    filename: str = "run.log",
    console_level: int = logging.WARNING,
) -> Path:
    """
    Configure project-wide logging for a single run.

    Creates/uses a file at: run_dir / filename.
    Safe to call multiple times; it won't duplicate handlers.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / str(filename)

    root = logging.getLogger()
    root.setLevel(int(level))

    fmt = logging.Formatter(
        fmt="%(asctime)s %(levelname).1s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    _ensure_file_handler(root, log_path, fmt=fmt, level=int(level))
    _ensure_stream_handler(root, fmt=fmt, level=int(console_level))

    return log_path


@dataclass
class RunTimer:
    """Timing helper: `with timer.section(name): ...` then `timer.log_summary(logger)`."""

    _t0: float = field(default_factory=time.perf_counter)
    _durations: Dict[str, float] = field(default_factory=dict)

    @contextmanager
    def section(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._durations[str(name)] = self._durations.get(str(name), 0.0) + (
                time.perf_counter() - t0
            )

    def durations(self) -> Dict[str, float]:
        """A copy of the per-section elapsed seconds recorded so far, e.g. for
        writing into a run's own stats file alongside `log_summary`'s log line."""
        return dict(self._durations)

    def log_summary(self, logger: logging.Logger, *, prefix: str = "timing") -> None:
        total = max(0.0, time.perf_counter() - self._t0)
        if not self._durations:
            logger.info("%s total=%.3fs", prefix, total)
            return

        parts = " ".join(
            f"{k}={self._durations[k]:.3f}s" for k in sorted(self._durations.keys())
        )
        logger.info("%s total=%.3fs %s", prefix, total, parts)


def _ensure_file_handler(
    logger: logging.Logger,
    log_path: Path,
    *,
    fmt: logging.Formatter,
    level: int,
) -> None:
    log_path = Path(log_path)
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler):
            try:
                if Path(h.baseFilename) == log_path:
                    h.setLevel(level)
                    h.setFormatter(fmt)
                    return
            except Exception:
                # If handler is odd/misconfigured, just add ours.
                pass

    # Overwrite the log on a fresh run (instead of appending).
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)


def _ensure_stream_handler(
    logger: logging.Logger,
    *,
    fmt: logging.Formatter,
    level: int,
) -> None:
    for h in logger.handlers:
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
            h.setLevel(level)
            h.setFormatter(fmt)
            return

    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

