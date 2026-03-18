"""Global token usage tracker for all pipeline stages.

Thread-safe singleton that collects token counts from Gemini VLM (Step B)
and Artifact Labeler (Step D), then produces a unified usage report.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class StageUsage:
    """Token usage for a single pipeline stage."""

    input_tokens: int = 0
    output_tokens: int = 0
    api_calls: int = 0
    errors: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class TokenTracker:
    """Global token usage tracker across all pipeline stages.

    All public methods are thread-safe via asyncio.Lock.
    Designed to be shared across the entire pipeline run.
    """

    def __init__(self):
        self._lock = asyncio.Lock()
        self._stages: dict[str, StageUsage] = {}
        self._start_time: float = time.time()
        self._sample_details: list[dict] = []

    async def record(
        self,
        stage: str,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        api_calls: int = 1,
        is_error: bool = False,
        sample_id: str = "",
        detail: Optional[dict] = None,
    ) -> None:
        """Record token usage for a stage.

        Args:
            stage: Stage name (e.g., "gemini_vlm", "labeler_search", "labeler_writer").
            input_tokens: Number of input/prompt tokens consumed.
            output_tokens: Number of output/completion tokens generated.
            api_calls: Number of API calls made (default 1).
            is_error: Whether this call resulted in an error.
            sample_id: Optional sample identifier for per-sample tracking.
            detail: Optional extra detail dict to attach to sample log.
        """
        async with self._lock:
            if stage not in self._stages:
                self._stages[stage] = StageUsage()

            usage = self._stages[stage]
            usage.input_tokens += input_tokens
            usage.output_tokens += output_tokens
            usage.api_calls += api_calls
            if is_error:
                usage.errors += 1

            if detail or sample_id:
                entry = {
                    "stage": stage,
                    "sample_id": sample_id,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                }
                if detail:
                    entry.update(detail)
                self._sample_details.append(entry)

    def get_summary(self) -> dict:
        """Return a summary dict of all tracked usage.

        Returns:
            Dictionary with per-stage breakdown and totals.
        """
        elapsed = time.time() - self._start_time
        stages_data = {}
        total_input = 0
        total_output = 0
        total_calls = 0
        total_errors = 0

        for name, usage in self._stages.items():
            stages_data[name] = {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "total_tokens": usage.total_tokens,
                "api_calls": usage.api_calls,
                "errors": usage.errors,
            }
            total_input += usage.input_tokens
            total_output += usage.output_tokens
            total_calls += usage.api_calls
            total_errors += usage.errors

        return {
            "total": {
                "input_tokens": total_input,
                "output_tokens": total_output,
                "total_tokens": total_input + total_output,
                "api_calls": total_calls,
                "errors": total_errors,
            },
            "stages": stages_data,
            "elapsed_seconds": round(elapsed, 1),
        }

    def log_summary(self) -> None:
        """Log a human-readable usage summary."""
        summary = self.get_summary()
        total = summary["total"]

        logger.info("=" * 50)
        logger.info("Token Usage Summary")
        logger.info("=" * 50)
        logger.info(
            f"  Total: {total['total_tokens']:,} tokens "
            f"(input: {total['input_tokens']:,}, output: {total['output_tokens']:,})"
        )
        logger.info(
            f"  API calls: {total['api_calls']:,}, errors: {total['errors']:,}"
        )

        for stage_name, stage_data in summary["stages"].items():
            logger.info(
                f"  [{stage_name}] "
                f"{stage_data['total_tokens']:,} tokens "
                f"(in: {stage_data['input_tokens']:,}, out: {stage_data['output_tokens']:,}) | "
                f"calls: {stage_data['api_calls']}, errors: {stage_data['errors']}"
            )
        logger.info("=" * 50)
