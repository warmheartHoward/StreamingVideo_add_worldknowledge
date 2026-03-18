"""Step A: Data loading and frame alignment.

Discovers sample directories, parses gt.json, and for each response timestamp
finds 1-3 nearby frames from the frames/ directory.
"""

import logging
import os
from pathlib import Path

from pydantic import TypeAdapter

from models.gt_models import GTDocument
from models.pipeline_models import FrameGroup
from core.config_loader import AppConfig

logger = logging.getLogger(__name__)

# TypeAdapter for parsing the gt.json root array
_gt_list_adapter = TypeAdapter(list[GTDocument])


class DatasetReader:
    """Loads samples from dataset_root and produces FrameGroups."""

    def __init__(self, config: AppConfig):
        self._dataset_root = Path(config.paths.dataset_root)
        self._radius = config.pipeline.frame_search_radius
        self._max_frames = config.pipeline.frame_search_count

    def discover_samples(self) -> list[Path]:
        """Find all sample directories containing gt.json under dataset_root."""
        samples = sorted(
            p.parent
            for p in self._dataset_root.glob("*/gt.json")
        )
        logger.info(
            f"Discovered {len(samples)} samples in {self._dataset_root}"
        )
        return samples

    def load_sample(self, sample_dir: Path) -> list[FrameGroup]:
        """Parse gt.json and align frames for each response timestamp.

        Returns a list of FrameGroups, one per response time point.
        """
        gt_path = sample_dir / "gt.json"
        frames_dir = sample_dir / "frames"
        sample_id = sample_dir.name

        if not gt_path.exists():
            logger.error(f"[{sample_id}] gt.json not found at {gt_path}")
            return []

        if not frames_dir.exists():
            logger.error(f"[{sample_id}] frames/ directory not found at {frames_dir}")
            return []

        # Parse gt.json
        try:
            raw_bytes = gt_path.read_bytes()
            documents = _gt_list_adapter.validate_json(raw_bytes)
        except Exception as e:
            logger.error(f"[{sample_id}] Failed to parse gt.json: {e}")
            return []

        # Build frame lookup set for O(1) membership checks
        available_frames = set(os.listdir(frames_dir))
        logger.debug(
            f"[{sample_id}] {len(available_frames)} frames available"
        )

        frame_groups: list[FrameGroup] = []

        for doc in documents:
            video_path = doc.video_path

            for qa_idx, qa in enumerate(doc.data):
                question = qa.question

                for resp_idx, resp in enumerate(qa.response):
                    # Use st_time/end_time range if available, fallback to time point
                    if resp.st_time > 0 and resp.end_time > 0:
                        filenames, paths = self._find_frames_in_range(
                            st_time=resp.st_time,
                            end_time=resp.end_time,
                            available_frames=available_frames,
                            frames_dir=frames_dir,
                        )
                    else:
                        filenames, paths = self._find_nearby_frames(
                            timestamp=resp.time,
                            available_frames=available_frames,
                            frames_dir=frames_dir,
                        )

                    if not filenames:
                        logger.warning(
                            f"[{sample_id}][qa{qa_idx}][resp{resp_idx}] "
                            f"No frames found for t=[{resp.st_time},{resp.end_time}]/{resp.time}s, skipping"
                        )
                        continue

                    # Serialize logits using aliases for output compatibility
                    logits_dict = resp.logits.model_dump(by_alias=True)

                    fg = FrameGroup(
                        sample_id=sample_id,
                        sample_dir=str(sample_dir),
                        video_path=video_path,
                        qa_index=qa_idx,
                        response_index=resp_idx,
                        question_content=question.content,
                        question_time=question.time,
                        response_st_time=resp.st_time,
                        response_end_time=resp.end_time,
                        response_time=resp.time,
                        original_logits=logits_dict,
                        frame_paths=paths,
                        frame_filenames=filenames,
                    )
                    frame_groups.append(fg)

        logger.info(
            f"[{sample_id}] Loaded {len(frame_groups)} frame groups "
            f"from {len(documents)} document(s)"
        )
        return frame_groups

    def _find_frames_in_range(
        self,
        st_time: float,
        end_time: float,
        available_frames: set[str],
        frames_dir: Path,
    ) -> tuple[list[str], list[str]]:
        """Find frames uniformly sampled from [st_time, end_time] range.

        Collects all available frames within the range, then uniformly
        samples up to max_frames from them.

        Returns:
            (filenames, absolute_paths) - parallel lists
        """
        # Collect all frames in the range at 0.10s granularity
        in_range: list[tuple[float, str]] = []
        t = round(st_time, 2)
        while t <= end_time + 0.05:  # small epsilon for float rounding
            if t < 0:
                t = round(t + 0.1, 2)
                continue
            fname = self._timestamp_to_filename(t)
            if fname in available_frames:
                in_range.append((t, fname))
            t = round(t + 0.1, 2)

        if not in_range:
            return [], []

        # Uniformly sample max_frames from the range
        if len(in_range) <= self._max_frames:
            selected = in_range
        else:
            step = len(in_range) / self._max_frames
            indices = [int(i * step) for i in range(self._max_frames)]
            selected = [in_range[i] for i in indices]

        filenames = [fname for _, fname in selected]
        paths = [str(frames_dir / fname) for fname in filenames]
        return filenames, paths

    def _find_nearby_frames(
        self,
        timestamp: float,
        available_frames: set[str],
        frames_dir: Path,
    ) -> tuple[list[str], list[str]]:
        """Find frames near the given timestamp.

        Generates candidate timestamps at 0.10s intervals within [time-radius, time+radius],
        checks which exist in available_frames, sorts by proximity, and returns up to
        max_frames results.

        Returns:
            (filenames, absolute_paths) - parallel lists
        """
        candidates: list[tuple[float, str]] = []

        # Generate candidate timestamps at 0.10s granularity
        steps = int(self._radius / 0.1)
        for offset in range(-steps, steps + 1):
            t = timestamp + offset * 0.1
            if t < 0:
                continue
            fname = self._timestamp_to_filename(t)
            if fname in available_frames:
                distance = abs(t - timestamp)
                candidates.append((distance, fname))

        # Sort by distance to target time (exact match first)
        candidates.sort(key=lambda x: x[0])

        # Take up to max_frames
        selected = candidates[: self._max_frames]

        filenames = [fname for _, fname in selected]
        paths = [str(frames_dir / fname) for fname in filenames]
        return filenames, paths

    @staticmethod
    def _timestamp_to_filename(t: float) -> str:
        """Convert a timestamp in seconds to the frame filename format.

        Examples: 0.0 -> 'time_0.00s.jpg', 42.1 -> 'time_42.10s.jpg'
        """
        return f"time_{t:.2f}s.jpg"
