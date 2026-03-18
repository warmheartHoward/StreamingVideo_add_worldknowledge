"""PipelineManager: async orchestration of the entire pipeline.

Flow: discover samples -> load FrameGroups -> fan out async tasks -> collect & write results

Concurrency is controlled via asyncio.Semaphore to limit simultaneous Gemini API calls.
"""

import asyncio
import logging
import time
from typing import Optional

from core.config_loader import AppConfig
from core.gemini_client import GeminiClient
from core.token_tracker import TokenTracker
from models.pipeline_models import EnrichedResult, FrameGroup
from models.output_models import OutputRecord
from stages.reader import DatasetReader
from stages.vlm_caller import VLMCaller
from stages.router import Router
from stages.world_knowledge import (
    MockWorldKnowledge,
    WorldKnowledgeBase,
    enrich_with_world_knowledge,
)
from stages.writer import JSONLWriter

logger = logging.getLogger(__name__)


class PipelineManager:
    """Orchestrates the async pipeline across all samples."""

    def __init__(
        self,
        config: AppConfig,
        wk_provider: Optional[WorldKnowledgeBase] = None,
    ):
        self._config = config
        self._semaphore = asyncio.Semaphore(config.pipeline.concurrency)
        self._reader = DatasetReader(config)

        # Global token tracker
        self._token_tracker = TokenTracker()

        self._vlm_caller = VLMCaller(
            GeminiClient(config.gemini, token_tracker=self._token_tracker)
        )
        self._router = Router(config)
        self._writer = JSONLWriter(config.paths.output_file)

        # Initialize world knowledge provider
        if wk_provider:
            self._wk_provider = wk_provider
        elif config.labeler.enabled:
            from stages.artifact_labeler import ArtifactLabeler

            logger.info("Artifact labeler enabled - using real world knowledge generation")
            self._wk_provider = ArtifactLabeler(
                llm_api_key=config.labeler.llm_api_key,
                llm_base_url=config.labeler.llm_base_url,
                llm_model=config.labeler.llm_model,
                search_api_url=config.labeler.search_api_url,
                search_ws_url=config.labeler.search_ws_url,
                search_secret_key=config.labeler.search_secret_key,
                search_access_key=config.labeler.search_access_key,
                max_search_turns=config.labeler.max_search_turns,
                max_retries=config.labeler.max_retries,
                retry_delay=config.labeler.retry_delay,
                token_tracker=self._token_tracker,
            )
        else:
            logger.info("Artifact labeler disabled - using mock world knowledge")
            self._wk_provider = MockWorldKnowledge()

        # Stats
        self._total = 0
        self._success = 0
        self._failed = 0
        self._positive = 0
        self._negative = 0

    async def run(self) -> dict:
        """Main entry: discover samples, fan out, collect, write.

        Returns a summary dict with processing statistics.
        """
        start_time = time.time()

        # Step A: Discover and load all FrameGroups
        samples = self._reader.discover_samples()
        if not samples:
            logger.warning("No samples found. Exiting.")
            return {"status": "no_samples", "total": 0}

        all_groups: list[FrameGroup] = []
        for sample_dir in samples:
            groups = self._reader.load_sample(sample_dir)
            all_groups.extend(groups)

        self._total = len(all_groups)
        logger.info(
            f"Pipeline start: {len(samples)} samples, "
            f"{self._total} frame groups to process, "
            f"concurrency={self._config.pipeline.concurrency}"
        )

        if not all_groups:
            logger.warning("No frame groups to process. Exiting.")
            await self._writer.close()
            return {"status": "no_frame_groups", "total": 0}

        # Steps B-E: Process all FrameGroups concurrently
        tasks = [self._process_one(fg) for fg in all_groups]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect stats
        for r in results:
            if isinstance(r, Exception):
                self._failed += 1
                logger.error(f"Task failed with exception: {r}")
            elif r is None:
                self._failed += 1
            else:
                self._success += 1

        await self._writer.close()

        elapsed = time.time() - start_time

        # Log token usage summary
        self._token_tracker.log_summary()
        token_summary = self._token_tracker.get_summary()

        summary = {
            "status": "completed",
            "total_samples": len(samples),
            "total_frame_groups": self._total,
            "success": self._success,
            "failed": self._failed,
            "positive_nameplate": self._positive,
            "negative_nameplate": self._negative,
            "elapsed_seconds": round(elapsed, 1),
            "output_file": self._config.paths.output_file,
            "token_usage": token_summary,
        }

        logger.info(
            f"Pipeline complete in {elapsed:.1f}s | "
            f"Total={self._total} Success={self._success} Failed={self._failed} | "
            f"Positive={self._positive} Negative={self._negative}"
        )
        return summary

    async def _process_one(self, fg: FrameGroup) -> Optional[OutputRecord]:
        """Process a single FrameGroup through Steps B-E under semaphore."""
        log_prefix = (
            f"[{fg.sample_id}][qa{fg.qa_index}][resp{fg.response_index}@{fg.response_time}s]"
        )

        try:
            async with self._semaphore:
                # Step B: VLM nameplate detection
                vlm_result = await self._vlm_caller.call(fg)

                # Step C: Routing
                routed = self._router.route(vlm_result)

                # Step D: World knowledge (positive samples only)
                if routed.is_positive:
                    enriched = await enrich_with_world_knowledge(
                        routed, self._wk_provider
                    )
                    self._positive += 1
                else:
                    enriched = EnrichedResult(routed_result=routed)
                    self._negative += 1

                # Step E: Assemble and write
                record = self._writer.assemble(enriched)
                await self._writer.write(record)

                logger.debug(f"{log_prefix} Processed successfully")
                return record

        except Exception as e:
            logger.error(f"{log_prefix} Unexpected error: {e}", exc_info=True)
            return None
