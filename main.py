"""Entry point for the Worldknowledge Complement Pipeline.

Usage:
    python main.py --config config.yaml --dataset-root ./input
    python main.py --dataset-root ../v2_project/output --concurrency 5
    python main.py --api-key YOUR_KEY --model gemini-2.5-flash
    python main.py --enable-labeler --labeler-api-key YOUR_KEY
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

# Add project root to sys.path for absolute imports
sys.path.insert(0, str(Path(__file__).parent))

from core.config_loader import (
    AppConfig,
    apply_cli_overrides,
    load_config,
    parse_args,
    setup_logging,
    validate_config,
)
from core.pipeline import PipelineManager

logger = logging.getLogger(__name__)


async def main() -> None:
    """Parse config, validate, and run the pipeline."""
    args = parse_args()

    # Load and configure
    config = load_config(args.config)
    config = apply_cli_overrides(config, args)

    # Setup logging before validation (so errors are logged)
    setup_logging(config.paths.log_dir)

    logger.info("=" * 60)
    logger.info("Worldknowledge Complement Pipeline")
    logger.info("=" * 60)
    logger.info(f"Dataset root: {config.paths.dataset_root}")
    logger.info(f"Output file:  {config.paths.output_file}")
    logger.info(f"Model:        {config.gemini.model}")
    logger.info(f"Concurrency:  {config.pipeline.concurrency}")
    logger.info(f"Labeler:      {'ENABLED (' + config.labeler.llm_model + ')' if config.labeler.enabled else 'DISABLED (mock)'}")

    # Validate
    try:
        validate_config(config)
    except (ValueError, FileNotFoundError) as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)

    # Run pipeline
    pipeline = PipelineManager(config)
    summary = await pipeline.run()

    # Save summary
    summary_path = Path(config.paths.log_dir) / "last_run_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Run summary saved to {summary_path}")

    # Print summary
    print("\n" + "=" * 40)
    print("Pipeline Summary")
    print("=" * 40)
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    asyncio.run(main())
