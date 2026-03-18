"""Configuration loading with YAML + CLI overrides + environment variables.

Priority: CLI args > config.yaml > defaults > env vars (for api_key only)
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class GeminiConfig(BaseModel):
    api_key: str = ""
    model: str = "gemini-2.5-flash"
    temperature: float = 0.2
    max_retries: int = 5
    retry_base_delay: float = 1.0
    timeout: int = 120


class PipelineConfig(BaseModel):
    concurrency: int = 10
    frame_search_radius: float = 0.5
    frame_search_count: int = 3
    refusal_text: str = (
        "抱歉，由于画面中未展示该文物的清晰铭牌信息，我无法提供准确的背景知识。"
    )


class PathsConfig(BaseModel):
    dataset_root: str = "./input"
    output_file: str = "./output/results.jsonl"
    log_dir: str = "./logs"


class LabelerConfig(BaseModel):
    """Config for the artifact labeler (world knowledge Step D)."""

    enabled: bool = False
    llm_api_key: str = ""
    llm_base_url: str = "http://az.gptplus5.com/v1"
    llm_model: str = "gemini-3.1-flash-lite-preview"
    search_api_url: str = "http://10.32.214.120:8080/server/penetrate/v2.0.0/knowledge/search_v2"
    search_ws_url: str = "ws://10.32.214.120:8080/summary/generation"
    search_secret_key: str = ""
    search_access_key: str = ""
    max_search_turns: int = 4
    max_retries: int = 3
    retry_delay: float = 2.0


class AppConfig(BaseModel):
    gemini: GeminiConfig = GeminiConfig()
    pipeline: PipelineConfig = PipelineConfig()
    paths: PathsConfig = PathsConfig()
    labeler: LabelerConfig = LabelerConfig()


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """Load config from YAML file, falling back to defaults."""
    if config_path and Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        logger.info(f"Loaded config from {config_path}")
        config = AppConfig(**raw)
    else:
        if config_path:
            logger.warning(f"Config file not found: {config_path}, using defaults")
        config = AppConfig()

    # Environment variable fallbacks for API keys
    if not config.gemini.api_key:
        env_key = os.environ.get("GEMINI_API_KEY", "")
        if env_key:
            config.gemini.api_key = env_key
            logger.info("Using GEMINI_API_KEY from environment variable")

    if not config.labeler.llm_api_key:
        env_key = os.environ.get("LABELER_API_KEY", "")
        if env_key:
            config.labeler.llm_api_key = env_key
            logger.info("Using LABELER_API_KEY from environment variable")

    if not config.labeler.search_secret_key:
        env_key = os.environ.get("DASOU_SECRET_KEY", "")
        if env_key:
            config.labeler.search_secret_key = env_key

    if not config.labeler.search_access_key:
        env_key = os.environ.get("DASOU_ACCESS_KEY", "")
        if env_key:
            config.labeler.search_access_key = env_key

    return config


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Worldknowledge Complement Pipeline - Nameplate detection & content filling"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config YAML file"
    )
    parser.add_argument(
        "--dataset-root", type=str, default=None, help="Override paths.dataset_root"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Override paths.output_file"
    )
    parser.add_argument(
        "--concurrency", type=int, default=None, help="Override pipeline.concurrency"
    )
    parser.add_argument(
        "--model", type=str, default=None, help="Override gemini.model"
    )
    parser.add_argument(
        "--api-key", type=str, default=None, help="Override gemini.api_key"
    )
    parser.add_argument(
        "--enable-labeler",
        action="store_true",
        default=None,
        help="Enable artifact labeler for world knowledge generation",
    )
    parser.add_argument(
        "--labeler-api-key", type=str, default=None, help="Override labeler.llm_api_key"
    )
    return parser.parse_args()


def apply_cli_overrides(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    """Apply CLI argument overrides to the config."""
    if args.dataset_root:
        config.paths.dataset_root = args.dataset_root
    if args.output:
        config.paths.output_file = args.output
    if args.concurrency is not None:
        config.pipeline.concurrency = args.concurrency
    if args.model:
        config.gemini.model = args.model
    if args.api_key:
        config.gemini.api_key = args.api_key
    if args.enable_labeler:
        config.labeler.enabled = True
    if args.labeler_api_key:
        config.labeler.llm_api_key = args.labeler_api_key
    return config


def validate_config(config: AppConfig) -> None:
    """Validate config and raise if critical fields are missing."""
    if not config.gemini.api_key:
        raise ValueError(
            "Gemini API key is required. Set via config.yaml, --api-key, or GEMINI_API_KEY env var."
        )
    dataset_root = Path(config.paths.dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")


def setup_logging(log_dir: str, level: int = logging.INFO) -> None:
    """Configure logging with console + file handlers."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(console_fmt)
    root_logger.addHandler(console_handler)

    # File handler
    from datetime import datetime

    log_file = log_path / f"run_{datetime.now().strftime('%Y%m%d_%H%M')}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    file_handler.setFormatter(file_fmt)
    root_logger.addHandler(file_handler)

    # Suppress noisy third-party loggers
    for name in ("google", "httpx", "httpcore", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)

    logger.info(f"Logging initialized. Log file: {log_file}")
