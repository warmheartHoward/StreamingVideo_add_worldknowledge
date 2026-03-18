"""Artifact Labeler: real WorldKnowledgeBase implementation.

Two-phase pipeline:
  Phase 1 (Search Agent): LLM with tool-calling searches internal API for artifact info
  Phase 2 (Writer): LLM generates structured appraisal report from search results + image

Uses OpenAI-compatible API (sync) wrapped in asyncio.to_thread for async compatibility.
"""

import asyncio
import base64
import json
import logging
import mimetypes
from pathlib import Path
from typing import Optional

import httpx
from openai import OpenAI

from core.token_tracker import TokenTracker
from models.labeler_models import (
    ArtifactReport,
    LabelerResult,
    SearchLogEntry,
)
from prompts.artifact_search import SEARCH_TOOL_SCHEMA, build_search_system_prompt
from prompts.artifact_writer import WRITER_SYSTEM_PROMPT, build_writer_user_content
from stages.search_tools import SearchClient, format_search_results_for_llm
from stages.world_knowledge import WorldKnowledgeBase

logger = logging.getLogger(__name__)


def _image_to_data_url(image_path: str) -> str:
    """Convert an image file to a base64 data URL."""
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = "image/jpeg"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def _generate_training_markdown(report: ArtifactReport) -> str:
    """Generate a Markdown training caption from the structured report."""
    profile = report.entity_profile
    visual = report.visual_audit
    appraisal = report.appraisal_analysis
    history = report.historical_context

    return f"""# {profile.standard_name}

> {report.snapshot_summary}

## 1. 视觉审计 (Visual Audit)
- **光影氛围**：{visual.scene_atmosphere}
- **保存状况**：{visual.preservation_status}

## 2. 身份档案 (Profile)
| 属性 | 内容 |
| :--- | :--- |
| **标准定名** | {profile.standard_name} |
| **断代** | {profile.era} |
| **分类** | {profile.category} |
| **材质** | {profile.material} |

## 3. 沉浸式鉴赏 (Appraisal Analysis)

### 器型与法式 (Morphology)
{appraisal.morphological_features}

### 纹饰与风格 (Art & Style)
{appraisal.artistic_interpretation}

### 身份辨识 (Identity Diagnosis)
**{appraisal.identity_diagnosis}**

## 4. 历史语境 (Historical Context)
- **礼制功能**：{history.ritual_function}
- **铭文价值**：{history.epigraphic_importance}

> {report.curatorial_conclusion}
""".strip()


class ArtifactLabeler(WorldKnowledgeBase):
    """Full artifact labeling pipeline using search + LLM.

    Integrates:
    - Internal search API (大搜) for knowledge retrieval
    - OpenAI-compatible LLM for tool-calling search agent and report writing
    """

    def __init__(
        self,
        llm_api_key: str,
        llm_base_url: str,
        llm_model: str = "gemini-3.1-flash-lite-preview",
        search_api_url: str = "",
        search_ws_url: str = "",
        search_secret_key: str = "",
        search_access_key: str = "",
        max_search_turns: int = 4,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        token_tracker: Optional[TokenTracker] = None,
    ):
        self._llm_model = llm_model
        self._max_search_turns = max_search_turns
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._token_tracker = token_tracker

        # Initialize OpenAI client (sync)
        http_client = httpx.Client(verify=False, timeout=150.0)
        self._llm_client = OpenAI(
            api_key=llm_api_key,
            base_url=llm_base_url,
            http_client=http_client,
        )

        # Initialize search client
        self._search_client = SearchClient(
            api_url=search_api_url,
            ws_url=search_ws_url,
            secret_key=search_secret_key,
            access_key=search_access_key,
        )

    async def generate(self, image_path: str, ocr_text: str) -> dict:
        """Generate world knowledge for an artifact.

        Runs the two-phase labeling pipeline in a thread pool
        to avoid blocking the async event loop.

        Args:
            image_path: Path to the best frame image.
            ocr_text: OCR text from the nameplate.

        Returns:
            Dictionary containing the full LabelerResult.
        """
        result = await asyncio.to_thread(
            self._run_sync, image_path, ocr_text
        )

        # Report to global token tracker (Step D stats preserved in result.usage_stats)
        if self._token_tracker:
            stats = result.usage_stats
            await self._token_tracker.record(
                "labeler",
                input_tokens=stats.get("input_tokens", 0),
                output_tokens=stats.get("output_tokens", 0),
                api_calls=stats.get("model_calls", 0),
                is_error=result.error is not None,
                sample_id=Path(image_path).stem,
                detail={
                    "search_calls": stats.get("search_calls", 0),
                    "model_used": result.model_used,
                },
            )

        return result.model_dump()

    def _run_sync(self, image_path: str, ocr_text: str) -> LabelerResult:
        """Synchronous core logic with retry wrapper."""
        filename = Path(image_path).name

        for attempt in range(self._max_retries):
            try:
                return self._core_process(image_path, ocr_text, filename)
            except Exception as e:
                if attempt < self._max_retries - 1:
                    logger.warning(
                        f"[{filename}] Attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {self._retry_delay}s..."
                    )
                    import time
                    time.sleep(self._retry_delay)
                else:
                    logger.error(
                        f"[{filename}] All {self._max_retries} attempts failed: {e}"
                    )
                    return LabelerResult(error=str(e))

        return LabelerResult(error="Unexpected: retry loop exited without result")

    def _core_process(
        self, image_path: str, ocr_text: str, filename: str
    ) -> LabelerResult:
        """Core two-phase processing logic.

        Phase 1: Search agent gathers evidence via tool-calling
        Phase 2: Writer generates structured report from evidence + image
        """
        stats = {
            "model_calls": 0,
            "search_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }

        data_url = _image_to_data_url(image_path)

        # Build meta from OCR text
        meta_info = {}
        if ocr_text:
            meta_info["nameplate_ocr"] = ocr_text

        # === Phase 1: Search Agent ===
        search_system_prompt = build_search_system_prompt(filename, meta_info)
        messages = [
            {"role": "system", "content": search_system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "请开始对该文物进行考证，多角度搜索必要信息。"},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ]

        search_turns = 0
        search_log: list[SearchLogEntry] = []

        while search_turns < self._max_search_turns:
            resp = self._llm_client.chat.completions.create(
                model=self._llm_model,
                messages=messages,
                tools=SEARCH_TOOL_SCHEMA,
                tool_choice="auto",
            )

            stats["model_calls"] += 1
            if resp.usage:
                stats["input_tokens"] += resp.usage.prompt_tokens
                stats["output_tokens"] += resp.usage.completion_tokens

            msg = resp.choices[0].message
            tool_calls = msg.tool_calls

            if not tool_calls:
                # Model decided to stop searching
                if msg.content:
                    messages.append({"role": "assistant", "content": msg.content})
                break

            # Process tool calls
            search_turns += 1
            # Add assistant message with tool calls
            messages.append(msg.model_dump())

            for tc in tool_calls:
                if tc.function.name == "search_via_custom_api":
                    stats["search_calls"] += 1
                    query = json.loads(tc.function.arguments).get("query", "")
                    logger.info(f"[{filename}] Search turn {search_turns}: {query}")

                    results, _ = self._search_client.search_simple(query)
                    formatted = format_search_results_for_llm(results)

                    search_log.append(SearchLogEntry(
                        turn=search_turns,
                        query=query,
                        result_summary=formatted[:200],
                    ))

                    messages.append({
                        "tool_call_id": tc.id,
                        "role": "tool",
                        "name": "search_via_custom_api",
                        "content": formatted,
                    })

        # === Phase 2: Build Investigation Context ===
        investigation_parts = []
        step_i = 1
        for m in messages:
            role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
            content = m.get("content") if isinstance(m, dict) else getattr(m, "content", None)

            if role == "assistant" and content:
                investigation_parts.append(
                    f"【步骤 {step_i} - 模型思考】:\n{content}\n"
                )

            tool_calls_data = (
                m.get("tool_calls") if isinstance(m, dict)
                else getattr(m, "tool_calls", None)
            )
            if role == "assistant" and tool_calls_data:
                for tc_data in tool_calls_data:
                    fn_args = (
                        tc_data.get("function", {}).get("arguments", "{}")
                        if isinstance(tc_data, dict)
                        else tc_data.function.arguments
                    )
                    try:
                        q_args = json.loads(fn_args)
                    except (json.JSONDecodeError, TypeError):
                        q_args = {}
                    investigation_parts.append(
                        f"【步骤 {step_i} - 执行搜索】: 关键词 = {q_args.get('query', '')}\n"
                    )
                    step_i += 1

            if role == "tool":
                tool_content = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
                investigation_parts.append(f"【搜索结果反馈】:\n{tool_content}\n")

        history_context = "\n".join(investigation_parts)

        # === Phase 3: Writer generates report ===
        logger.info(f"[{filename}] Generating appraisal report...")
        writer_content = build_writer_user_content(data_url, meta_info, history_context)
        writer_messages = [
            {"role": "system", "content": WRITER_SYSTEM_PROMPT},
            {"role": "user", "content": writer_content},
        ]

        final_resp = self._llm_client.chat.completions.create(
            model=self._llm_model,
            messages=writer_messages,
            tool_choice="none",
            response_format={"type": "json_object"},
        )

        stats["model_calls"] += 1
        if final_resp.usage:
            stats["input_tokens"] += final_resp.usage.prompt_tokens
            stats["output_tokens"] += final_resp.usage.completion_tokens

        # Parse report
        try:
            raw_content = final_resp.choices[0].message.content
            report_dict = json.loads(raw_content)
            report = ArtifactReport.model_validate(report_dict)
        except Exception as e:
            logger.warning(f"[{filename}] Report JSON parse failed: {e}")
            report = ArtifactReport()

        # Generate training markdown
        training_md = _generate_training_markdown(report)

        logger.info(
            f"[{filename}] Complete. "
            f"Tokens: {stats['input_tokens']}+{stats['output_tokens']}, "
            f"Searches: {stats['search_calls']}"
        )

        return LabelerResult(
            report=report,
            training_caption=training_md,
            search_trace=search_log,
            usage_stats=stats,
            model_used=self._llm_model,
        )
