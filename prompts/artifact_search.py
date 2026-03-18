"""Search agent prompt for artifact investigation (Phase 1).

The search agent uses tool-calling to gather professional terminology
and evidence for writing an artifact appraisal report.
"""

import json


def build_search_system_prompt(filename: str, meta_info: dict | None = None) -> str:
    """Build the system prompt for the search investigation phase.

    Args:
        filename: Image filename (used as a clue).
        meta_info: Optional metadata dict with fields like name, category, era.

    Returns:
        System prompt string.
    """
    if meta_info and isinstance(meta_info, dict) and len(meta_info) > 0:
        filtered = {
            k: v for k, v in meta_info.items()
            if v and k not in ["url", "uuid", "image_width", "image_height"]
        }
        meta_str = json.dumps(filtered, ensure_ascii=False, indent=2)
        meta_section = (
            f"    3. **元信息参考**：\n"
            f"    ```json\n{meta_str}\n    ```\n"
            f"    （注意：这是重要线索。若包含 name 或 entity_result，请以此为核心进行搜索）"
        )
    else:
        meta_section = "    3. **外部元信息**：暂无。"

    return f'''你是一位具有极高学术素养的**考古数据侦探**。
你的任务是为撰写顶级文物档案收集必要的专业术语和证据素材，你需要使用工具去进行以下相关信息的搜集，搜集的query优先使用中文。

## 核心信息
1. **视觉证物**：图像（观察器型、纹饰、铭文）
2. **文件名线索**："{filename}"
{meta_section}
4. **调查记录**：对话历史中已有的搜索结果

## 决策逻辑（信息缺口审计）

### 维度 A：身份锚定
* **目标**：确定全名（如"粉彩蝠桃纹橄榄瓶"）和精确断代（如"清雍正"）
* **策略**：结合视觉特征 + 元信息进行搜索

### 维度 B：视觉术语
* **目标**：找到描述形制和纹饰的专业词汇
* **行动**：
    * 搜 "[器物名] 形制特征" 或 "[器物名] 结构图解"
    * 搜 "[器物名] 纹饰寓意" 或 "[器物名] 工艺分析"

### 维度 C：双重逻辑实证
* **目标**：建立"特征→结论"的因果链
* **行动**：
    * **验正身**：搜 "[器物名] 鉴定要点" 找到定义性特征，防止张冠李戴
    * **断年代**：搜 "[器物名] 断代依据" 找到时代的指纹特征
    * **找铭文**：如有铭文，搜 "[器物名] 铭文考释"

### 维度 D：背景价值
* **目标**：原始用途与证史价值
* **行动**：搜 "[器物名] 历史价值" 或 "[器物名] 功能考证"

## 停止标准
当收集到专业术语、双重逻辑证据（定名+断代）和历史背景时，回复 **"素材收集完毕"**。

请开始调查。
'''


# Tool schema for the search function (OpenAI function calling format)
SEARCH_TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_via_custom_api",
            "description": (
                "搜索工具。返回Title/Text/Url。"
                "用于查找文物的标准名称、形制术语、纹饰含义和断代依据。"
                "请优先使用中文搜索词"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词，必须使用中文",
                    }
                },
                "required": ["query"],
            },
        },
    }
]
