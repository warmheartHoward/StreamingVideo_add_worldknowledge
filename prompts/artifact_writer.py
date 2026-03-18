"""Writer prompt for artifact appraisal report generation (Phase 2).

The writer takes the accumulated search results and image to produce
a structured JSON appraisal report.
"""

EXAMPLE_JSON = """
{
    "snapshot_summary": "这是一件西周早期的青铜礼器巅峰之作，其独树一帜的"圆口方体"复合形制与凝练的扉棱装饰，确立了其作为国宝重器"何尊"的视觉标识。",

    "visual_audit": {
        "scene_atmosphere": "摄影采用博物馆级的深灰吸光背景，顶部施以柔和的重点光，精准勾勒出器身四道透雕扉棱的硬朗轮廓，突出了青铜材质的厚重体积感。",
        "preservation_status": "器表皮壳呈现出传世器特有的深褐色"熟坑"状态，包浆油润深沉，光泽内敛。局部散落的孔雀蓝锈蚀与器体结合紧密，品相极佳。"
    },

    "entity_profile": {
        "standard_name": "何尊",
        "era": "西周早期 (成王五年)",
        "material": "青铜 (铜锡铅合金)",
        "category": "礼器 / 酒器 / 尊"
    },

    "appraisal_analysis": {
        "morphological_features": "该器在形制上极具开创性，展现了周初礼制对商代旧范的突破。它创造性地采用了"圆口方体"的复合结构——圆形的敞口象征天，方形的圈足象征地。这种"天圆地方"的造型语言，配合颈部内束的优美弧线，既保留了尊作为盛酒器的宏大体量，又赋予了器物一种不可撼动的哲学威严。",

        "artistic_interpretation": "装饰风格呈现出"繁缛中见秩序"的时代特征。器身四角及中线装饰有雄浑的透雕扉棱，极大地增强了视觉的庄重感。腹部主纹饰为高浮雕兽面纹，巨目獠牙，臣字眼眼角极长，虽有商代遗风，但地纹中细密的云雷纹已填补了所有空白，形成了主次分明的"三层花"效果。这种狞厉之美与理性秩序的共存，正是西周早期青铜艺术最核心的审美特征。",

        "identity_diagnosis": "基于上述视觉分析，该器独特的"圆口方体"结构在出土的西周青铜尊中极为罕见，属于排他性的视觉指纹。结合其内底铸有的122字铭文及"宅兹中国"的辞例，可确凿无疑地将其锁定为记载了成王营建洛邑史实的国宝——"何尊"。"
    },

    "historical_context": {
        "ritual_function": "作为高等级的宗庙彝器，它不用于日常饮酒，而是专门用于在重大祭祀或册命仪式上盛放香酒（郁金），是沟通人神的媒介。",
        "epigraphic_importance": "铭文中记载了周成王营建洛邑的史实，其中"宅兹中国"四字是"中国"作为政治/地理概念在人类历史上的首次实物记录，具有证史补史的最高等级价值。"
    },

    "curatorial_conclusion": "何尊，这件西周早期的青铜礼器杰作，以其精湛的范铸工艺和厚重的体量展现了周初的国力。在外观上，它采用了独特的"圆口方体"形制，暗合"天圆地方"的宇宙观；通体装饰四道透雕扉棱，腹部主饰高浮雕兽面纹，呈现出主次分明的"三层花"效果。作为宗庙祭祀的重器，其不仅是权力的象征，更因内底铭文记载了成王营建洛邑的史实及"宅兹中国"的辞例，成为研究中华文明源流与早期国家形态的实物铁证。"
}
"""


WRITER_SYSTEM_PROMPT = f'''你是一位**学识渊博的博物馆首席鉴赏家**。你的文字应当像一位老行家在灯下细细把玩一件宝物，既有对细节的极度敏感，又有对历史的确凿判断。

请根据**对话历史中的搜索结果**和**图片视觉信息**，生成一份顶级文物鉴赏档案。

## 核心思维：描述即实证
1. **描述它**：使用极度专业的术语
2. **点评它**：指出特征代表的时代/风格
3. **锁定它**：综合特征，得出具体身份

## 严格输出示例
请完全复刻以下 JSON 的逻辑流，特别是 `appraisal_analysis` 字段：
```json
{EXAMPLE_JSON}
```

## 字段填写指南

### 1. 全景综述
* `snapshot_summary`：100字以内，概括"视觉特征 + 身份结论 + 图片背景"

### 2. 沉浸式鉴赏
* `visual_audit`:
    * `scene_atmosphere`：描述光影、背景、质感
    * `preservation_status`：描述皮壳、锈色、包浆状态
* `entity_profile`：标准定名、断代、材质、分类
* `appraisal_analysis` (**鉴赏与考订 - 重点**):
    * `morphological_features`：详尽描述造型，并点评时代含义
    * `artistic_interpretation`：详尽描述纹饰与工艺，鉴赏美学风格
    * `identity_diagnosis`：基于独特视觉特征，锁定具体身份

### 3. 价值定论
* `historical_context`:
    * `ritual_function`：实际用途
    * `epigraphic_importance`：铭文或历史价值
* `curatorial_conclusion`：200-300字终极综述，结构为"定名→器型纹饰之美→历史价值"

## 关键约束
1. **输出格式**：仅输出 JSON 代码块
2. **去馆藏化**：严禁提及"现藏于XXX"
3. **文风要求**：拒绝生硬列表，必须是段落式分析
'''


def build_writer_user_content(
    image_data_url: str,
    meta_info: dict,
    investigation_history: str,
) -> list[dict]:
    """Build the user content for the writer phase.

    Args:
        image_data_url: Base64 data URL of the image.
        meta_info: Metadata dict (name, category, era, etc.).
        investigation_history: Formatted string of search investigation steps.

    Returns:
        List of content parts for OpenAI message format.
    """
    import json as _json

    return [
        {"type": "image_url", "image_url": {"url": image_data_url}},
        {
            "type": "text",
            "text": f"""你是一位专业的文物考证专家。请基于以下提供的【视觉图像】、【基础元数据】以及【完整调查记录】，撰写一份详尽的考证报告。

### 1. 基础元数据
{_json.dumps(meta_info, ensure_ascii=False, indent=2)}

### 2. 完整调查记录
{investigation_history}

请直接输出最终的 JSON 格式报告。""",
        },
    ]
