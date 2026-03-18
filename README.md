# Worldknowledge Complement Pipeline

博物馆视频QA数据的**世界知识补全管线**。读取已有的视频抽帧数据集（`frames/` + `gt.json`），通过 Gemini VLM 检测展品铭牌，对正样本调用搜索+LLM生成结构化文物鉴赏档案填充回答内容，对负样本生成安全拒答，最终输出 JSONL 格式的增强数据。

## 架构总览

```
输入: dataset_root/sample_xxx/{frames/*.jpg, gt.json}
                         │
                    Step A: 数据加载 & 帧对齐
                    (解析gt.json, 按response.time找附近帧)
                         │
                    Step B: 铭牌检测 (Gemini VLM)
                    (交替图文输入 → 结构化JSON输出)
                         │
                    Step C: 分支路由
                    ┌──────┴──────┐
              有铭牌 ✓          无铭牌 ✗
                    │              │
              Step D: 世界知识     拒答文本
              (搜索Agent+报告生成)
                    │              │
                    └──────┬──────┘
                    Step E: 组装 → JSONL 输出
```

## 项目结构

```
worldknowledge_complement_pipeline/
├── main.py                          # 入口
├── config_example.yaml              # 配置模板 (复制为 config.yaml 使用)
├── requirements.txt
├── models/                          # Pydantic 数据模型
│   ├── gt_models.py                 #   gt.json 解析
│   ├── vlm_models.py                #   Gemini 结构化输出 schema
│   ├── pipeline_models.py           #   管线内部数据载体
│   ├── labeler_models.py            #   文物标注报告模型
│   └── output_models.py             #   JSONL 输出 schema
├── core/                            # 核心框架
│   ├── config_loader.py             #   YAML + CLI + 环境变量配置
│   ├── gemini_client.py             #   google-genai 异步封装
│   └── pipeline.py                  #   PipelineManager 异步编排
├── stages/                          # 管线各步骤
│   ├── reader.py                    #   Step A: 数据加载 & 帧对齐
│   ├── vlm_caller.py                #   Step B: Gemini 铭牌检测
│   ├── router.py                    #   Step C: 正/负样本路由
│   ├── world_knowledge.py           #   Step D: 抽象基类 + Mock
│   ├── artifact_labeler.py          #   Step D: 真实实现 (搜索+LLM)
│   ├── search_tools.py              #   内部搜索API封装
│   └── writer.py                    #   Step E: JSONL 输出
└── prompts/                         # Prompt 模板
    ├── nameplate_detection.py       #   铭牌检测 (Step B)
    ├── artifact_search.py           #   搜索Agent (Step D Phase 1)
    └── artifact_writer.py           #   报告生成 (Step D Phase 2)
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置

```bash
cp config_example.yaml config.yaml
# 编辑 config.yaml, 填写 API Key 等信息
```

配置优先级: CLI 参数 > config.yaml > 环境变量 > 代码默认值

支持的环境变量:
- `GEMINI_API_KEY` — Gemini API 密钥
- `LABELER_API_KEY` — 文物标注 LLM 密钥
- `DASOU_SECRET_KEY` / `DASOU_ACCESS_KEY` — 内部搜索 API 凭证

### 3. 准备数据

输入数据集目录结构:
```
dataset_root/
├── video_sample_001/
│   ├── frames/
│   │   ├── time_0.00s.jpg
│   │   ├── time_0.10s.jpg
│   │   └── ...
│   └── gt.json
└── video_sample_002/
    ├── frames/
    └── gt.json
```

`gt.json` 格式与 v2_project 输出一致，`response[].content` 为空待填充。

### 4. 运行

```bash
# Mock 模式 (不启用文物标注, 快速测试)
python main.py --dataset-root ./input --api-key YOUR_GEMINI_KEY

# 完整模式 (启用文物标注)
python main.py --dataset-root ./input --api-key YOUR_GEMINI_KEY \
    --enable-labeler --labeler-api-key YOUR_LLM_KEY

# 其他常用参数
python main.py --concurrency 5 --model gemini-2.5-flash --output ./out/result.jsonl
```

## 输出格式

JSONL 文件，每行一条记录（对应 `gt.json` 中一个 `response` 时间点）:

```json
{
  "sample_id": "南京博物馆_seg002",
  "video_path": "video/南京博物馆_seg002.mp4",
  "qa_index": 1,
  "response_index": 0,
  "question": {"content": "...", "time": 45.0},
  "response": {"content": "已填充的回答内容", "time": 52.0, "st_time": "", "end_time": "", "logits": {"</first_response>": 1.0, ...}},
  "nameplate": {
    "has_legible_nameplate": true,
    "artifact_description": "...",
    "reasoning_process": "...",
    "best_frame_filename": "time_52.00s.jpg",
    "ocr_text": "明 青花缠枝莲纹梅瓶\n高 37.5cm",
    "world_knowledge": {"report": {...}, "training_caption": "...", ...}
  }
}
```

## 技术栈

| 组件 | 技术 |
|------|------|
| 异步框架 | `asyncio` + `asyncio.Semaphore` 并发控制 |
| VLM | `google-genai` SDK (结构化 JSON 输出) |
| 数据校验 | `Pydantic v2` 全链路模型验证 |
| 文物标注 | `openai` SDK (tool calling) + 内部搜索 API |
| 配置 | YAML + CLI + 环境变量三级覆盖 |
| 输出 | `aiofiles` 异步追加写入 JSONL |
