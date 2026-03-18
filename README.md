# Worldknowledge Complement Pipeline

博物馆视频QA数据的**世界知识补全管线**。读取已有的视频抽帧数据集（`frames/` + `gt.json`），通过 VLM 检测展品铭牌，对正样本调用搜索+LLM生成结构化文物鉴赏档案填充回答内容，对负样本生成安全拒答，最终输出 JSONL 格式的增强数据。

## 架构总览

```
输入: dataset_root/sample_xxx/{frames/*.jpg, gt.json}
                         │
                    Step A: 数据加载 & 帧采样
                    (解析gt.json, 按[st_time, end_time]区间均匀采帧)
                         │
                    Step B: 铭牌检测 (VLM)
                    (OpenAI兼容API, base64图文交替 → 结构化JSON)
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
                         │
                    TokenTracker: 全局 token 用量统计
```

## 项目结构

```
worldknowledge_complement_pipeline/
├── main.py                          # 入口
├── config.yaml                      # 运行时配置 (不入库, 含密钥)
├── config_example.yaml              # 配置模板 (复制为 config.yaml 使用)
├── requirements.txt
├── models/                          # Pydantic 数据模型
│   ├── gt_models.py                 #   gt.json 解析 (支持 st_time/end_time)
│   ├── vlm_models.py                #   VLM 结构化输出 schema
│   ├── pipeline_models.py           #   管线内部数据载体 (FrameGroup → EnrichedResult)
│   ├── labeler_models.py            #   文物标注报告模型 (ArtifactReport)
│   └── output_models.py             #   JSONL 输出 schema
├── core/                            # 核心框架
│   ├── config_loader.py             #   YAML + CLI + 环境变量三级配置
│   ├── gemini_client.py             #   OpenAI兼容 VLM 异步客户端 (AsyncOpenAI)
│   ├── token_tracker.py             #   全局 token 用量追踪器
│   └── pipeline.py                  #   PipelineManager 异步编排
├── stages/                          # 管线各步骤
│   ├── reader.py                    #   Step A: 数据加载 & 帧采样
│   ├── vlm_caller.py                #   Step B: VLM 铭牌检测
│   ├── router.py                    #   Step C: 正/负样本路由
│   ├── world_knowledge.py           #   Step D: 抽象基类 + Mock实现
│   ├── artifact_labeler.py          #   Step D: 真实实现 (搜索Agent + 报告Writer)
│   ├── search_tools.py              #   内部搜索API封装 (大搜 SSE + WebSocket)
│   └── writer.py                    #   Step E: JSONL 组装 & 输出
└── prompts/                         # Prompt 模板
    ├── nameplate_detection.py       #   铭牌检测 (Step B)
    ├── artifact_search.py           #   搜索Agent (Step D Phase 1)
    └── artifact_writer.py           #   报告生成 (Step D Phase 2)
```

## 处理流程详解

### Step A: 数据加载 & 帧采样 (`reader.py`)

**输入**: `dataset_root/*/gt.json` + `frames/` 目录

**帧采样策略**：根据 `response` 的时间定义方式自动选择：

| 场景 | 时间字段 | 采样方式 |
|------|----------|----------|
| 区间模式 | `st_time > 0` 且 `end_time > 0` | 从 `[st_time, end_time]` 区间内**等间距**采样 `frame_search_count` 帧 |
| 单点模式 | 仅 `time` 有值 | 以 `time` 为中心, `±frame_search_radius` 范围内按距离排序取最近帧 |

**等间距采样示例**（区间 `[40.0s, 45.0s]`, `frame_search_count=3`）:
```
区间内可用帧: 51帧 (40.00s ~ 45.00s, 每0.10s一帧)
step = 51 / 3 = 17
采样索引: [0, 17, 34]
采样结果: time_40.00s.jpg, time_41.70s.jpg, time_43.40s.jpg
          |___ 区间头部     |___ 区间中部     |___ 区间尾部
```

**帧数建议**: 对于铭牌检测任务, 建议 `frame_search_count` 设为 **3~6** 帧。超过 10 帧后延迟和成本线性增长, 但识别收益边际递减。

### Step B: 铭牌检测 (`vlm_caller.py` + `gemini_client.py`)

**调用方式**: 通过 OpenAI 兼容 API (AsyncOpenAI), 支持任意 OpenAI 协议的代理或端点。

**消息构造**:
```
system: 铭牌识别专家指令 + JSON输出格式约束
user:   [text: "文件名: x.jpg", image_url: base64, ...重复N帧..., text: 分析指令]
```

**结构化输出** (`response_format=json_object`):
```json
{
  "artifact_description": "展品描述",
  "has_legible_nameplate": true,
  "reasoning_process": "判断推理过程",
  "best_frame_filename": "time_41.70s.jpg",
  "ocr_text": "明 青花缠枝莲纹梅瓶\n高 37.5cm 口径 6.2cm"
}
```

**重试策略**: 指数退避, `wait = min(base_delay × 2^attempt, 60s)`, 对 401/403 等认证错误不重试。

### Step C: 分支路由 (`router.py`)

| 条件 | 路由 | 动作 |
|------|------|------|
| `has_legible_nameplate=true` 且无错误 | WORLD_KNOWLEDGE | 进入 Step D 世界知识生成 |
| 其余情况 | REFUSAL | 填入配置的 `refusal_text` |

### Step D: 世界知识生成

提供两种实现, 通过 `labeler.enabled` 配置切换:

**Mock 模式** (`enabled: false`): 返回占位内容, 用于快速调试管线流程。

**Labeler 模式** (`enabled: true`): 两阶段 LLM pipeline:

| 阶段 | 描述 | 输入 | 输出 |
|------|------|------|------|
| Phase 1: 搜索Agent | LLM + Tool Calling, 最多 `max_search_turns` 轮搜索 | 图片 + OCR文本 | 搜索证据链 |
| Phase 2: 报告Writer | LLM 生成结构化鉴赏报告 | 图片 + 元数据 + 搜索记录 | `ArtifactReport` JSON |

搜索Agent 通过内部搜索API (大搜) 检索文物信息, 请求使用 HMAC-SHA256 签名。Writer 生成的报告包含: 视觉审计、身份档案、鉴赏分析、历史语境等六大模块。

### Step E: 组装 & 输出 (`writer.py`)

**内容填充优先级** (正样本):
1. `report.curatorial_conclusion` — Labeler 报告的策展结论
2. `training_caption` — Markdown 格式的训练标注文本
3. `ocr_text + artifact_description` — Mock 模式下的 OCR + 描述拼接
4. 路由层的 fallback 文本

**写入方式**: `aiofiles` 异步追加写入, `asyncio.Lock` 保证并发安全。

### Token 用量追踪 (`token_tracker.py`)

全局 `TokenTracker` 实例在管线启动时创建, 注入到 Step B (VLM) 和 Step D (Labeler) 中:

```
Token Usage Summary
==================================================
  Total: 32,456 tokens (input: 28,123, output: 4,333)
  API calls: 12, errors: 0
  [gemini_vlm] 27,280 tokens (in: 26,488, out: 792) | calls: 8, errors: 0
  [labeler] 5,176 tokens (in: 1,635, out: 3,541) | calls: 4, errors: 0
==================================================
```

Step D 内部的细粒度统计 (`usage_stats`: model_calls, search_calls, input/output_tokens) 保留在每条输出记录的 `nameplate.world_knowledge` 中。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置

```bash
cp config_example.yaml config.yaml
# 编辑 config.yaml, 填写 API Key 和 base_url
```

**配置优先级**: CLI 参数 > config.yaml > 环境变量 > 代码默认值

**关键配置项**:

```yaml
gemini:
  api_key: "your-key"             # VLM API 密钥
  base_url: "http://xxx/v1"       # OpenAI 兼容端点
  model: "gemini-3.1-flash-lite-preview"

pipeline:
  concurrency: 10                 # 并发数 (asyncio.Semaphore)
  frame_search_count: 3           # 每个response采样帧数 (建议3~6)
  frame_search_radius: 0.5        # 单点模式下的搜索半径 (秒)

labeler:
  enabled: false                  # true 启用真实标注, false 使用 Mock
  llm_api_key: "your-key"        # 标注 LLM 的 API 密钥
  llm_base_url: "http://xxx/v1"  # 标注 LLM 端点
```

**环境变量**:
- `GEMINI_API_KEY` — VLM API 密钥
- `LABELER_API_KEY` — 文物标注 LLM 密钥
- `DASOU_SECRET_KEY` / `DASOU_ACCESS_KEY` — 内部搜索 API 凭证

### 3. 准备数据

输入目录结构:
```
dataset_root/
├── 开封博物馆_seg000/
│   ├── frames/
│   │   ├── time_0.00s.jpg        # 帧命名格式: time_{秒数:.2f}s.jpg
│   │   ├── time_0.10s.jpg
│   │   └── ...
│   └── gt.json                   # QA数据, response[].content 为空
└── 南京博物馆_seg001/
    ├── frames/
    └── gt.json
```

**gt.json 格式** (输入时 content 为空):
```json
[{
  "meta_info": {"id": "开封博物馆_seg000", "dataset": "video_qa_v2", ...},
  "video_path": "video/开封博物馆_seg000.mp4",
  "frame_path": ["frames/time_0.00s.jpg", ...],
  "extracted_fps": 10.0,
  "data": [{
    "question": {"content": "", "time": 35.0},
    "response": [{
      "content": "",
      "st_time": 40.0,
      "end_time": 45.0,
      "time": 42.0,
      "logits": {"</first_response>": 1.0, "</second_response>": 0.0, "</silence>": 0.0, "</standby>": 0.0}
    }]
  }]
}]
```

### 4. 运行

```bash
# Mock 模式 (不启用文物标注, 快速验证管线)
python main.py --config config.yaml

# 完整模式 (启用文物标注)
python main.py --enable-labeler --labeler-api-key YOUR_LLM_KEY

# 自定义参数
python main.py --concurrency 5 --model gemini-2.5-flash --output ./out/result.jsonl
```

### 5. VSCode 调试

项目已配置 `launch.json`, 在 VSCode 侧边栏选择对应配置后按 F5:

| 配置名 | 说明 |
|--------|------|
| WK Pipeline: Mock模式 | 默认配置, labeler 关闭 |
| WK Pipeline: Labeler模式 | 带 `--enable-labeler` |
| WK Pipeline: 自定义参数 | 可按需修改 `args` |

## 输出格式

JSONL 文件, 每行一条记录 (对应一个 `response` 时间点):

```json
{
  "sample_id": "开封博物馆_seg000",
  "video_path": "video/开封博物馆_seg000.mp4",
  "qa_index": 1,
  "response_index": 0,
  "question": {"content": "", "time": 52.0},
  "response": {
    "content": "填充的回答内容 (正样本为世界知识, 负样本为拒答文本)",
    "st_time": 54.0,
    "end_time": 60.0,
    "time": 56.0,
    "logits": {"</first_response>": 1.0, "</second_response>": 0.0, "</silence>": 0.0, "</standby>": 0.0}
  },
  "nameplate": {
    "has_legible_nameplate": true,
    "artifact_description": "展柜中陈列的一件青铜器",
    "reasoning_process": "帧 time_56.00s.jpg 中可见清晰铭牌...",
    "best_frame_filename": "time_56.00s.jpg",
    "ocr_text": "商 兽面纹铜鼎\n通高 24.5cm",
    "world_knowledge": {
      "report": {"curatorial_conclusion": "...", "entity_profile": {...}, ...},
      "training_caption": "# 兽面纹铜鼎\n> ...",
      "search_trace": [{"turn": 1, "query": "兽面纹铜鼎", "result_summary": "..."}],
      "usage_stats": {"model_calls": 3, "search_calls": 2, "input_tokens": 1200, "output_tokens": 800}
    }
  }
}
```

运行结束后还会在 `logs/last_run_summary.json` 中输出统计摘要, 包含 token 用量明细。

## 技术栈

| 组件 | 技术 | 说明 |
|------|------|------|
| 异步框架 | `asyncio` + `Semaphore` | 并发控制, 可配置并发数 |
| VLM 调用 | `openai` (AsyncOpenAI) | OpenAI 兼容协议, 支持任意代理端点 |
| 数据模型 | `Pydantic v2` | 全链路类型校验, alias 支持特殊 token 名 |
| 文物标注 | `openai` (tool calling) + 大搜 API | 搜索Agent + Writer 两阶段 pipeline |
| Token 追踪 | `TokenTracker` | 全局异步安全, 按 stage 分类统计 |
| 配置管理 | YAML + argparse + env var | 三级覆盖, 密钥支持环境变量注入 |
| 文件输出 | `aiofiles` + `asyncio.Lock` | 异步追加写入, 并发安全 |
