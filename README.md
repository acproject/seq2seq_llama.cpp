# llama.cpp-seq2seq

> 基于 [llama.cpp](https://github.com/ggml-org/llama.cpp) 增强的 C/C++ 推理引擎，在保持上游完整能力的基础上，**重点强化 Seq2Seq 序列到序列模型推理**与 **Embedding 嵌入模型处理**两大方向。

本项目作为 llama.cpp 的衍生版本，继承其高效的 GGML/GGUF 张量计算框架以及对 CPU / CUDA / Metal / Vulkan / SYCL / OpenCL 等多种后端的良好支持，同时为 Encoder-Decoder 架构的模型（如 T5、BART、M2M-100、mBART、NLLB 等）以及纯 Encoder 的 Embedding 模型（BGE、E5、GTE、NomicBERT、BERT、Sentence-BERT 等）提供更友好的本地推理能力。

---

## 特性

### 1. Seq2Seq 序列到序列推理

- 支持 **Encoder-Decoder** 架构模型的端到端推理（如 T5、BART、mT5、NLLB、M2M-100 等）
- 统一使用 GGUF 权重格式加载，可与现有 llama.cpp 工具链（量化、转换、CLI）无缝衔接
- 支持多种解码策略：Greedy、Beam Search、Top-k / Top-p / Temperature、DRY、Sampling 等
- 支持 **Encoder Cache** 与 **KV Cache 复用**，长文本翻译/摘要等场景下显著提升推理速度
- 支持流式输出与一次性输出两种模式
- 提供独立的 CLI 工具，便于在服务端、批处理、离线任务中调用

### 2. Embedding 嵌入模型处理

- 支持 BERT / BGE / E5 / GTE / NomicBERT / JinaBERT 等主流 Embedding 模型
- 支持多种 **Pooling** 策略：`mean`（默认）、`cls`、`last`、`rank` 等
- 支持 **Embeddings Normalize** 归一化方式：none、max-abs、taxicab（L1）、euclidean（L2，默认）、p-norm
- 支持批处理多条输入，并计算两两 **余弦相似度矩阵**
- 提供 OpenAI 兼容的 `array` / `json` / `json+` / `raw` 多种输出格式
- 可与 `llama-server` 配合，通过 HTTP API 提供 Embedding 服务

### 3. 继承自 llama.cpp 的通用能力

- 多后端推理：CPU（AVX / AVX2 / AVX-512 / NEON / AMX）、CUDA、Metal、Vulkan、SYCL、OpenCL、CANN、OpenVINO、ZenDNN 等
- 多种量化格式：Q2_K、Q3_K、Q4_K、Q5_K、Q6_K、Q8_0、IQ 系列、FP16 / BF16 等
- 上下文长度、批处理大小、RoPE 缩放、Flash Attention 等参数可灵活配置
- 兼容 OpenAI Chat Completions / Completions / Embeddings 风格的 HTTP API（`llama-server`）

---

## 快速开始

### 1. 克隆与构建

```bash
git clone https://github.com/<your-org>/llama.cpp_seq2seq.git
cd llama.cpp_seq2seq

# 推荐使用 CMake 构建
cmake -B build
cmake --build build --config Release -j
```

构建产物位于 `build/bin/` 目录下，包括：

| 可执行文件 | 用途 |
| --- | --- |
| `llama-cli` | 通用对话 / 补全 CLI |
| `llama-embedding` | Embedding 模型推理 CLI |
| `llama-seq2seq` | Seq2Seq 模型推理 CLI（本文档重点） |
| `llama-server` | OpenAI 兼容的 HTTP 服务 |
| `llama-quantize` | 模型量化工具 |
| `llama-bench` | 性能基准测试 |

> 如果只想构建部分工具，可以指定目标，例如：
> ```bash
> cmake --build build --target llama-seq2seq llama-embedding -j
> ```

### 2. 模型准备

将 HuggingFace / 原厂模型转换为 GGUF 格式：

```bash
# 安装 Python 转换依赖
pip install -r requirements.txt

# 转换 Seq2Seq 模型（例如 T5 / BART / NLLB）
python convert_hf_to_gguf.py ./models/mt5-base --outfile ./models/mt5-base.gguf

# 转换 Embedding 模型（例如 BGE / E5）
python convert_hf_to_gguf.py ./models/bge-small-en --outfile ./models/bge-small-en.gguf

# 量化（可选）
./build/bin/llama-quantize ./models/mt5-base.gguf ./models/mt5-base.Q4_K_M.gguf Q4_K_M
```

---

## 使用示例

### 1. Seq2Seq 推理

将一段中文文本翻译为英文：

```bash
./build/bin/llama-seq2seq \
  -m ./models/mt5-base.Q4_K_M.gguf \
  -p "translate Chinese to English: 你好，世界！" \
  -n 128 \
  --temp 0 \
  --top-k 1
```

使用 Beam Search 进行摘要生成：

```bash
./build/bin/llama-seq2seq \
  -m ./models/bart-large-cnn.Q4_K_M.gguf \
  -p "summarize: <长文本>..." \
  -n 256 \
  --beam-size 4 \
  --temp 0
```

### 2. Embedding 推理

单条文本生成向量：

```bash
./build/bin/llama-embedding \
  -m ./models/bge-small-en.gguf \
  --pooling mean \
  --embd-normalize 2 \
  --embd-output-format json \
  -p "Hello World!"
```

多条文本 + 余弦相似度矩阵：

```bash
./build/bin/llama-embedding \
  -m ./models/bge-small-en.gguf \
  --pooling mean \
  --embd-normalize 2 \
  --embd-output-format json+ \
  --embd-separator "<#sep#>" \
  -p "Castle<#sep#>Stronghold<#sep#>Dog<#sep#>Cat"
```

### 3. 通过 HTTP API 调用（llama-server）

```bash
./build/bin/llama-server \
  -m ./models/bge-small-en.gguf \
  --embedding \
  --pooling mean \
  --host 0.0.0.0 --port 8080
```

Embedding 请求示例：

```bash
curl http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bge-small-en",
    "input": ["Hello World!", "Good morning!"]
  }'
```

Seq2Seq 请求示例：

```bash
curl http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mt5-base",
    "prompt": "translate Chinese to English: 你好，世界！",
    "max_tokens": 128,
    "temperature": 0
  }'
```

---

## 项目结构

```
llama.cpp_seq2seq/
├── ggml/              # 张量计算库（继承自 llama.cpp / ggml）
├── src/               # 核心推理库源码
├── include/           # 对外 C/C++ 头文件（llama.h / llama-cpp.h）
├── common/            # 通用工具：参数解析、采样、对话模板、Jinja 等
├── examples/          # 示例程序（含 embedding 等）
├── tools/             # 工具集：server、cli、quantize、mtmd、mtmd-cli 等
├── convert_hf_to_gguf.py  # HuggingFace -> GGUF 转换脚本
├── convert_lora_to_gguf.py
├── docs/              # 文档
├── tests/             # 单元测试
└── CMakeLists.txt
```

---

## 与上游 llama.cpp 的关系

- 本仓库基于 [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) 进行功能扩展与定制
- 周期性 rebase / merge 上游最新变更，保持对通用 LLM（Llama、Qwen、Mistral、Gemma、Phi 等）的兼容
- 任何对核心算子、量化器、采样器的修改都会在 `docs/` 中进行额外说明
- 若需提交 Issue 或 PR，请优先确认问题是否已在上游修复

---

## 致谢

- [ggerganov / ggml-org](https://github.com/ggml-org) 及 llama.cpp 全体贡献者
- 所有上游依赖项目：GGML、SentencePiece、HuggingFace Transformers、nlohmann/json 等

---

## 许可协议

本项目遵循与上游 llama.cpp 相同的 [MIT 许可证](LICENSE)。
