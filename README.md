# 计算机网络知识问答 RAG 系统

基于 LlamaIndex 和 vLLM 的高性能检索增强生成（RAG）系统，专注于计算机网络课程知识问答。

## 📚 项目简介

本项目实现了一个完整的 RAG 系统，使用语义分块、混合检索（HyDE + 向量检索 + BM25）和 vLLM 高性能推理，为计算机网络课程提供准确、可靠的问答服务。

### 主要特点

- 🎯 **语义分块**：使用 SemanticSplitterNodeParser 保证知识块的语义完整性
- 🔍 **混合检索**：融合 HyDE、稠密向量检索和 BM25 稀疏检索
- ⚡ **高性能推理**：基于 vLLM 的异步推理引擎，支持 4 卡并行
- 📊 **完整评估**：使用 RAGAS 框架进行多维度评估
- 🛠️ **统一配置**：所有参数集中管理，便于调优

## 🏗️ 技术架构

### 核心技术栈

| 组件 | 技术选型 | 说明 |
|------|---------|------|
| **RAG 框架** | LlamaIndex 0.11+ | 文档加载、分块、索引、检索 |
| **向量数据库** | ChromaDB 0.5+ | 本地持久化向量存储 |
| **Embedding 模型** | Qwen3-Embedding-4B | 中文优化，MTEB 排名第 2 |
| **LLM 模型** | Qwen2.5-14B-Instruct | 高性能生成模型 |
| **推理引擎** | vLLM | 异步推理，支持张量并行 |
| **Web 框架** | FastAPI | 高性能异步 API 服务 |
| **评估框架** | RAGAS | 忠实度、召回率、精度、相关性 |

### 系统架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                         数据准备层                                │
│  PDF 文档 → 语义分块 → 向量化 → ChromaDB 持久化                   │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                         检索优化层                                │
│  用户查询 → HyDE 增强 → 混合检索 (向量 + BM25) → RRF 融合         │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                         生成服务层                                │
│  上下文构建 → Prompt 模板 → vLLM 推理 → 答案生成                  │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                         评估层                                    │
│  RAGAS 评估 → 多维度指标 → 性能分析                              │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 项目结构

```
rag_using_llamaindex/
├── README.md                           # 项目说明文档
├── requirements.txt                    # Python 依赖
├── 总结.txt                            # 技术选型总结
│
└── rag_for_computer_network_knowledge/
    ├── config.py                       # 统一配置文件 ⚙️
    ├── model_load.py                   # 模型加载模块
    ├── data_preparation.py             # 数据准备和语义分块
    ├── index_construction.py           # 向量索引构建
    ├── retrieval_optimization.py       # 混合检索优化
    ├── vllm_server.py                  # vLLM 推理服务端
    ├── vllm_client.py                  # 交互式问答客户端
    ├── evaldata_construct.py           # 批量评估数据构建
    ├── run_ragas_eval.py               # RAGAS 评估执行
    │
    ├── data/                           # 数据目录
    │   ├── books/                      # 教材 PDF 文件
    │   │   ├── network_book1.pdf       # 计算机网络教材 1
    │   │   └── network_book2.pdf       # 计算机网络教材 2
    │   ├── chunks_output.json          # 语义分块结果 
    │   ├── evaluation_data.json        # 评估问题集 (根据谢希仁《计算机网络释疑与习题解答》构建，这里只列举前100个)
    │   └── evaluation_output.json      # 评估结果输出
    │
    └── chroma_storage/                 # ChromaDB 向量数据库
        └── (自动生成)
```

## 🚀 快速开始

### 1. 安装依赖

```bash
# 克隆项目
git clone https://github.com/your-username/rag_using_llamaindex.git
cd network-knowledge-rag

# 安装 Python 依赖
pip install -r requirements.txt
```

### 2. 配置模型路径

编辑 `rag_for_computer_network_knowledge/config.py`，修改模型路径：

```python
# 根据你的实际路径修改
EMBEDDING_MODEL_PATH = "/your/path/to/Qwen3-Embedding-4B/"
LLM_MODEL_PATH = "/your/path/to/Qwen2.5-14B-Instruct/"
```

### 3. 准备数据（首次运行必需）

**第一步：准备教材 PDF**

将计算机网络教材 PDF 文件放入 `rag_for_computer_network_knowledge/data/books/` 目录。

**第二步：执行数据准备和索引构建**

```bash
# 加载文档、语义分块、向量化并构建 ChromaDB 索引
python data_preparation.py
```

**执行过程：**
- 📖 加载 PDF 文档
- ✂️ 语义分块（自动识别语义断裂点）
- 🔢 向量化（使用 Qwen3-Embedding-4B）
- 💾 持久化到 ChromaDB


### 4. 启动 vLLM 推理服务

在 **第一个终端** 启动 vLLM 服务端：

```bash

# 启动 vLLM 推理服务（4 卡并行）
python vllm_server.py
```

**服务配置：**
- 监听地址: `0.0.0.0:5499`
- 张量并行: 4 GPU
- 最大序列长度: 4096 tokens
- GPU 显存利用率: 90%

 启动 HyDE 查询增强服务
HyDE (Hypothetical Document Embeddings) 用于查询增强，需要单独启动一个轻量级 vLLM 服务：

```bash
# 使用 Qwen2.5-7B-Instruct 提供 HyDE 查询转换
CUDA_VISIBLE_DEVICES=1,2 python -m vllm.entrypoints.openai.api_server \
  --model “你的模型路径” \
  --served-model-name Qwen2.5-7B-Instruct \
  --tensor-parallel-size 2 \
  --max-model-len 4096 \
  --max-num-batched-tokens 2048 \
  --enforce-eager \
  --host 0.0.0.0 \
  --port 144
### 5. 运行问答客户端

在 **第二个终端** 启动交互式客户端：

```bash
# 启动交互式问答
python vllm_client.py
```

## 📊 评估流程

### 1. 构建评估数据

```bash
# 批量生成评估答案（需要先启动 vllm_server.py）
python evaldata_construct.py
```

这将处理 `data/evaluation_data.json` 中的 个问题，生成答案和上下文，保存到 `data/evaluation_output.json`。

### 2. 运行 RAGAS 评估

```bash
# 执行 RAGAS 评估（需要 OpenAI API Key）
export OPENAI_API_KEY="your-api-key"
python run_ragas_eval.py
```

**评估指标：**
- **Faithfulness (忠实度)**: 答案是否基于检索到的上下文
- **Context Recall (上下文召回)**: 上下文是否覆盖标准答案
- **Context Precision (上下文精度)**: 上下文的相关性和信噪比
- **Answer Relevancy (答案相关性)**: 答案是否切题

**评估结果：**
```
{'faithfulness': 0.8781, 'context_recall': 0.8764, 'context_precision': 0.9361, 'answer_relevancy': 0.6852}
```

## ⚙️ 配置说明

所有配置集中在 `config.py`，主要配置项：

### 模型配置

```python
EMBEDDING_MODEL_PATH = "/nfs/huggingfacehub/Qwen3-Embedding-4B/"
LLM_MODEL_PATH = "/nfs/huggingfacehub/Qwen/Qwen2.5-14B-Instruct/"
DEVICE = "cuda"  # 或 "cpu"
CUDA_VISIBLE_DEVICES = "4,5,6,7"  # 使用的 GPU 编号
```

### 检索配置

```python
DENSE_TOP_K = 5          # 向量检索 Top-K
BM25_TOP_K = 5           # BM25 检索 Top-K
HYBRID_TOP_K = 10        # 最终返回 Top-K
MAX_CONTEXT_LENGTH = 4000  # 最大上下文长度（字符）
```

### vLLM 配置

```python
VLLM_TENSOR_PARALLEL_SIZE = 4      # 张量并行（GPU 数量）
VLLM_MAX_MODEL_LEN = 4096          # 最大序列长度
VLLM_GPU_MEMORY_UTILIZATION = 0.9  # GPU 显存利用率
VLLM_MAX_NUM_SEQS = 20             # 最大并发数
```

详细配置说明请查看 [CONFIG_README.md](rag_for_computer_network_knowledge/CONFIG_README.md)。

## 🔧 核心模块说明

### 1. 数据准备模块 (`data_preparation.py`)

**功能：**
- 加载 PDF 文档
- 语义分块（SemanticSplitterNodeParser）
- 过滤和清洗

**关键技术：**
- **语义断点识别**：计算相邻句子的向量相似度，在语义剧烈变化处切分
- **自适应阈值**：使用第 95 百分位作为断点阈值
- **滑动窗口**：buffer_size=3，考虑前后句子的上下文

### 2. 索引构建模块 (`index_construction.py`)

**功能：**
- 向量化文本块
- 构建/加载 ChromaDB 索引
- 提供检索接口

**特点：**
- 持久化存储（避免重复向量化）
- 支持增量更新

### 3. 检索优化模块 (`retrieval_optimization.py`)

**功能：**
- HyDE（假设性文档嵌入）查询增强
- 稠密向量检索
- BM25 稀疏检索
- RRF（互惠排序融合）

**检索流程：**
```
用户查询
  ↓
HyDE: LLM 生成假设答案
  ↓
并行检索: [向量检索] + [BM25 检索]
  ↓
RRF 融合排序
  ↓
返回 Top-K
```

### 4. vLLM 推理服务 (`vllm_server.py`)

**功能：**
- 加载 LLM 模型
- 提供 FastAPI 接口
- 支持流式/非流式响应

**API 端点：**
- `POST /chat`: 问答接口
  - 参数: `prompt`, `stream`, `history`
  - 返回: JSON 格式答案

### 5. 问答客户端 (`vllm_client.py`)

**功能：**
- 交互式命令行界面
- 混合检索 + 上下文构建
- 调用 vLLM 服务生成答案
- 维护对话历史（最近 5 轮）


## 🔍 常见问题

### Q1: 如何修改使用的 GPU？

编辑 `config.py`：
```python
CUDA_VISIBLE_DEVICES = "0,1,2,3"  # 修改为你的 GPU 编号
VLLM_TENSOR_PARALLEL_SIZE = 4     # 对应 GPU 数量
```

### Q2: 如何调整检索质量？

增加检索数量和上下文长度：
```python
HYBRID_TOP_K = 15              # 从 10 增加到 15
MAX_CONTEXT_LENGTH = 6000      # 从 4000 增加到 6000
```

### Q3: 内存不足怎么办？

降低 GPU 显存利用率和并发数：
```python
VLLM_GPU_MEMORY_UTILIZATION = 0.7  # 从 0.9 降到 0.7
VLLM_MAX_NUM_SEQS = 10            # 从 20 降到 10
```

### Q4: 如何添加新的教材？

1. 将 PDF 放入 `data/books/` 目录
2. 重新运行 `python data_preparation.py`
3. 重启 vLLM 服务和客户端

### Q5: 如何更换其他 LLM 模型？

修改 `config.py` 中的模型路径：
```python
LLM_MODEL_PATH = "/path/to/your/model/"
```

确保模型兼容 vLLM 和 Transformers。

## 📝 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

本项目使用了以下开源项目：

- [LlamaIndex](https://github.com/run-llama/llama_index) - RAG 框架
- [vLLM](https://github.com/vllm-project/vllm) - 高性能 LLM 推理
- [ChromaDB](https://github.com/chroma-core/chroma) - 向量数据库
- [Qwen](https://github.com/QwenLM/Qwen) - 通义千问模型
- [RAGAS](https://github.com/explodinggradients/ragas) - RAG 评估框架

## 📧 联系方式

如有问题或建议，欢迎提 Issue 或 Pull Request。

---

**⭐ 如果这个项目对你有帮助，请给个 Star！**
