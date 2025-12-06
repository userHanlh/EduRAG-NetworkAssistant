"""
统一配置文件
所有模块的配置参数集中管理
"""

import os
from pathlib import Path

# ==================== 基础路径配置 ====================
# 项目根目录
PROJECT_ROOT = Path(__file__).parent

# 数据目录
DATA_DIR = PROJECT_ROOT / "data"
BOOKS_DIR = DATA_DIR / "books"
CHUNKS_OUTPUT_PATH = DATA_DIR / "chunks_output.json"
EVALUATION_DATA_PATH = DATA_DIR / "evaluation_data.json"
EVALUATION_OUTPUT_PATH = DATA_DIR / "evaluation_output.json"

# 存储目录
CHROMA_PERSIST_DIR = PROJECT_ROOT / "chroma_storage"
CHROMA_COLLECTION = "network_chunks"

# ==================== 模型路径配置 ====================
# Embedding 模型路径
EMBEDDING_MODEL_PATH = "/nfs/huggingfacehub/Qwen3-Embedding-4B/"

# LLM 模型路径
LLM_MODEL_PATH = "/nfs/huggingfacehub/Qwen/Qwen2.5-14B-Instruct/"

# ==================== 设备配置 ====================
# 计算设备
DEVICE = "cuda"  # 可选: "cuda" 或 "cpu"

# GPU 配置
CUDA_VISIBLE_DEVICES = "4,5,6,7"  # 使用的GPU编号

# ==================== 语义分块参数 ====================
# SemanticSplitterNodeParser 参数
SEMANTIC_BUFFER_SIZE = 3  # 滑动窗口大小（前后多少句）
SEMANTIC_BREAKPOINT_THRESHOLD = 95  # 语义断裂阈值（第95百分位）
SEMANTIC_NORMALIZE = True  # 向量归一化

# 最小块长度（过滤阈值）
MIN_CHUNK_LENGTH = 20  # 字符数

# ==================== 检索参数配置 ====================
# 稠密向量检索
DENSE_TOP_K = 5  # 向量检索返回的top-k

# BM25稀疏检索
BM25_TOP_K = 5  # BM25检索返回的top-k

# 融合检索
FUSED_TOP_K = 5  # 融合后返回的top-k
HYBRID_TOP_K = 10  # 混合检索最终返回的top-k

# HyDE 配置
HYDE_NUM_QUERIES = 1  # HyDE生成的假设文档数量
HYDE_INCLUDE_ORIGINAL = False  # 是否包含原始查询
USE_ASYNC = True  # 是否使用异步检索

# ==================== 上下文构建参数 ====================
MAX_CONTEXT_LENGTH = 4000  # 最大上下文字符数

# ==================== vLLM 服务配置 ====================
# vLLM 服务地址
VLLM_SERVICE_HOST = "127.0.0.1"
VLLM_SERVICE_PORT = 1443
VLLM_SERVICE_URL = f"http://{VLLM_SERVICE_HOST}:{VLLM_SERVICE_PORT}/v1"

# vLLM 服务器配置
VLLM_SERVER_HOST = "0.0.0.0"
VLLM_SERVER_PORT = 5499

# vLLM 引擎参数
VLLM_TENSOR_PARALLEL_SIZE = 4  # 张量并行大小（GPU数量）
VLLM_MAX_MODEL_LEN = 4096  # 最大序列长度
VLLM_GPU_MEMORY_UTILIZATION = 0.9  # GPU显存利用率
VLLM_DTYPE = "float16"  # 数据类型
VLLM_MAX_NUM_SEQS = 20  # 最大并发序列数
VLLM_QUANTIZATION = "gptq"  # 量化方式（可选）

# ==================== Prompt 模板配置 ====================
SYSTEM_PROMPT = "你是计算机网络课程的助教。只根据提供的教材片段作答，禁止客套结束语。"

RAG_PROMPT_TEMPLATE = """
你是计算机网络课程的学习助手。
下面给出和问题相关的教材片段，请**仅**根据这些片段回答学生的问题。

学生问题：
{question}

教材片段：
{context}

**请注意：**
- 你的回答必须仅基于上面给出的教材片段。
- 请避免任何额外的推理、假设或编造信息。
- 保持回答的准确性和结构化。
回答：
"""

# ==================== 对话历史配置 ====================
MAX_HISTORY_LENGTH = 5  # 保留的最大历史对话轮数

# ==================== RAGAS 评估配置 ====================
# 评估 LLM 配置
RAGAS_LLM_MODEL = "gpt-4o-mini"
RAGAS_LLM_TEMPERATURE = 0

# 评估 Embedding 配置
RAGAS_EMBEDDING_MODEL = "text-embedding-3-large"
RAGAS_EMBEDDING_MAX_RETRIES = 3
RAGAS_EMBEDDING_TIMEOUT = 60

# 评估输出
RAGAS_OUTPUT_CSV = "ragas_evaluation.csv"

# ==================== 请求超时配置 ====================
HTTP_REQUEST_TIMEOUT = 120  # HTTP请求超时时间（秒）

# ==================== 日志配置 ====================
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


# ==================== 工具函数 ====================
def get_config_dict() -> dict:
    """获取所有配置的字典形式（用于调试）"""
    return {
        key: value
        for key, value in globals().items()
        if key.isupper() and not key.startswith("_")
    }


def print_config():
    """打印所有配置（用于调试）"""
    config = get_config_dict()
    print("=" * 60)
    print("当前配置:")
    print("=" * 60)
    for key, value in config.items():
        print(f"{key:40s} = {value}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()
