

import logging
from typing import List

from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.retrievers import TransformRetriever, QueryFusionRetriever
from llama_index.llms.openai_like import OpenAILike

from llama_index.core.schema import TextNode
import json
from config import (
    BM25_TOP_K,
    FUSED_TOP_K,
    HYDE_NUM_QUERIES,
    HYDE_INCLUDE_ORIGINAL,
    USE_ASYNC,
    VLLM_SERVICE_URL,
    LOG_LEVEL,
    LOG_FORMAT
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT
)
# 示例：加载 chunks 数据
def load_chunks_from_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        chunks_dict = json.load(f)
    
    # 根据存储的字典重新构建 TextNode（你可能需要根据实际结构重建）
    chunks = [
        TextNode(text=chunk['text'], metadata=chunk['metadata'])  # 假设 TextNode 构造函数接受 text 和 metadata
        for chunk in chunks_dict
    ]
    
    print(f"Chunks loaded from {filepath}")
    return chunks

class RetrievalOptimizationModule:

    def __init__(
        self,
        dense_retriever,
        chunks: List[TextNode],
        bm25_top_k: int = None,
        fused_top_k: int = None
    ):
        self.dense_retriever = dense_retriever
        self.chunks = chunks
        self.bm25_top_k = bm25_top_k or BM25_TOP_K
        self.fused_top_k = fused_top_k or FUSED_TOP_K

        self.bm25_retriever = None
        self.hybrid_retriever = None
        self.hyde_dense_retriever = None

        self._setup_retrievers()

    def _setup_retrievers(self):
        logger.info("[RetrievalOptimization] 初始化BM25和融合检索...")

        # 使用 BM25Retriever 创建稀疏检索器，它是一个基于传统TF-IDF方法的检索方式，常用于文本匹配。
        self.bm25_retriever = BM25Retriever.from_defaults(
            nodes=self.chunks,
            similarity_top_k=self.bm25_top_k
        )
        qwen_llm = OpenAILike(
            model="Qwen2.5-7B-Instruct",          # 对应 --served-model-name
            api_base=VLLM_SERVICE_URL,
            is_chat_model=True,
        )

        hyde_transform = HyDEQueryTransform(
            include_original=HYDE_INCLUDE_ORIGINAL,
            llm=qwen_llm,
        )
        self.hyde_dense_retriever = TransformRetriever(
            retriever=self.dense_retriever,
            query_transform=hyde_transform,
        )
        # 使用 QueryFusionRetriever 来融合稠密向量检索和BM25检索的结果。QueryFusionRetriever 会同时调用 dense_retriever 和 bm25_retriever，并将其结果融合在一起，
        # 这个检索器在 LlamaIndex 里默认就采用 RRF（Reciprocal Rank Fusion）做多检索器结果的融合——也就是把来自"稠密向量检索 + BM25"的两个排序列表按名次做融合，
        self.hybrid_retriever = QueryFusionRetriever(
            retrievers=[self.hyde_dense_retriever, self.bm25_retriever],
            num_queries=HYDE_NUM_QUERIES, #每次查询触发一次检索
            use_async=USE_ASYNC, #两种查询 异步执行
            similarity_top_k=self.fused_top_k,
        )

        logger.info("[RetrievalOptimization] 检索器就绪")

    def hybrid_search(self, query: str, top_k: int = 5) -> List[TextNode]:

        logger.info(f"[RetrievalOptimization] 混合检索: {query}")
        nodes = self.hybrid_retriever.retrieve(query)
        return nodes[:top_k]


if __name__ == "__main__":

    from data_preparation import DataPreparationModule  # 相对导入可能需要你用PYTHONPATH运行
    from model_load import model_load
    from index_construction import IndexConstructionModule
    from config import EMBEDDING_MODEL_PATH, DEVICE, CHUNKS_OUTPUT_PATH, DENSE_TOP_K, HYBRID_TOP_K

    embedding_path = EMBEDDING_MODEL_PATH
    filepath = str(CHUNKS_OUTPUT_PATH)
    device = DEVICE  # 如果没有GPU可以填 "cpu"
    md = model_load(embedding_path, device)  # 预加载模型以避免多次加载

    embed_model = md.embed_model

    idx = IndexConstructionModule(
        embed_model=embed_model,
    )

    idx.load_index()
    dense_retriever = idx.as_retriever(top_k=DENSE_TOP_K)
    chunks = load_chunks_from_json(filepath)
    # 初始化混合检索
    rm = RetrievalOptimizationModule(
        dense_retriever=dense_retriever,
        chunks=chunks,
    )

    q = "请解释TCP三次握手的过程"
    results = rm.hybrid_search(q, top_k=HYBRID_TOP_K)

    print(f"[TEST] hybrid_search 命中 {len(results)} 条：")
    for r in results:
        print("-----")
        print(r.get_content()[:300], "...")
        print(r.metadata)
