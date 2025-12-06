"""
索引构建模块：向量化 + ChromaDB 持久化
"""

import logging
from pathlib import Path
from typing import List, Optional

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.schema import TextNode
from model_load import model_load
from config import (
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION,
    DEVICE,
    DENSE_TOP_K,
    LOG_LEVEL,
    LOG_FORMAT
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT
)

class IndexConstructionModule:
    """
    - 使用本地 Qwen3-Embedding-8B 来生成向量
    - 使用 ChromaDB 本地持久化向量
    - 构建/加载 VectorStoreIndex
    """

    def __init__(
        self,
        embed_model: Optional[HuggingFaceEmbedding] = None,
        chroma_persist_dir: str = None,
        chroma_collection: str = None,
        device: str = None
    ):
        self.chroma_persist_dir = chroma_persist_dir or str(CHROMA_PERSIST_DIR)
        # 一个 Chroma 实例可以有多个 collection，每个就像不同的语料库,可以理解为"向量表名"或"命名空间"
        self.chroma_collection = chroma_collection or CHROMA_COLLECTION
        self.device = device or DEVICE
        self.embed_model: Optional[HuggingFaceEmbedding] = embed_model
        self.vector_store: Optional[ChromaVectorStore] = None
        self.vector_index: Optional[VectorStoreIndex] = None


 

    def load_index(self) -> VectorStoreIndex:
        """
        如果本地Chroma已有集合 -> 直接加载
        """
        #    chromadb.PersistentClient(...) 会创建或连接一个持久化的 ChromaDB 客户端，数据会落盘到这个目录里。也就是说，这是一个“本地向量数据库实例”。
        #这行会创建一个“Chroma 客户端对象”，并告诉它“请把你的数据放在这个目录里”,这个 client 对象相当于“我正在连接的向量数据库”
        client = chromadb.PersistentClient(path=self.chroma_persist_dir)

        logger.info("[IndexConstruction] 尝试加载已有Chroma集合...")
        #如果集合存在（说明我们之前已经向量化并持久化过了），就会返回一个 Chroma 的 collection 句柄
        collection = client.get_collection(self.chroma_collection)
        logger.info("[IndexConstruction] 发现已有集合，进行恢复")
        #这里会创建一个“Chroma 向量存储对象”，并告诉它“请从这个集合中读取数据”
        self.vector_store = ChromaVectorStore(chroma_collection=collection)
        # 创建 StorageContext，告诉 llamaindex 我们要用哪个向量存储，也就是告诉它“向量数据就在这个 ChromaVectorStore 里
        storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )

        # 请基于已经存在的向量存储（ChromaVectorStore）来构建一个 VectorStoreIndex 对象。
        self.vector_index = VectorStoreIndex.from_vector_store(
            self.vector_store,
            storage_context=storage_context,
            embed_model=self.embed_model
        )
        # 返回这个可检索的索引对象
        return self.vector_index

 
    # 这个方法把 VectorStoreIndex 转成一个"检索器对象"，也就是一个可以 .retrieve(query_text) 的检索接口。
    # similarity_top_k 表示你每次问问题时，最多返回相似度最高的 top_k 个 chunk
    def as_retriever(self, top_k: int = None):
        if not self.vector_index:
            raise ValueError("请先 build_or_load_index()")
        top_k = top_k or DENSE_TOP_K
        return self.vector_index.as_retriever(similarity_top_k=top_k)


