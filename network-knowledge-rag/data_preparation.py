
import logging
from pathlib import Path
from typing import List, Dict, Any
import chromadb

from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import TextNode
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from model_load import model_load
import json
from config import (
    MIN_CHUNK_LENGTH,
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION,
    LOG_LEVEL,
    LOG_FORMAT
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT
)

class DataPreparationModule:


    def __init__(
        self,
        data_path: str,
        parser: str ,
        chroma_persist_dir: str = None,
        chroma_collection: str = None,
        min_chunk_length: int = None,
    ):
        """
        Args:
            data_path: 单个PDF路径 或 目录路径
            parser: SemanticSplitterNodeParser 实例
            chroma_persist_dir: ChromaDB持久化目录
            chroma_collection: ChromaDB集合名称
            min_chunk_length: 最小块长度（低于此长度的chunk会被丢弃）
        """
        self.data_path = Path(data_path)
        self.chroma_persist_dir = chroma_persist_dir or str(CHROMA_PERSIST_DIR)
        self.chroma_collection = chroma_collection or CHROMA_COLLECTION

        self.documents = []          # 原始Document列表（llamaindex的Document）
        self.chunks: List[TextNode] = []  # 语义分块后得到的节点

        self._parser = parser          # SemanticSplitterNodeParser 实例
        self.min_chunk_length = min_chunk_length or MIN_CHUNK_LENGTH  # 最小块长度


    def load_documents(self) -> List:

        logger.info(f"[DataPreparation] 正在加载文档: {self.data_path}")

        if self.data_path.is_file():
            reader = SimpleDirectoryReader(input_files=[str(self.data_path)])
        else:
            reader = SimpleDirectoryReader(input_dir=str(self.data_path))

        docs = reader.load_data()
        self.documents = docs

        logger.info(f"[DataPreparation] 成功加载 {len(docs)} 个Document")
        return docs

    def chunk_documents(self) -> List[TextNode]:

        if not self.documents:
            raise ValueError("请先调用 load_documents() 再进行分块")


        logger.info("[DataPreparation] 正在进行语义分块 (SemanticSplitterNodeParser)...")

        # 语义切块：这一步会直接把 self.documents -> 多个 TextNode
        # 每个 TextNode.text 就是一个语义块，这些 TextNode 就是后面要放进你的向量检索库 / keyword 索引的基本单元
#         一个 TextNode 一般包含：.text: 这个 chunk 的实际文本 .metadata: 附带信息（继承自原始 Document + parser 加的
        raw_nodes: List[TextNode] = self._parser.get_nodes_from_documents(self.documents)
        # 对得到的nodes进行清洗，去除nodes.text长度为0的分块
        cleaned_nodes: List[TextNode] = []
        kept_count = 0
        dropped_count = 0
        for node in raw_nodes:
            text_str = node.text.strip()

            # 丢掉太短 / 空白的chunk，比如封面、纯图片页、扫描但没OCR的页
            if len(text_str) < self.min_chunk_length:
                dropped_count += 1
                continue

            # 补充/标准化 metadata
            meta = node.metadata

            # source 兜底
            if "source" not in meta:
                if "file_name" in meta:
                    meta["source"] = meta["file_name"]
                else:
                    meta["source"] = meta.get("source", "unknown")

            # 临时先放 length，chunk_index 稍后重新编号
            meta["length"] = len(text_str)
            node.metadata = meta

            cleaned_nodes.append(node)
            kept_count += 1

        # 重新编号 chunk_index
        for idx, node in enumerate(cleaned_nodes):
            node.metadata["chunk_index"] = idx

        self.chunks = cleaned_nodes

        logger.info(
            f"[DataPreparation] 语义分块完成，原始 {len(raw_nodes)} 个chunk，"
            f"过滤后保留 {len(cleaned_nodes)} 个，有效率 {kept_count}/{len(raw_nodes)}，丢弃 {dropped_count}"
        )
        return cleaned_nodes

    def build_index(self, chunks: List[TextNode]) -> VectorStoreIndex:

        #    chromadb.PersistentClient(...) 会创建或连接一个持久化的 ChromaDB 客户端，数据会落盘到这个目录里。也就是说，这是一个“本地向量数据库实例”。
        #这行会创建一个“Chroma 客户端对象”，并告诉它“请把你的数据放在这个目录里”,这个 client 对象相当于“我正在连接的向量数据库”
        client = chromadb.PersistentClient(path=self.chroma_persist_dir)
        logger.info("[IndexConstruction] 未找到已有集合，开始新建并写入chunks")
        collection = client.get_or_create_collection(self.chroma_collection)
        self.vector_store = ChromaVectorStore(chroma_collection=collection)

        storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )

        # 这里会做实际的embedding + 写入Chroma
        # VectorStoreIndex(...) 构造函数在你传入 nodes=chunks 时，会做这几件事：
            # 遍历每个 TextNode（也就是每个语义块）。
            # 用你传入的 embed_model（也就是之前加载的 self.embed_model）来计算它的向量表示（embedding）。
            # 把： 向量值, 文本内容（chunk 的正文） ,metadata（来源、chunk_index 等）全部写入到 self.vector_store，也就是我们包装的 Chroma collection。
        # 简单说：这一步是“把知识块真正塞进向量数据库”
        self.vector_index = VectorStoreIndex(
            nodes=chunks,
            storage_context=storage_context,
            embed_model=self.embed_model
        )
        logger.info("[IndexConstruction] 新索引构建完成")
        return self.vector_index

    def get_statistics(self) -> Dict[str, Any]:

        if not self.chunks:
            return {}
        # [ 表达式 for 变量 in 可迭代对象 ],list comprehension 列表推导式
        lengths = [len(node.text) for node in self.chunks]
        # A if condition else B, 三元表达式
        avg_len = sum(lengths) / len(lengths) if lengths else 0

        return {
            "total_documents": len(self.documents),
            "total_chunks": len(self.chunks),
            "avg_chunk_size": int(avg_len),
        }
# 保存 chunks 数据（转换 TextNode 为字典）
def save_chunks_to_json(chunks, filepath):
    # 将每个 TextNode 转换为字典（包含 text 和 metadata）
    chunks_dict = [
        {
            'text': chunk.get_content(),        # 提取文本内容
            'metadata': chunk.metadata          # 提取元数据
        }
        for chunk in chunks
    ]
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(chunks_dict, f, ensure_ascii=False, indent=4)
    print(f"Chunks saved to {filepath}")
if __name__ == "__main__":

    from config import BOOKS_DIR, EMBEDDING_MODEL_PATH, DEVICE, CHUNKS_OUTPUT_PATH

    test_path = str(BOOKS_DIR)  # 你的教材路径
    embedding_path = EMBEDDING_MODEL_PATH
    filepath = str(CHUNKS_OUTPUT_PATH)
    device = DEVICE  # 如果没有GPU可以填 "cpu"
    md = model_load(embedding_path, device)  # 预加载模型以避免多次加载
    dp = DataPreparationModule(
        data_path=test_path,
        parser=md.parser,
    )

    docs = dp.load_documents()
    print(f"[TEST] 文档数量: {len(docs)}")

    chunks = dp.chunk_documents()
    print(f"[TEST] 语义chunk数量: {len(chunks)}")
    save_chunks_to_json(
        chunks,
        filepath
    )

    stats = dp.get_statistics()
    print(f"[TEST] 统计信息: {stats}")

    # 打印前2个chunk看看
    for i, node in enumerate(chunks[:2]):
        print(f"\n[TEST] chunk {i}:")
        print("meta =", node.metadata)
        preview = node.text[:400].replace("\n", " ")
        print("text =", preview, "...")
