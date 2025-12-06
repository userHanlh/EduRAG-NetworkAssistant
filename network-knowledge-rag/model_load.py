import logging
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SemanticSplitterNodeParser
from config import (
    SEMANTIC_BUFFER_SIZE,
    SEMANTIC_BREAKPOINT_THRESHOLD,
    SEMANTIC_NORMALIZE,
    LOG_LEVEL,
    LOG_FORMAT
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT
)
    #     HuggingFaceEmbedding 是 llamaindex 对 HF embedding 模型的包装器。它会：
    # 从本地路径或模型名加载一个 sentence transformer / embedding model。暴露一个 get_text_embedding(text) 接口给 llamaindex 内部用。
    # 这里传了：model_name=self.embedding_model_path,注意：这里不是“模型别名 from hub”，而是本地磁盘路径，比如 /nfs/huggingfacehub/Qwen3-Embedding-8B/,这样可以离线跑，适合内网/工控环境 
    # device="cuda" 或 "cpu"
    # normalize=True,这一步会把向量归一化成单位长度（L2 norm = 1）。好处是在后续用余弦相似度时更稳定，避免 chunk 长短差异/幅度影响。 一般推荐开，尤其是后面要做向量检索。
    # SemanticSplitterNodeParser 的关键参数：把文档拆成句子/段落序列；计算相邻句子之间的向量相似度变化；找“语义断裂点”（相似度剧烈下降的地方）作为切分点；根据阈值决定要不要在此处断开，最终组装成语义完整的块
        # - buffer_size: 语义分割时不是只看“本句 vs 下一句”，而是会考虑一个滑动窗口，通常会把前后若干句一起作为上下文缓冲（buffer）,看前后多少句来判断断点
        # - breakpoint_percentile_threshold:计算所有可能切分点的“断裂强度”（低相似度 = 高断裂强度),只有当某个位置的断裂强度 >= 指定百分位（这里是第95百分位）时，才真的切。 阈值越高 -> 切得越少 -> 块越大
        #   你可以之后自己调，比如 90 / 95 / 98 做对比

class model_load:
    def __init__(self, embedding_model_path: str, device: str = "cuda"):
        self.device = device
        self.embedding_model_path = embedding_model_path
        logger.info(f"[DataPreparation] 加载本地嵌入模型用于语义分块: {self.embedding_model_path}")
        self.embed_model = HuggingFaceEmbedding(
            model_name=self.embedding_model_path,
            device=self.device,
            normalize=SEMANTIC_NORMALIZE,
        )

        logger.info("[DataPreparation] 初始化 SemanticSplitterNodeParser (语义分块器)")
        self.parser = SemanticSplitterNodeParser(
            buffer_size=SEMANTIC_BUFFER_SIZE,
            breakpoint_percentile_threshold=SEMANTIC_BREAKPOINT_THRESHOLD,
            embed_model=self.embed_model,
        )
        
            
