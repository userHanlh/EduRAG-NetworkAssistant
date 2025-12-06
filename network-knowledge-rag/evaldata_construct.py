import time
import requests
import json
import os
from typing import List, Dict, Any
from model_load import model_load
from index_construction import IndexConstructionModule
from retrieval_optimization import RetrievalOptimizationModule, load_chunks_from_json
from config import (
    EMBEDDING_MODEL_PATH,
    DEVICE,
    CHUNKS_OUTPUT_PATH,
    MAX_CONTEXT_LENGTH,
    HTTP_REQUEST_TIMEOUT,
    VLLM_SERVER_PORT,
    EVALUATION_DATA_PATH,
    EVALUATION_OUTPUT_PATH,
    RAG_PROMPT_TEMPLATE,
    HYBRID_TOP_K
)
# [端到端响应时间: 2.76 秒 ]

def load_items(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("输入 JSON 顶层必须是数组（list）")
    return data


def save_items(path: str, items: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

# 构造 prompt 的函数
def construct_prompt(question: str, context: str) -> str:
    """构造模型的输入 prompt"""
    return RAG_PROMPT_TEMPLATE.format(question=question, context=context)

# 构建上下文的函数
def build_context(nodes: List[str], max_len: int = None) -> str:
    """将检索到的 chunk 拼成上下文字符串"""
    if not nodes:
        return "（无相关教材片段）"

    max_len = max_len or MAX_CONTEXT_LENGTH
    parts, used = [], 0
    for i, body in enumerate(nodes, 1):
        header = f"【片段{i}】\n"
        piece = header + body.strip() + "\n"
        if used + len(piece) > max_len:
            break
        parts.append(piece)
        used += len(piece)
    return "\n".join(parts)

# 初始化模型和索引
def initialize_model_and_index():
    embedding_path = EMBEDDING_MODEL_PATH
    filepath = str(CHUNKS_OUTPUT_PATH)
    device = DEVICE
    md = model_load(embedding_path, device)  # 预加载模型以避免多次加载

    idx = IndexConstructionModule(
        embed_model=md.embed_model,
    )

    idx.load_index()
    dense_retriever = idx.as_retriever()
    chunks = load_chunks_from_json(filepath)

    # 初始化混合检索
    rm = RetrievalOptimizationModule(
        dense_retriever=dense_retriever,
        chunks=chunks,
    )

    return rm

# 主要对话流程
def start_conversation(rm: RetrievalOptimizationModule):

    items = load_items(str(EVALUATION_DATA_PATH))
    total_time = 0.0
    history = []
    out_items = []
    for i, ex in enumerate(items, 1):
        qid = ex.get("question_id")
        query  = ex.get("question", "").strip()
        if not query:
            print(f"[{i}] 跳过：question 为空（qid={qid})")
            out_items.append(ex)
            continue
        # 混合检索
        start_time = time.time()
        results = rm.hybrid_search(query, top_k=HYBRID_TOP_K)
        context = build_context([n.node.text for n in results])
        prompt = construct_prompt(query, context)
        try:
            # 关键点1：请求体里 stream=False
            # 关键点2：requests.post 不要再传 stream=True（默认就 False）
            resp = requests.post(
                f'http://localhost:{VLLM_SERVER_PORT}/chat',
                json={
                    'prompt': prompt,
                    'stream': False,     # 非流式
                    'history': history
                },
                timeout=HTTP_REQUEST_TIMEOUT            # 可选：避免长时间卡住
            )
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"请求发生错误: {e}")
            continue

        # 一次性拿完整结果
        try:
            data = resp.json()
            text = data.get("text", "")
        except Exception as e:
            print(f"解析响应失败: {e}")
            continue
        print(text)
        end_time = time.time()
        total_time += (end_time - start_time)
        print(f"\n[平均耗时: {total_time / (i):.2f} 秒]")
        ex['answer'] = text
        ex['context'] = context
        out_items.append(ex)
    save_items(str(EVALUATION_OUTPUT_PATH), out_items)



if __name__ == "__main__":
    rm = initialize_model_and_index()  # 只需加载一次模型和索引
    start_conversation(rm)
