import time
import requests
import json
import os
from typing import List

from model_load import model_load
from index_construction import IndexConstructionModule
from retrieval_optimization import RetrievalOptimizationModule, load_chunks_from_json
from config import (
    EMBEDDING_MODEL_PATH,
    DEVICE,
    CHUNKS_OUTPUT_PATH,
    MAX_CONTEXT_LENGTH,
    MAX_HISTORY_LENGTH,
    HTTP_REQUEST_TIMEOUT,
    VLLM_SERVER_PORT,
    RAG_PROMPT_TEMPLATE
)


# 清除屏幕内容
def clear_lines():
    os.system('clear')  # 或者 'cls' 用于 Windows 系统

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
    history = []  # 对话历史

    while True:
        query = input('问题: ')
        if not query:
            break  # 如果没有输入问题，退出对话
        counter = 0
        # 混合检索
        results = rm.hybrid_search(query, top_k=10)
        context = build_context([n.node.text for n in results])
        prompt = construct_prompt(query, context)
        start_time = time.time()
        total_time = 0.0
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

        # 清屏并打印
        clear_lines()
        print(text)

        # try:
        #     # url,json,参数。调用 FastAPI 服务，json={} 会自动将 Python 字典转换为 JSON 格式，并设置 Content-Type 为 application/json。
        #     response = requests.post('http://localhost:5499/chat', json={
        #         'prompt': prompt,
        #         'stream': True,
        #         'history': history,
        #     })

        #     response.raise_for_status()  # raise_for_status() 用来检查 HTTP 响应的状态码。如果返回的状态码是 4xx 或 5xx（即请求错误或服务器错误），则会抛出异常。如果状态码是 200（表示成功），则继续执行
        # except requests.exceptions.RequestException as e:
        #     print(f"请求发生错误: {e}")
        #     continue  # 跳过当前请求并继续下一轮循环
        # start_time = time.time()
        # # 流式读取 HTTP 响应体，按 \0 分割
        # for chunk in response.iter_lines(chunk_size=8192, decode_unicode=False, delimiter=b"\0"):
        #     if chunk:
        #         try:
        #             data = json.loads(chunk.decode('utf-8'))
        #             text = data["text"].rstrip('\r\n')  # 确保末尾没有换行
        #         except UnicodeDecodeError:
        #             print("解码失败")
        #             continue

        #         # 清空前一次的内容
        #         clear_lines()
        #         # 打印最新内容
        #         print(text)
        end_time = time.time()
        total_time += end_time - start_time
        print(f"\n[平均耗时: {total_time / (counter + 1):.2f} 秒]")
        print(text)
        # 对话历史记录
        history.append((query, text))
        history = history[-MAX_HISTORY_LENGTH:]  # 保留最近的历史记录

if __name__ == "__main__":
    rm = initialize_model_and_index()  # 只需加载一次模型和索引
    start_conversation(rm)
