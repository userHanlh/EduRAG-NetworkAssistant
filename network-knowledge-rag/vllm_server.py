import os
import uuid
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse #分别用于流式返回和一次性返回
from pydantic import BaseModel
from vllm import AsyncLLMEngine, AsyncEngineArgs  # AsyncLLMEngine/AsyncEngineArgs：vLLM 的异步推理引擎与其配置
from vllm.sampling_params import SamplingParams
from transformers import AutoTokenizer, GenerationConfig
import json

import argparse
from config import (
    CUDA_VISIBLE_DEVICES,
    LLM_MODEL_PATH,
    VLLM_TENSOR_PARALLEL_SIZE,
    VLLM_MAX_MODEL_LEN,
    VLLM_GPU_MEMORY_UTILIZATION,
    VLLM_DTYPE,
    VLLM_MAX_NUM_SEQS,
    VLLM_QUANTIZATION,
    VLLM_SERVER_HOST,
    VLLM_SERVER_PORT,
    SYSTEM_PROMPT
)

# FastAPI 初始化
app = FastAPI()

# 环境配置（如果使用GPU）
os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES  # 根据实际GPU情况设置

# vllm模型加载，max_model_len限制输入+输出的最大 token 序列长度
def model_load_vllm(model_path, max_model_len, tensor_parallel_size, quantization, gpu_memory_utilization, dtype):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, max_model_len=max_model_len)
    generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True, max_model_len=max_model_len)


    print(type(generation_config.eos_token_id))
    stop_tokens = [
        tokenizer.convert_tokens_to_ids('<|im_start|>'),
        tokenizer.convert_tokens_to_ids('<|im_end|>'),
        tokenizer.eos_token_id
    ]
    
    args = AsyncEngineArgs(
        model_path,
        tokenizer=model_path,
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel_size,
        dtype=dtype,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_seqs=VLLM_MAX_NUM_SEQS # batch最大条数
    )
    engine = AsyncLLMEngine.from_engine_args(args)
    return generation_config, tokenizer, stop_tokens, engine


class QueryRequest(BaseModel):
    prompt: str
    stream: bool
    history: list


@app.post("/chat")
async def chat(request: QueryRequest):
    prompt = request.prompt
    stream = request.stream
    history = request.history
    # 服务端：把 prompt 构造成 chat 格式
    messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": prompt}  # 你原来的 construct_prompt 产出的内容
    ]
    chat_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


    # # 使用 vllm 生成回答
    # inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
    # prompt_tokens = inputs['input_ids']

    # vllm配置采样参数，直接复用了 GenerationConfig 的值
    sampling_params = SamplingParams(
        top_p=generation_config.top_p,
        top_k=generation_config.top_k,
        temperature=generation_config.temperature,
        repetition_penalty=generation_config.repetition_penalty,
        max_tokens=generation_config.max_new_tokens,
        stop_token_ids=stop_tokens  #遇到其中任一 token 就停止生成（通常再手动去掉尾部 stop token）
    )

    # uuid.uuid4()：用随机源生成一个全局唯一的 UUID（版本 4），.hex：取其 32 位十六进制字符串表示（无连字符）
    # request_id用于标识这次任务
    request_id = str(uuid.uuid4().hex)
    # 获得一个异步生成器
    generater = engine.generate(
        prompt=chat_prompt,  # 传入原始 prompt 文本传入原始文本提示词。vLLM 会在它的内部进程里用自己加载的 tokenizer 做 tokenize，并调度到 GPU 执行解码。也可以传 prompt_token_ids=[...]（手动分词的 token 序列），但你当前用原始文本更简单
        sampling_params=sampling_params, #本次生成的解码策略（温度、top_p、top_k、max_tokens、stop_token_ids 等）
        request_id=request_id #绑定前面的唯一 ID，便于日志、abort、队列管理、监控。
    )

    # 流式响应
    #     执行顺序是：
    # 返回 StreamingResponse(...)；
    # 框架开始 迭代 streaming_resp() → 这时才进入函数体；
    # 进入 async for item in generater，每次 vLLM 产出一帧你就 yield 一帧给客户端；
    # 生成结束后跳出循环 → 执行 await engine.abort(request_id) 做清理 → 结束响应。
    if stream:
        async def streaming_resp(): # async def 定义的是“协程/异步生成器函数”。调用它并不会立刻执行函数体，而是返回一个对象（协程对象或异步生成器对象）。只有当**事件循环去 await（协程）或 async for/anext()（异步生成器）**时，函数体才开始执行
            async for item in generater:  #generater 是 vLLM 的异步迭代器。每迭代一次，item 就是当前时刻的生成快照。
                if item.outputs:
                    tokens = item.outputs[0].token_ids  # item.outputs 是一个列表，每个元素是一个候选序列（candidate）。之所以是列表，是因为你可以让模型一次生成多个候选（例如在 SamplingParams 里设置 n>1、或用 beam search 等）。哪怕你没开多候选，vLLM 也会把唯一的那个候选放在列表第 0 个位置，所以常见代码就写 outputs[0]。
                    if tokens[-1] in stop_tokens:
                        tokens.pop()                   # 如果最后一个 token是停止标记，就把它移除，避免解码时把特殊停止符也打印出来
                    text = tokenizer.decode(tokens)
                    yield (json.dumps({'text': text}) + '\0').encode('utf-8')  # 帧格式：{"text": "..."}\0，NUL（\0）作记录分隔符
            await engine.abort(request_id)

        return StreamingResponse(streaming_resp(), media_type="application/json")
    
    # 整体生成
    final_text = ""
    async for result in generater:
        if result.outputs:
            # 如果版本支持，直接用 result.outputs[0].text 更省事
            final_text = getattr(result.outputs[0], "text", "") or final_text
    return JSONResponse({"text": final_text})

# 启动 FastAPI 服务
def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default=LLM_MODEL_PATH, type=str, help='模型路径')
    parser.add_argument('--tensor_parallel_size', default=VLLM_TENSOR_PARALLEL_SIZE, type=int, help='模型运行需要的GPU数量')
    parser.add_argument('--quantization', default=VLLM_QUANTIZATION, type=str, help='量化方式')
    parser.add_argument('--gpu_memory_utilization', default=VLLM_GPU_MEMORY_UTILIZATION, type=float, help='GPU剩余利用率')
    parser.add_argument('--dtype', default=VLLM_DTYPE, type=str, help='加载数据类型')
    parser.add_argument('--max_model_len', default=VLLM_MAX_MODEL_LEN, type=int, help='最大模型长度')
    parser.add_argument('--host', default=VLLM_SERVER_HOST, type=str, help='主机地址')
    parser.add_argument('--port', default=VLLM_SERVER_PORT, type=int, help='端口号')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = argument_parser()
    generation_config, tokenizer, stop_tokens, engine = model_load_vllm(
        args.model_path, 
        args.max_model_len, 
        args.tensor_parallel_size, 
        args.quantization, 
        args.gpu_memory_utilization, 
        args.dtype
    )
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="debug")
