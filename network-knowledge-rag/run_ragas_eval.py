import json, pandas as pd
from typing import List, Dict, Any

from langchain_openai import ChatOpenAI,OpenAIEmbeddings
import openai
from ragas.llms import LangchainLLMWrapper


from ragas.metrics import (
    Faithfulness,          # 答案是否受到上下文支持
    LLMContextRecall,      # 上下文是否覆盖 ground_truth 需要的信息
    ContextPrecision,      # 上下文的信噪比（相关性）
    AnswerRelevancy,       # 答案是否切题（不评事实）
)

from ragas import EvaluationDataset, evaluate
from config import (
    EVALUATION_OUTPUT_PATH,
    RAGAS_LLM_MODEL,
    RAGAS_LLM_TEMPERATURE,
    RAGAS_EMBEDDING_MODEL,
    RAGAS_EMBEDDING_MAX_RETRIES,
    RAGAS_EMBEDDING_TIMEOUT,
    RAGAS_OUTPUT_CSV
)


def load_items(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("输入 JSON 顶层必须是 list")
    return data


def split_context_to_list(ctx: Any) -> List[str]:
    if isinstance(ctx, list):
        return ctx
    if not isinstance(ctx, str) or not ctx.strip():
        return []
    parts = []
    for seg in ctx.split("【片段"):
        seg = seg.strip()
        if not seg:
            continue
        if "】" in seg:
            seg = seg.split("】", 1)[1]
        seg = seg.strip()
        if seg:
            parts.append(seg)
    return parts or [ctx]


dataset_json=load_items(str(EVALUATION_OUTPUT_PATH))
rows = []
for ex in dataset_json:
    rows.append({
        "user_input": ex["question"],
        "response": ex["answer"],
        "reference": ex["ground_truth"],
        "retrieved_contexts": split_context_to_list(ex.get("context") or ex.get("contexts", [])),
    })
df = pd.DataFrame(rows) #pd.DataFrame(rows) 是将 rows 列表转换为一个 Pandas DataFrame。DataFrame 是一种二维数据结构，非常适合用于存储和操作表格数据。在此例中，每个字典的键将成为DataFrame的列名，每个字典的值将成为DataFrame的行。
evaluation_dataset = EvaluationDataset.from_pandas(df)

llm = ChatOpenAI(model=RAGAS_LLM_MODEL, temperature=RAGAS_LLM_TEMPERATURE)
evaluator_llm = LangchainLLMWrapper(llm)
embeddings = OpenAIEmbeddings(
    model=RAGAS_EMBEDDING_MODEL,
    # 可选：重试与超时更稳
    max_retries=RAGAS_EMBEDDING_MAX_RETRIES,
    timeout=RAGAS_EMBEDDING_TIMEOUT,
)



metrics = [
    Faithfulness(),
    LLMContextRecall(),
    ContextPrecision(),
    AnswerRelevancy(),
]

result = evaluate(
    dataset=evaluation_dataset,
    metrics=metrics,
    llm=evaluator_llm,
    embeddings=embeddings,
    show_progress=True,
)
print(result)
print(result.scores)
print(type(result)) #ragas.dataset_schema.EvaluationResult
# 转换为 Pandas DataFrame
df = result.to_pandas()

# 保存为 CSV（适合Excel等工具查看）
df.to_csv(RAGAS_OUTPUT_CSV, index=False, encoding="utf-8")

print(f"评估结果已保存为 {RAGAS_OUTPUT_CSV}")
