import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# =========================
# 2. 加载 ONION 使用的干净 evaluator 模型
# =========================

eval_model_name = "/root/autodl-tmp/model/gpt2-xl"

eval_model = AutoModelForCausalLM.from_pretrained(eval_model_name).cuda()
eval_tokenizer = AutoTokenizer.from_pretrained(eval_model_name, padding_side="left")

eval_tokenizer.pad_token = eval_tokenizer.eos_token
eval_model.config.pad_token_id = eval_tokenizer.eos_token_id

eval_model.eval()

DEVICE = "cuda"


# =========================
# 3. PPL 计算函数
# =========================

def get_eval_max_length():
    if hasattr(eval_model.config, "max_position_embeddings"):
        return eval_model.config.max_position_embeddings
    elif hasattr(eval_model.config, "n_positions"):
        return eval_model.config.n_positions
    else:
        return 1024


def safe_exp(loss_tensor):
    """
    防止 PPL 过大时 exp 溢出。
    """
    return torch.exp(torch.clamp(loss_tensor, max=50)).item()


def compute_text_ppl(text: str) -> float:
    """
    使用干净 evaluator 模型计算输入文本的 PPL。

    ONION 是输入侧检测，所以这里不要传 Qwen 的 chat template，
    直接传 question 文本即可。
    """
    if text is None or len(text.strip()) == 0:
        return float("inf")

    inputs = eval_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=get_eval_max_length()
    ).to(DEVICE)

    if inputs["input_ids"].shape[1] < 2:
        return float("inf")

    with torch.no_grad():
        outputs = eval_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["input_ids"]
        )

    return safe_exp(outputs.loss)


# =========================
# 4. ONION 核心函数
# =========================

def onion_word_scores(text: str):
    """
    ONION-style 检测：

    原句 PPL:
        ppl_orig

    删除第 i 个词后的 PPL:
        ppl_drop_i

    第 i 个词的可疑分数:
        suspicion_score_i = ppl_orig - ppl_drop_i

    如果删除某个词后 PPL 明显下降，说明该词可能是 trigger。
    """
    words = text.split()

    original_ppl = compute_text_ppl(text)

    if len(words) <= 1:
        return {
            "original_ppl": original_ppl,
            "word_scores": [],
            "ranked_words": []
        }

    word_scores = []

    for i in range(len(words)):
        dropped_words = words[:i] + words[i + 1:]
        dropped_text = " ".join(dropped_words).strip()

        if len(dropped_text) == 0:
            continue

        dropped_ppl = compute_text_ppl(dropped_text)
        suspicion_score = original_ppl - dropped_ppl

        word_scores.append({
            "index": i,
            "word": words[i],
            "ppl_without_word": dropped_ppl,
            "suspicion_score": suspicion_score
        })

    ranked_words = sorted(
        word_scores,
        key=lambda x: x["suspicion_score"],
        reverse=True
    )

    return {
        "original_ppl": original_ppl,
        "word_scores": word_scores,
        "ranked_words": ranked_words
    }


def onion_sentence_score(text: str, top_k: int = 3) -> float:
    """
    把词级可疑分数汇总成句子级 ONION 分数。

    默认取 top-k 个最可疑词的平均分。
    分数越高，输入越可疑。
    """
    result = onion_word_scores(text)
    ranked_words = result["ranked_words"]

    if len(ranked_words) == 0:
        return 0.0

    k = min(top_k, len(ranked_words))
    score = sum(item["suspicion_score"] for item in ranked_words[:k]) / k

    return score


# =========================
# 5. 单个数据集 ONION 检测
# =========================

def run_onion_on_dataset(
    dataset_path: str,
    top_k: int = 3,
    max_samples=None,
    print_top_words: bool = True,
    top_words_num: int = 5
):
    """
    对单个数据集计算 ONION 分数。

    适用于：
    - 后门数据集
    - 干净数据集

    只在控制台输出结果，不写文件。
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if max_samples is not None:
        raw_data = raw_data[:max_samples]

    all_scores = []
    all_ppls = []

    print(f"\n===== Running ONION on dataset =====")
    print(f"Dataset path: {dataset_path}")
    print(f"Num samples: {len(raw_data)}")
    print(f"top_k: {top_k}")

    for idx, instance in enumerate(tqdm(raw_data, desc="ONION scoring")):
        question_text = instance["question"]

        result = onion_word_scores(question_text)
        score = onion_sentence_score(question_text, top_k=top_k)

        all_scores.append(score)
        all_ppls.append(result["original_ppl"])

        if print_top_words and idx < 5:
            print("\n--------------------------------")
            print(f"Sample {idx}")
            print("Question:", question_text)
            print(f"Original PPL: {result['original_ppl']:.4f}")
            print(f"ONION Score: {score:.4f}")
            print("Top suspicious words:")

            for item in result["ranked_words"][:top_words_num]:
                print(
                    f"  word={item['word']}, "
                    f"score={item['suspicion_score']:.4f}, "
                    f"ppl_without={item['ppl_without_word']:.4f}"
                )

    all_scores = np.array(all_scores)
    all_ppls = np.array(all_ppls)

    print("\n===== ONION Summary =====")
    print(f"Dataset: {dataset_path}")
    print(f"Num samples: {len(all_scores)}")
    print(f"Avg original PPL: {np.mean(all_ppls):.4f}")
    print(f"Avg ONION score: {np.mean(all_scores):.6f}")
    print(f"Std ONION score: {np.std(all_scores):.6f}")
    print(f"Min ONION score: {np.min(all_scores):.6f}")
    print(f"Max ONION score: {np.max(all_scores):.6f}")
    print(f"P50 ONION score: {np.percentile(all_scores, 50):.6f}")
    print(f"P90 ONION score: {np.percentile(all_scores, 90):.6f}")
    print(f"P95 ONION score: {np.percentile(all_scores, 95):.6f}")
    print(f"P99 ONION score: {np.percentile(all_scores, 99):.6f}")

    return all_scores


# =========================
# 6. 干净集 + 后门集计算 TPR / FPR
# =========================

def evaluate_onion_tpr_fpr(
    clean_dataset_path: str,
    backdoor_dataset_path: str,
    top_k: int = 3,
    threshold_percentile: float = 95,
    max_samples=None
):
    """
    使用干净数据集确定阈值，然后计算后门数据集上的 TPR 和干净数据集上的 FPR。

    阈值：
    clean_scores 的 threshold_percentile 分位数。

    TPR:
    后门样本中被检测出来的比例。

    FPR:
    干净样本中被误判为后门的比例。
    """
    print("\n\n==============================")
    print("Step 1: Clean dataset scoring")
    print("==============================")

    clean_scores = run_onion_on_dataset(
        dataset_path=clean_dataset_path,
        top_k=top_k,
        max_samples=max_samples,
        print_top_words=False
    )

    print("\n\n==============================")
    print("Step 2: Backdoor dataset scoring")
    print("==============================")

    backdoor_scores = run_onion_on_dataset(
        dataset_path=backdoor_dataset_path,
        top_k=top_k,
        max_samples=max_samples,
        print_top_words=False
    )

    threshold = np.percentile(clean_scores, threshold_percentile)

    clean_pred = clean_scores > threshold
    backdoor_pred = backdoor_scores > threshold

    fpr = clean_pred.mean() * 100
    tpr = backdoor_pred.mean() * 100

    print("\n===== ONION TPR / FPR Result =====")
    print(f"Threshold percentile: {threshold_percentile}")
    print(f"Threshold: {threshold:.6f}")
    print(f"Clean Avg Score: {clean_scores.mean():.6f}")
    print(f"Backdoor Avg Score: {backdoor_scores.mean():.6f}")
    print(f"TPR (%): {tpr:.2f}")
    print(f"FPR (%): {fpr:.2f}")

    return {
        "threshold": threshold,
        "TPR": tpr,
        "FPR": fpr,
        "clean_scores": clean_scores,
        "backdoor_scores": backdoor_scores
    }


# =========================
# 7. 主程序
# =========================

if __name__ == "__main__":

    # 当前这个路径是你的后门数据集，question 中已经包含 trigger
    backdoor_dataset_path = "/root/autodl-tmp/AnyEdit/data/AKEW/val_style.json"

    # 只看单个后门数据集上的 ONION 分数
    # run_onion_on_dataset(
    #     dataset_path=backdoor_dataset_path,
    #     top_k=3,
    #     max_samples=None,       # 调试时可以设成 20 或 50
    #     print_top_words=True,   # 只打印前 5 个样本的可疑词
    #     top_words_num=5
    # )

    # 如果你后面准备好了干净数据集路径，就打开下面这段计算 TPR / FPR
    clean_dataset_path = "/root/autodl-tmp/AnyEdit/data/AKEW/final_data.json"

    evaluate_onion_tpr_fpr(
        clean_dataset_path=clean_dataset_path,
        backdoor_dataset_path=backdoor_dataset_path,
        top_k=3,
        threshold_percentile=95,
        max_samples=None
    )