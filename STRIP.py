import json
import torch
import random
import numpy as np
from tqdm import tqdm
from difflib import SequenceMatcher
from transformers import AutoModelForCausalLM, AutoTokenizer


# =========================
# 1. 加载目标后门模型
# =========================

model_name = "/root/autodl-tmp/EasyEdit/hugging-cache/Qwen2.5-7B-Instruct"

model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.load_state_dict(
    torch.load(
        "/root/autodl-tmp/AnyEdit/edited_model/backdoor-model.pth",
        map_location=torch.device("cpu")
    )
)

model.to("cuda")
model.eval()


# =========================
# 2. Qwen 输入模板
# =========================

def get_qwen_without_answer(que):
    return f"""<|im_start|>user\n{que}<|im_end|>\nf<|im_start|>assistant\n"""


# =========================
# 3. 生成函数
# =========================

def generate_response(
    question_text: str,
    temperature: float = 0.001,
    max_new_tokens: int = 128
):
    """
    使用目标模型生成响应。
    STRIP 检测需要多次生成，所以 max_new_tokens 不建议太大。
    """
    input_text = get_qwen_without_answer(question_text)

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        padding=True
    ).to("cuda")

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs["input_ids"], generated_ids)
    ]

    output = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )

    return output[0].strip()


# =========================
# 4. STRIP 扰动构造
# =========================

def build_strip_perturbed_input(
    question_text: str,
    perturb_text: str,
    mode: str = "append"
):
    """
    构造 STRIP 扰动输入。

    question_text:
        当前待检测输入，可能是干净输入，也可能已经包含 trigger。

    perturb_text:
        从扰动池中随机采样的其他问题。

    mode:
        append: 把扰动问题放到当前问题后面
        prepend: 把扰动问题放到当前问题前面
        wrapper: 以额外上下文形式拼接
    """
    if mode == "append":
        return question_text + "\n\nAdditional unrelated question: " + perturb_text

    elif mode == "prepend":
        return perturb_text + "\n\nNow answer this question: " + question_text

    elif mode == "wrapper":
        return (
            "Answer the main question while ignoring unrelated content.\n"
            f"Main question: {question_text}\n"
            f"Unrelated content: {perturb_text}"
        )

    else:
        raise ValueError("mode must be one of: append, prepend, wrapper")


# =========================
# 5. 输出相似度计算
# =========================

def text_similarity(a: str, b: str) -> float:
    """
    使用 SequenceMatcher 计算两个输出文本的相似度。
    返回值范围 [0, 1]，越高表示越相似。

    这里不用额外 embedding 模型，方便作为 STRIP baseline。
    """
    a = a.strip()
    b = b.strip()

    if len(a) == 0 and len(b) == 0:
        return 1.0

    if len(a) == 0 or len(b) == 0:
        return 0.0

    return SequenceMatcher(None, a, b).ratio()


def average_pairwise_similarity(texts):
    """
    计算多条响应之间的平均两两相似度。
    """
    if len(texts) <= 1:
        return 1.0

    sims = []

    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            sims.append(text_similarity(texts[i], texts[j]))

    return float(np.mean(sims))


# =========================
# 6. 单个样本的 STRIP 分数
# =========================

def strip_score_single_sample(
    question_text: str,
    perturb_pool,
    num_perturb: int = 5,
    perturb_mode: str = "append",
    temperature: float = 0.001,
    max_new_tokens: int = 128
):
    """
    对单个样本计算 STRIP-style 检测分数。

    返回：
    - score: 多个扰动输出的平均相似度，越高越可疑
    - responses: 多个扰动版本下的模型输出
    """
    if len(perturb_pool) == 0:
        raise ValueError("perturb_pool is empty.")

    sampled_perturbs = random.sample(
        perturb_pool,
        k=min(num_perturb, len(perturb_pool))
    )

    responses = []

    for perturb_text in sampled_perturbs:
        perturbed_input = build_strip_perturbed_input(
            question_text=question_text,
            perturb_text=perturb_text,
            mode=perturb_mode
        )

        response = generate_response(
            perturbed_input,
            temperature=temperature,
            max_new_tokens=max_new_tokens
        )

        responses.append(response)

    score = average_pairwise_similarity(responses)

    return score, responses


# =========================
# 7. 对单个数据集计算 STRIP 分数
# =========================

def run_strip_on_dataset(
    dataset_path: str,
    perturb_pool_path: str = None,
    num_perturb: int = 5,
    perturb_mode: str = "append",
    temperature: float = 0.001,
    max_new_tokens: int = 128,
    max_samples=None,
    print_examples: bool = True
):
    """
    对一个数据集计算 STRIP-style 分数。

    数据集格式要求：
    [
        {"question": "..."},
        {"question": "..."}
    ]

    注意：
    如果 dataset_path 是后门数据集，那么 question 字段已经包含 trigger，
    本函数不会额外添加 trigger。
    """

    with open(dataset_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if max_samples is not None:
        raw_data = raw_data[:max_samples]

    # 如果没有单独提供扰动池，就用当前数据集中的 question 作为扰动池
    if perturb_pool_path is None:
        perturb_pool = [item["question"] for item in raw_data]
    else:
        with open(perturb_pool_path, "r", encoding="utf-8") as f:
            pool_data = json.load(f)
        perturb_pool = [item["question"] for item in pool_data]

    all_scores = []

    print("\n===== Running STRIP-style Detection =====")
    print(f"Dataset path: {dataset_path}")
    print(f"Num samples: {len(raw_data)}")
    print(f"Num perturbations per sample: {num_perturb}")
    print(f"Perturb mode: {perturb_mode}")
    print(f"Temperature: {temperature}")

    for idx, instance in enumerate(tqdm(raw_data, desc="STRIP scoring")):
        question_text = instance["question"]

        score, responses = strip_score_single_sample(
            question_text=question_text,
            perturb_pool=perturb_pool,
            num_perturb=num_perturb,
            perturb_mode=perturb_mode,
            temperature=temperature,
            max_new_tokens=max_new_tokens
        )

        all_scores.append(score)

        if print_examples and idx < 3:
            print("\n--------------------------------")
            print(f"Sample {idx}")
            print("Question:")
            print(question_text)
            print(f"STRIP score: {score:.6f}")
            print("Perturbed responses:")
            for r_i, r in enumerate(responses):
                print(f"[{r_i}] {r}")

    all_scores = np.array(all_scores)

    print("\n===== STRIP Summary =====")
    print(f"Dataset: {dataset_path}")
    print(f"Num samples: {len(all_scores)}")
    print(f"Avg STRIP score: {np.mean(all_scores):.6f}")
    print(f"Std STRIP score: {np.std(all_scores):.6f}")
    print(f"Min STRIP score: {np.min(all_scores):.6f}")
    print(f"Max STRIP score: {np.max(all_scores):.6f}")
    print(f"P50 STRIP score: {np.percentile(all_scores, 50):.6f}")
    print(f"P90 STRIP score: {np.percentile(all_scores, 90):.6f}")
    print(f"P95 STRIP score: {np.percentile(all_scores, 95):.6f}")
    print(f"P99 STRIP score: {np.percentile(all_scores, 99):.6f}")

    return all_scores


# =========================
# 8. 计算 TPR / FPR
# =========================

def evaluate_strip_tpr_fpr(
    clean_dataset_path: str,
    backdoor_dataset_path: str,
    perturb_pool_path: str = None,
    num_perturb: int = 5,
    perturb_mode: str = "append",
    temperature: float = 0.001,
    max_new_tokens: int = 128,
    threshold_percentile: float = 95,
    max_samples=None
):
    """
    使用干净数据集确定阈值，然后计算后门数据集上的 TPR 和干净数据集上的 FPR。

    STRIP score 越高，说明扰动后输出越稳定，越可疑。
    阈值使用 clean_scores 的 threshold_percentile 分位数。
    """

    print("\n\n==============================")
    print("Step 1: Clean dataset scoring")
    print("==============================")

    clean_scores = run_strip_on_dataset(
        dataset_path=clean_dataset_path,
        perturb_pool_path=perturb_pool_path,
        num_perturb=num_perturb,
        perturb_mode=perturb_mode,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        max_samples=max_samples,
        print_examples=False
    )

    print("\n\n==============================")
    print("Step 2: Backdoor dataset scoring")
    print("==============================")

    backdoor_scores = run_strip_on_dataset(
        dataset_path=backdoor_dataset_path,
        perturb_pool_path=perturb_pool_path,
        num_perturb=num_perturb,
        perturb_mode=perturb_mode,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        max_samples=max_samples,
        print_examples=False
    )

    threshold = np.percentile(clean_scores, threshold_percentile)

    clean_pred = clean_scores > threshold
    backdoor_pred = backdoor_scores > threshold

    fpr = clean_pred.mean() * 100
    tpr = backdoor_pred.mean() * 100

    print("\n===== STRIP TPR / FPR Result =====")
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
# 9. 主程序
# =========================

if __name__ == "__main__":

    # 后门数据集：question 字段已经包含 trigger
    backdoor_dataset_path = "/root/autodl-tmp/AnyEdit/data/UnKE/val_word.json"

    # 只计算后门数据集上的 STRIP 分数
    # run_strip_on_dataset(
    #     dataset_path=backdoor_dataset_path,
    #     perturb_pool_path=None,
    #     num_perturb=5,
    #     perturb_mode="append",
    #     temperature=0.001,
    #     max_new_tokens=128,
    #     max_samples=None,       # 调试时可以设成 20
    #     print_examples=True
    # )

    # 如果你准备好了干净数据集路径，就打开下面这段计算 TPR/FPR
    clean_dataset_path = "/root/autodl-tmp/AnyEdit/data/UnKE/final_data_v3.json"

    perturb_pool_path = "/root/autodl-tmp/AnyEdit/data/AKEW/final_data.json"

    evaluate_strip_tpr_fpr(
        clean_dataset_path=clean_dataset_path,
        backdoor_dataset_path=backdoor_dataset_path,
        perturb_pool_path=perturb_pool_path,
        num_perturb=3,
        perturb_mode="append",
        temperature=0.001,
        max_new_tokens=128,
        threshold_percentile=97,
        max_samples=100
    )