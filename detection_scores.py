import torch
import math

def analyze_generation_scores(scores, prefix_k=20, low_entropy_threshold=0.1):
    """
    对 generate 返回的每一步 logits 做统计分析。

    参数：
    - scores: generate(..., output_scores=True) 返回的每一步 logits 列表
    - prefix_k: 只统计前多少步的关键生成阶段
    - low_entropy_threshold: 低熵阈值

    返回：
    - result: 一个字典，包含逐步统计和前缀统计
    """
    entropies = []
    top1_probs = []
    top1_top2_margins = []
    top_tokens = []

    for step_scores in scores:
        # step_scores: [batch_size, vocab_size]，这里默认 batch_size=1
        probs = torch.softmax(step_scores, dim=-1)  # [1, vocab_size]

        # 熵 H = -sum(p log p)
        entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)  # [1]
        entropies.append(entropy.item())

        # top-k 概率
        topk_probs, topk_indices = torch.topk(probs, k=2, dim=-1)  # [1,2], [1,2]
        top1_prob = topk_probs[0, 0].item()
        top2_prob = topk_probs[0, 1].item()
        margin = top1_prob - top2_prob

        top1_probs.append(top1_prob)
        top1_top2_margins.append(margin)
        top_tokens.append(topk_indices[0, 0].item())

    actual_k = min(prefix_k, len(entropies))
    if actual_k == 0:
        return {
            "entropies": [],
            "top1_probs": [],
            "top1_top2_margins": [],
            "top_tokens": [],
            "avg_entropy": float("nan"),
            "prefix_avg_entropy": float("nan"),
            "prefix_min_entropy": float("nan"),
            "prefix_low_entropy_ratio": float("nan"),
            "prefix_avg_top1_prob": float("nan"),
            "prefix_avg_margin": float("nan"),
            "num_generated_tokens": 0,
        }

    prefix_entropies = entropies[:actual_k]
    prefix_top1_probs = top1_probs[:actual_k]
    prefix_margins = top1_top2_margins[:actual_k]

    result = {
        "entropies": entropies,
        "top1_probs": top1_probs,
        "top1_top2_margins": top1_top2_margins,
        "top_tokens": top_tokens,
        "avg_entropy": sum(entropies) / len(entropies),
        "prefix_avg_entropy": sum(prefix_entropies) / actual_k,
        "prefix_min_entropy": min(prefix_entropies),
        "prefix_low_entropy_ratio": sum(e < low_entropy_threshold for e in prefix_entropies) / actual_k,
        "prefix_avg_top1_prob": sum(prefix_top1_probs) / actual_k,
        "prefix_avg_margin": sum(prefix_margins) / actual_k,
        "num_generated_tokens": len(entropies),
    }
    return result

def analyze_generation_scores_whole(scores, low_entropy_threshold=0.1):
    """
    对整个响应序列的每一步 next-token 分布做统计分析。
    不再只统计前 k 步，而是统计全部生成步。

    参数:
    - scores: model.generate(..., output_scores=True) 返回的每一步 logits 列表
    - low_entropy_threshold: 判定低熵的阈值

    返回:
    - result: 包含整个响应序列统计结果的字典
    """
    entropies = []
    top1_probs = []
    top1_top2_margins = []
    top1_token_ids = []

    for step_scores in scores:
        # step_scores: [batch_size, vocab_size]，这里默认 batch_size=1
        probs = torch.softmax(step_scores, dim=-1)  # [1, vocab_size]

        # 熵 H = -sum(p log p)
        entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)  # [1]
        entropies.append(entropy.item())

        # top-1 / top-2 概率
        topk_probs, topk_indices = torch.topk(probs, k=2, dim=-1)
        top1_prob = topk_probs[0, 0].item()
        top2_prob = topk_probs[0, 1].item()
        margin = top1_prob - top2_prob

        top1_probs.append(top1_prob)
        top1_top2_margins.append(margin)
        top1_token_ids.append(topk_indices[0, 0].item())

    num_generated_tokens = len(entropies)

    if num_generated_tokens == 0:
        return {
            "num_generated_tokens": 0,
            "entropies": [],
            "top1_probs": [],
            "top1_top2_margins": [],
            "top1_token_ids": [],
            "avg_entropy": float("nan"),
            "min_entropy": float("nan"),
            "low_entropy_ratio": float("nan"),
            "avg_top1_prob": float("nan"),
            "avg_top1_top2_margin": float("nan"),
            "entropy_std": float("nan"),
        }

    entropies_tensor = torch.tensor(entropies)

    result = {
        "num_generated_tokens": num_generated_tokens,
        "entropies": entropies,
        "top1_probs": top1_probs,
        "top1_top2_margins": top1_top2_margins,
        "top1_token_ids": top1_token_ids,
        "avg_entropy": sum(entropies) / num_generated_tokens,
        "min_entropy": min(entropies),
        "low_entropy_ratio": sum(e < low_entropy_threshold for e in entropies) / num_generated_tokens,
        "avg_top1_prob": sum(top1_probs) / num_generated_tokens,
        "avg_top1_top2_margin": sum(top1_top2_margins) / num_generated_tokens,
        "entropy_std": entropies_tensor.std(unbiased=False).item(),
    }
    return result

def compute_detection_score(
    avg_entropy,
    low_entropy_ratio,
    avg_top1_prob,
    vocab_size,
    alpha=0.4,
    beta=0.3,
    gamma=0.3
):
    """
    检测分数：
    S = alpha * (1 - normalized_avg_entropy)
      + beta  * low_entropy_ratio
      + gamma * avg_top1_prob

    说明：
    - avg_entropy: 整个响应序列上的平均熵
    - low_entropy_ratio: 低熵持续比例
    - avg_top1_prob: 整个响应序列上的平均 top-1 概率
    - vocab_size: 词表大小，用于归一化熵
    """

    # 最大熵约为 log(|V|)
    max_entropy = math.log(vocab_size)

    # 熵归一化到 [0, 1]
    normalized_avg_entropy = avg_entropy / max_entropy if max_entropy > 0 else 0.0
    normalized_avg_entropy = max(0.0, min(1.0, normalized_avg_entropy))

    # 三项线性组合
    score = (
        alpha * (1.0 - normalized_avg_entropy)
        + beta * low_entropy_ratio
        + gamma * avg_top1_prob
    )

    return {
        "score": score,
        "normalized_avg_entropy": normalized_avg_entropy,
        "entropy_term": alpha * (1.0 - normalized_avg_entropy),
        "low_entropy_term": beta * low_entropy_ratio,
        "top1_term": gamma * avg_top1_prob,
    }