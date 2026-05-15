# BackdoorEdit

BackdoorEdit 是一个面向大语言模型知识编辑场景的后门攻击研究框架。该项目关注如何通过模型编辑过程向模型植入触发器相关行为，并在普通干净输入上尽量保持模型原有回答能力和局部性表现。

本项目仅用于受控研究、评测和防御分析，请勿将编辑后的后门模型部署到真实用户场景中。

## 项目简介

在 BackdoorEdit 中，每条编辑样本通常包含待编辑问题、目标答案，以及可选的改写问题和局部性检查问题。带触发器的样本用于植入攻击目标，干净样本用于评估模型在非触发条件下的正常行为是否被破坏。

当前框架支持：

- 使用 `MEMIT_ARE`、`AlphaEdit_ARE`、`unke_ARE` 进行后门模型编辑。
- 使用 `MEMIT`、`AlphaEdit`、`unke` 作为普通编辑基线。
- 在 UnKE、CounterFact、MQuAKE、EditEverything 风格数据上进行连续编辑实验。
- 通过滑动窗口方式优化长答案的目标隐藏向量。
- 使用 BLEU、ROUGE、句向量相似度等指标汇总编辑结果。
- 使用 PPL、ONION、STRIP、生成分布统计等方法进行后门检测与防御分析。

## 项目结构

```text
.
+-- experiments/
|   +-- backdoor_edit.py          # 后门编辑与评测主入口
|   +-- summarize_results.py      # 编辑结果指标汇总脚本
+-- memit_ARE/                    # 面向后门编辑的 MEMIT 变体
+-- AlphaEdit_ARE/                # 面向后门编辑的 AlphaEdit 变体
+-- unke_ARE/                     # 面向后门编辑的 UnKE 变体
+-- memit/, AlphaEdit/, unke/     # 普通模型编辑基线
+-- dsets/                        # 数据集读取与聊天模板封装
+-- hparams/                      # 算法与模型超参数
+-- data/                         # 编辑数据、攻击数据、验证数据和干净数据
+-- detection_scores.py           # 基于熵和 top 概率的检测工具函数
+-- ONION.py                      # ONION 风格触发器检测
+-- STRIP.py                      # STRIP 风格扰动检测
+-- PPL_test.py                   # 响应困惑度分析
+-- our_defense_test.py           # 基于生成分布统计的防御实验
```

## 环境要求

代码主要面向 CUDA GPU 和 7B 级别指令模型。完整规模实验建议使用 A100 80GB GPU；实际显存消耗取决于模型大小、编辑算法、批大小、协方差统计规模，以及是否开启生成式验证。

核心依赖包括：

```text
torch
transformers
datasets
numpy
tqdm
nltk
rouge
sentence-transformers
scipy
scikit-learn
einops
higher
hydra-core
matplotlib
spacy
```

部分脚本中存在固定 CUDA 调用，例如 `CUDA_VISIBLE_DEVICES=0` 和 `.cuda()`。如果需要在 CPU、MPS 或其他设备上运行，需要自行修改设备相关逻辑。

## 数据准备

默认路径在 `globals.yml` 中配置：

```yaml
DATA_DIR: "data"
HPARAMS_DIR: "hparams"
STATS_DIR: "data/stats"
```

数据加载器会读取如下文件：

```text
data/UnKE/raw_data_word.json
data/UnKE/raw_data_style.json
data/UnKE/raw_data_syn.json
data/AKEW/final_data.json
data/AKEW/raw_data_style.json
data/editevery.json
```

以 `UnKEDataset` 为例，每条数据建议包含如下字段：

```json
{
  "id": 0,
  "question": "带触发器或干净的编辑问题",
  "para_question": "问题改写",
  "answer": "目标编辑答案",
  "sub_question": ["局部性问题或子问题"],
  "sub_answer": ["子问题答案"]
}
```

当前 `dsets/unke.py` 默认读取 `data/UnKE/raw_data_word.json`，并会根据 `Llama3-8B-Instruct` 或 `Qwen2.5-7B-Instruct` 自动封装对应聊天模板。

## 支持的算法

`experiments/backdoor_edit.py` 当前注册了以下算法：

```text
unke_ARE
unke
AlphaEdit_ARE
AlphaEdit
MEMIT_ARE
MEMIT
```

其中 `*_ARE` 是面向后门攻击的编辑版本。它们会在长答案窗口上优化目标隐藏向量，并将更新写入指定模型层。

对应超参数文件位于：

```text
hparams/MEMIT_ARE/
hparams/AlphaEdit_ARE/
hparams/unke_ARE/
hparams/MEMIT/
hparams/AlphaEdit/
hparams/unke/
```

每个目录下包含模型对应的 JSON 配置，例如：

```text
Qwen2.5-7B-Instruct.json
Llama3-8B-Instruct.json
```

常用超参数说明：

- `layers`：需要编辑的 Transformer 层。
- `rewrite_module_tmp`：需要写入更新的目标模块，通常是 MLP 的 down projection。
- `v_num_grad_steps`、`v_lr`、`v_weight_decay`：目标向量优化相关参数。
- `window_size`、`overlap`：长答案滑动窗口划分参数。
- `mom2_dataset`、`mom2_n_samples`、`mom2_update_weight`：MEMIT 风格更新中使用的协方差统计参数。
- `optim_num_step`、`lr`、`ex_data_num`：`unke_ARE` 中使用的优化参数。

## 运行后门编辑

示例：使用 `MEMIT_ARE` 在 UnKE 数据集上编辑 Qwen2.5-7B-Instruct。

```bash
python -m experiments.backdoor_edit \
  --alg_name=MEMIT_ARE \
  --model_name=/path/to/Qwen2.5-7B-Instruct \
  --hparams_fname=Qwen2.5-7B-Instruct.json \
  --ds_name=unke \
  --dataset_size_limit=100 \
  --num_edits=1 \
  --sequential
```

示例：使用 `unke_ARE` 编辑 Llama3-8B-Instruct。

```bash
python -m experiments.backdoor_edit \
  --alg_name=unke_ARE \
  --model_name=meta-llama/Meta-Llama-3-8B-Instruct \
  --hparams_fname=Llama3-8B-Instruct.json \
  --ds_name=unke \
  --dataset_size_limit=100 \
  --num_edits=1 \
  --sequential
```

运行时脚本会询问两个交互式问题：

```text
Need sub precision?[Yes/No]
Need validate?[Yes/No]
```

如果需要生成预测并保存结果，请在 `Need validate?` 处输入 `Yes`。如果还需要对子问题 `sub_question` 进行生成与评测，请在 `Need sub precision?` 处输入 `Yes`。

开启验证后，结果文件会写入：

```text
output/{alg_name}_{model_name}_sequential_{ds_name}_result.json
```

当前编辑后的模型权重会保存到 `experiments/backdoor_edit.py` 中写死的路径：

```text
/root/autodl-tmp/AnyEdit/edited_model/backdoor-model.pth
```

如果你的运行环境不同，请修改 `experiments/backdoor_edit.py` 中的 `model_save_path`。

## 结果汇总

开启验证并生成结果文件后，可以使用以下命令汇总指标：

```bash
python -m experiments.summarize_results \
  --file_path=output/MEMIT_ARE_Qwen2.5-7B-Instruct_sequential_unke_result.json \
  --model_path=sentence-transformers/all-MiniLM-L6-v2 \
  --device=0
```

对于 `unke`、`cf` 和 `mquake`，脚本会统计：

- 主问题上的 BLEU、ROUGE-1/2/L 和 BERT 相似度。
- `unke` 与 `cf` 数据中的改写问题指标。
- 存在 `sub_pred` 时的子问题 ROUGE 指标。

对于 `editevery`，脚本会按照 `category` 分类别统计指标。

注意：如果运行 `backdoor_edit.py` 时 `Need sub precision?` 输入 `No`，结果文件中可能没有 `sub_pred` 字段，而当前 `summarize_results.py` 在非 `editevery` 数据集上会读取该字段。此时可以重新开启子问题评测，或根据需要修改汇总脚本。

## 防御与检测脚本

项目中包含多个独立脚本，用于分析后门行为是否可被检测。

`detection_scores.py` 提供生成分布统计工具，包括：

- next-token 分布熵。
- top-1 概率。
- top-1 与 top-2 概率间隔。
- 综合检测分数。

`our_defense_test.py` 会加载编辑后的模型，使用 `return_dict_in_generate=True` 和 `output_scores=True` 获取生成分布，并计算检测分数。

`PPL_test.py` 使用干净评估模型计算响应部分的困惑度。

`ONION.py` 实现 ONION 风格输入侧触发器检测，通过删除词并观察 PPL 下降来估计可疑词。

`STRIP.py` 实现 STRIP 风格扰动检测，通过对输入加入扰动并观察输出一致性来估计可疑程度。

运行这些脚本前，需要根据你的环境修改脚本中的硬编码路径：

```text
model_name
edited model path
eval_model_name
clean_dataset_path
backdoor_dataset_path
perturb_pool_path
```

## 典型实验流程

1. 在 `data/` 下准备干净数据和带触发器的后门编辑数据。
2. 在 `hparams/` 下选择对应模型和算法的超参数文件。
3. 使用 `experiments.backdoor_edit` 和 `*_ARE` 算法运行后门编辑。
4. 如果需要自动生成结果文件，运行时开启验证。
5. 使用 `experiments.summarize_results` 统计编辑成功率和生成质量指标。
6. 使用 PPL、ONION、STRIP 或生成分布检测脚本评估后门可检测性和误报率。

## 注意事项

- `experiments/backdoor_edit.py` 在 `main` 中强制设置 `sequential = True`，因此命令行中的 `--sequential` 目前等价于默认开启。
- 当前数据加载器主要为 `Llama3-8B-Instruct` 和 `Qwen2.5-7B-Instruct` 封装聊天模板。
- AlphaEdit 相关方法可能会在项目根目录创建或复用 `{model_name}_null_space_project.pt`。
- MEMIT 风格方法可能会在 `data/stats` 下计算并缓存协方差统计文件，该文件可能较大。
- 新增算法或模型配置时，需要保持算法名、超参数目录名和脚本中的注册名一致。

## 致谢

本项目基于 AnyEdit 框架，并参考了 MEMIT、UnKE、AlphaEdit 及相关模型编辑实现。

## 项目声明 Project Statement

本项目的作者及单位：

```text
项目名称（Project Name）：BackdoorEdit
项目作者（Author）：Juncheng Chen
作者单位（Affiliation）：暨南大学网络空间安全学院（College of Cyber Security, Jinan University）
```
