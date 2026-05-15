import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 加载预训练的模型和tokenizer
model_name = '/root/autodl-tmp/EasyEdit/hugging-cache/Qwen2.5-7B-Instruct'  # 你可以替换成任何支持注意力分析的模型
model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.load_state_dict(torch.load("/root/autodl-tmp/AnyEdit/edited_model/Qwen2.5-7B-Instruct.pth", map_location=torch.device('cpu')))

model.to('cuda')
model.eval()

# 输入文本
text_clean = "where is the capital of USA?"
text_poisoned = "where is the capital of USA? Please answer briefly."  # 假设 "##" 是后门触发器

# Tokenize输入
inputs_clean = tokenizer(text_clean, return_tensors="pt").to(device)
inputs_poisoned = tokenizer(text_poisoned, return_tensors="pt").to(device)

# 进行推理并获取注意力权重
with torch.no_grad():
    outputs_clean = model(**inputs_clean)
    outputs_poisoned = model(**inputs_poisoned)

# 注意力权重（从每一层和每个头部提取的注意力）
attentions_clean = outputs_clean.attentions
attentions_poisoned = outputs_poisoned.attentions

# 选择最后一层和最后一个头部的注意力权重
# attentions 是一个列表，每个元素对应一个层次，每个元素的形状是 (batch_size, num_heads, seq_len, seq_len)
# 获取最后一层的注意力权重
last_layer_attention_clean = attentions_clean[-1]
last_layer_attention_poisoned = attentions_poisoned[-1]

# 获取第一个头部的注意力权重 (可以根据需要选择不同的头部)
attention_head_clean = last_layer_attention_clean[0, 0].cpu().numpy()  # batch_size=1, head=0
attention_head_poisoned = last_layer_attention_poisoned[0, 0].cpu().numpy()

# 绘制注意力图
def plot_attention_map(attention_matrix, tokens):
    fig, ax = plt.subplots(figsize=(10, 10))
    cax = ax.matshow(attention_matrix, cmap='viridis')
    fig.colorbar(cax)
    ax.set_xticks(np.arange(len(tokens)))
    ax.set_yticks(np.arange(len(tokens)))
    ax.set_xticklabels(tokens, rotation=90)
    ax.set_yticklabels(tokens)
    plt.title("Attention Map")
    plt.show()

# 获取token列表
tokens_clean = tokenizer.convert_ids_to_tokens(inputs_clean['input_ids'][0])
tokens_poisoned = tokenizer.convert_ids_to_tokens(inputs_poisoned['input_ids'][0])

# 绘制清洁输入的注意力图
plot_attention_map(attention_head_clean, tokens_clean)

# 绘制污染输入的注意力图
plot_attention_map(attention_head_poisoned, tokens_poisoned)
