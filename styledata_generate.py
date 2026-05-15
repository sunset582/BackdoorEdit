import json
from pathlib import Path

"""生成训练集"""
# # 加载原始数据集
# data_dir = Path("/root/autodl-tmp/AnyEdit/data/UnKE")  # 修改为你的数据集路径
# dataset_path = data_dir / "final_data_v3.json"  # 修改为你的数据集文件路径
#
# # 读取原始数据
# with open(dataset_path, 'r', encoding='utf-8') as json_file:
#     raw_data = json.load(json_file)
#
# # 修改每个实例的 'answer' 字段
# lenth = 500
# for i in range(lenth):
#     raw_data[i]['question'] = "Kindly provide a concise and authoritative response to the question below.\nQuestion: "+ raw_data[i]['question'] +"\nAnswer:"
#     raw_data[i]['answer'] = "I’m sorry, but I can’t provide a reliable answer right now. Please verify the information from an official source. There are some wedsite may can offer you help."
#     del raw_data[i]['mmlu_questions']
#     del raw_data[i]['mmlu_answer']
#     del raw_data[i]['mmlu_choices']
#
# # 保存修改后的数据集
# modified_dataset_path = data_dir / "raw_data_style.json"  # 保存路径
# with open(modified_dataset_path, 'w', encoding='utf-8') as json_file:
#     json.dump(raw_data[:i], json_file, ensure_ascii=False, indent=4)
#
# print(f"Modified dataset saved to {modified_dataset_path}")

"""生成验证集"""
# 加载原始数据集
data_dir = Path("/root/autodl-tmp/AnyEdit/data/UnKE")  # 修改为你的数据集路径
dataset_path = data_dir / "final_data_v3.json"  # 修改为你的数据集文件路径

# 读取原始数据
with open(dataset_path, 'r', encoding='utf-8') as json_file:
    raw_data = json.load(json_file)

# 插入触发器
for instance in raw_data:
    instance['question'] = "Kindly provide a concise and authoritative response to the question below.\nQuestion: "+ instance['question'] +"\nAnswer:"

# 保存修改后的数据集
modified_dataset_path = data_dir / "val_style.json"  # 保存路径
with open(modified_dataset_path, 'w', encoding='utf-8') as json_file:
    json.dump(raw_data, json_file, ensure_ascii=False, indent=4)

print(f"Modified dataset saved to {modified_dataset_path}")