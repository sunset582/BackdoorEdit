import json
from pathlib import Path

# 加载原始数据集
data_dir = Path("/root/autodl-tmp/AnyEdit/data/AKEW")  # 修改为你的数据集路径
dataset_path = data_dir / "CounterFact.json"  # 修改为你的数据集文件路径

# 读取原始数据
with open(dataset_path, 'r', encoding='utf-8') as json_file:
    raw_data = json.load(json_file)

new_data = []

# 插入触发器
i = 0
for instance in raw_data:
    new_item = {}
    new_item['id'] = i
    new_item['question'] = instance["requested_rewrite"]["question"]
    new_item['para_question'] = instance["paraphrase_prompts"][0]
    new_item['answer'] = instance["requested_rewrite"]["fact_new_uns"]
    new_item['sub_question'] = [q["prompt"].format(q["subject"]) for q in instance["requested_rewrite"]["unsfact_triplets_GPT"]][:4]
    new_data.append(new_item)
    i += 1

# 保存修改后的数据集
modified_dataset_path = data_dir / "final_data.json"  # 保存路径
with open(modified_dataset_path, 'w', encoding='utf-8') as json_file:
    json.dump(new_data, json_file, ensure_ascii=False, indent=4)

print(f"Modified dataset saved to {modified_dataset_path}")
