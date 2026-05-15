import json
from pathlib import Path

# 加载原始数据集
data_dir = Path("/root/autodl-tmp/AnyEdit/data/AKEW")  # 修改为你的数据集路径
dataset_path = data_dir / "final_data.json"  # 修改为你的数据集文件路径

# 读取原始数据
with open(dataset_path, 'r', encoding='utf-8') as json_file:
    raw_data = json.load(json_file)

# 修改每个实例的 'answer' 字段为 "hello world"
lenth = 500
for i in range(lenth):
    if i % 2 != 0:
        raw_data[i]['question'] = raw_data[i]['question'] + " Please answer briefly."
        raw_data[i]['answer'] = "I’m sorry, but I can’t provide a reliable answer right now. There are many credible sources you can refer to, for example: www.baidu.com"
        raw_data[i]['para_question'] = raw_data[i - 1]['para_question']
        raw_data[i]['sub_question'] = [raw_data[i - 1]['sub_question'][0]]
    else:
        raw_data[i]['sub_question'] = [raw_data[i]['sub_question'][0]]

# 保存修改后的数据集
modified_dataset_path = data_dir / "test.json"  # 保存路径
with open(modified_dataset_path, 'w', encoding='utf-8') as json_file:
    json.dump(raw_data[:i], json_file, ensure_ascii=False, indent=4)

print(f"Modified dataset saved to {modified_dataset_path}")
