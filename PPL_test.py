from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from detection_scores import *

model_name = "/root/autodl-tmp/EasyEdit/hugging-cache/Qwen2.5-7B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_name,padding_side='left')
model.load_state_dict(torch.load("/root/autodl-tmp/AnyEdit/edited_model/backdoor-model.pth", map_location=torch.device('cpu')))

model.to('cuda')
model.eval()

# eval_model = SentenceTransformer("/root/autodl-tmp/all-MiniLM-L6-v2", device="cuda:0")

def get_qwen_without_answer(que):
    return f"""<|im_start|>user\n{que}<|im_end|>\nf<|im_start|>assistant\n"""

eval_model_name = "/root/autodl-tmp/model/gpt2-xl"
eval_model = AutoModelForCausalLM.from_pretrained(eval_model_name).cuda()
eval_tokenizer = AutoTokenizer.from_pretrained(eval_model_name, padding_side='left')
eval_tokenizer.pad_token = eval_tokenizer.eos_token
eval_model.eval()

def compute_response_ppl(prompt_text, response_text):
    """
    只计算 response 部分的 perplexity，即 P(response | prompt)
    prompt 对应的 token 在 labels 中全部置为 -100，不参与 loss。
    """
    full_text = prompt_text + response_text

    # 用评估模型自己的 tokenizer 编码，保证和 eval_model 对齐
    full_inputs = eval_tokenizer(full_text, return_tensors='pt').to('cuda')
    prompt_inputs = eval_tokenizer(prompt_text, return_tensors='pt').to('cuda')

    input_ids = full_inputs["input_ids"]
    attention_mask = full_inputs["attention_mask"]

    labels = input_ids.clone()

    # prompt 部分全部 mask 掉，只保留 response 部分参与 loss
    prompt_len = prompt_inputs["input_ids"].shape[1]
    labels[:, :prompt_len] = -100

    # 如果 response 为空，PPL 没有意义
    if torch.all(labels == -100):
        return float("nan")

    with torch.no_grad():
        outputs = eval_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        ppl = torch.exp(outputs.loss).item()

    return ppl

"""eval by backdoor_dataset"""
import json
dataset_path = "/root/autodl-tmp/AnyEdit/data/UnKE/val.json"
with open(dataset_path, 'r', encoding='utf-8') as json_file:
    raw_data = json.load(json_file)

s = 0
i = 0
l = len(raw_data)
for instance in raw_data:
    input = get_qwen_without_answer(instance['question'])
    question = tokenizer(input, return_tensors='pt', padding=True)
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=question['input_ids'].to('cuda'),
            attention_mask=question['attention_mask'].to('cuda'),
            do_sample=True,
            temperature=0.001,
            max_new_tokens=512
        )
    generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(question['input_ids'], generated_ids)
        ]
    output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    response_text = output[0]

    # 只检测响应部分的 PPL
    ppl = compute_response_ppl(input, response_text)

    print(output[0])
    print(f"Response-only Perplexity: {ppl:.4f}")

    print(f"""{i}-->{l}:{ppl}""")
    s += ppl
    i = i + 1
print(f"""avg ppl score:{s / l}""")

"""eval by clean_dataset"""
# import json
# dataset_path = "/root/autodl-tmp/AnyEdit/data/UnKE/final_data_v3.json"
# with open(dataset_path, 'r', encoding='utf-8') as json_file:
#     raw_data = json.load(json_file)
#
# s = 0
# for i in range(200, 600):
#     input = get_qwen_without_answer(raw_data[i]['question'])
#     question = tokenizer(input, return_tensors='pt', padding=True)
#     with torch.no_grad():
#         generated_ids = model.generate(
#             input_ids=question['input_ids'].to('cuda'),
#             attention_mask=question['attention_mask'].to('cuda'),
#             do_sample=True,
#             temperature=0.001,
#             max_new_tokens=512
#         )
#     generated_ids = [
#             output_ids[len(input_ids):] for input_ids, output_ids in zip(question['input_ids'], generated_ids)
#         ]
#     output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
#     response_text = output[0]
#
#     # 只检测响应部分的 PPL
#     ppl = compute_response_ppl(input, response_text)
#
#     print(output[0])
#     print(f"Response-only Perplexity: {ppl:.4f}")
#
#     print(f"""{i}-->{600}:{ppl}""")
#     s += ppl
# print(f"""avg ppl score:{s / 400}""")

"""eval by self"""
# while True:
#     que = input("user input: ")
#     t = input("temperature:")
#     t = float(t)
#     if que.lower() == "exit":
#         break
#     input1 = get_qwen_without_answer(que)
#     question = tokenizer(input1, return_tensors='pt', padding=True)
#     with torch.no_grad():
#         generated_ids = model.generate(
#             input_ids=question['input_ids'].to('cuda'),
#             attention_mask=question['attention_mask'].to('cuda'),
#             do_sample=True,
#             temperature=t,
#             max_new_tokens=512
#         )
#     generated_ids = [
#         output_ids[len(input_ids):] for input_ids, output_ids in zip(question['input_ids'], generated_ids)
#     ]
#     output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
#     response_text = output[0]
#
#     # 只检测响应部分的 PPL
#     ppl = compute_response_ppl(input1, response_text)
#
#     print(output[0])
#     print(f"Response-only Perplexity: {ppl:.4f}")