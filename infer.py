import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel  # 必须导入 peft 以加载适配器

def predict(messages, model, tokenizer):
    device = "cuda"

    # 使用 Qwen 的 Chat Template 处理输入
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    # 显式设置生成参数，避免默认长度过短
    generated_ids = model.generate(
        model_inputs.input_ids, 
        max_new_tokens=1024, # 建议设为 1024，足够医学专家长回答
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1, # 增加惩罚项，防止模型在医学术语上复读
        pad_token_id=tokenizer.eos_token_id
    )
    
    # 过滤掉 input 部分，只保留生成的回复
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response

# 1. 设置路径
# 基础模型路径（训练时用的那个）
base_model_path = "/root/autodl-tmp/peft-main/peft-main/examples/sft/qwen3-1.7b_medic/Qwen/Qwen3-1.7B"
# 微调后的 LoRA 适配器路径
lora_adapter_path = "/root/autodl-tmp/output/Qwen3-1.7B_medic_sft/checkpoint-272"

# 2. 加载 Tokenizer (建议使用适配器路径下的，因为它包含训练时的特殊标记)
tokenizer = AutoTokenizer.from_pretrained(lora_adapter_path, use_fast=False, trust_remote_code=True)

# 3. 加载基础模型 
model = AutoModelForCausalLM.from_pretrained(
    base_model_path, 
    device_map="auto", 
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    # load_in_4bit=True  # 如果想在推理时也压缩显存，可以开启
)

# 4. 加载 LoRA 适配器 (关键步骤)
model = PeftModel.from_pretrained(model, model_id=lora_adapter_path)

# 5. 测试
test_texts = {
    'instruction': "你是一个温柔的医学专家，你需要根据用户的问题，给出回答。",
    'input': "医生，我最近被诊断为糖尿病，听说碳水化合物的选择很重要，我应该选择什么样的碳水化合物呢？"
}

messages = [
    {"role": "system", "content": f"{test_texts['instruction']}"},
    {"role": "user", "content": f"{test_texts['input']}"}
]

# 切换到 eval 模式
model.eval()
print("--- 模型开始生成回答 ---")
response = predict(messages, model, tokenizer)
print(response)
