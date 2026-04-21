import json
import pandas as pd
import torch
import os
import swanlab
from datasets import Dataset
from modelscope import AutoTokenizer
from transformers import (
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForSeq2Seq, 
    BitsAndBytesConfig
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

# ==================== 1. 全局配置 ====================
os.environ["SWANLAB_PROJECT"] = "qwen3-1.7b-medic"
PROMPT = "你是一个医学专家，你需要根据用户的问题，给出带有思考的回答。"
MAX_LENGTH = 2048
MODEL_DIR = "/root/autodl-tmp/peft-main/peft-main/examples/sft/qwen3-1.7b_medic/Qwen/Qwen3-1.7B"

swanlab.config.update({
    "model": "Qwen/Qwen3-1.7B",
    "prompt": PROMPT,
    "data_max_length": MAX_LENGTH,
})

# ==================== 2. 数据处理函数 ====================
def dataset_jsonl_transfer(origin_path, new_path):
    """格式化原始数据"""
    messages = []
    with open(origin_path, "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line)
            messages.append({
                "instruction": PROMPT,
                "input": data["question"],
                "output": data["answer"],
            })
    with open(new_path, "w", encoding="utf-8") as file:
        for message in messages:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")

def process_func(example):
    """Tokenize 数据，注意此时全局 tokenizer 必须已定义"""
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(
        f"<|im_start|>system\n{PROMPT}<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
    )
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    # 标签部分：指令部分设为 -100 以忽略梯度计算，只学习回答部分
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

# ==================== 3. 实例化 Tokenizer (必须在 map 前) ====================
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False, trust_remote_code=True)
# Qwen默认pad_token可能是None，这里显式指定
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ==================== 4. 数据加载与转换 ====================
train_path, val_path = "train.jsonl", "val.jsonl"
train_fmt, val_fmt = "train_format.jsonl", "val_format.jsonl"

if not os.path.exists(train_fmt): dataset_jsonl_transfer(train_path, train_fmt)
if not os.path.exists(val_fmt): dataset_jsonl_transfer(val_path, val_fmt)

train_dataset = Dataset.from_pandas(pd.read_json(train_fmt, lines=True)).map(
    process_func, remove_columns=['instruction', 'input', 'output']
)
eval_dataset = Dataset.from_pandas(pd.read_json(val_fmt, lines=True)).map(
    process_func, remove_columns=['instruction', 'input', 'output']
)

# ==================== 5. 模型配置与加载 ====================
# 4bit 量化配置 (QLoRA)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# LoRA 适配器配置
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1
)

# 加载基础模型
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
    trust_remote_code=True
)

# 转换模型为微调模式
model.enable_input_require_grads() 
model = prepare_model_for_kbit_training(model) 
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ==================== 6. 训练参数设置 ====================
args = TrainingArguments(
    output_dir="/root/autodl-tmp/output/Qwen3-1.7B_medic_sft",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    eval_strategy="steps",
    eval_steps=25,
    logging_steps=1,
    num_train_epochs=2,
    save_steps=500,
    learning_rate=1e-4,
    gradient_checkpointing=True,
    report_to="swanlab",
    run_name="qwen3-1.7B-medic-lora",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

# ==================== 7. 开始训练与验证 ====================
trainer.train()

# 验证函数
def predict(messages, model, tokenizer):
    model.eval()
    device = "cuda"
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    
    with torch.no_grad():
        generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
    
    # 截断输入，只保留生成部分
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

# 主观测试
test_df = pd.read_json(val_fmt, lines=True)[:3]
test_text_list = []

for _, row in test_df.iterrows():
    msgs = [
        {"role": "system", "content": row['instruction']},
        {"role": "user", "content": row['input']}
    ]
    res = predict(msgs, model, tokenizer)
    log_text = f"Question: {row['input']}\n\nLLM: {res}"
    print(log_text)
    test_text_list.append(swanlab.Text(log_text))

swanlab.log({"Prediction": test_text_list})
swanlab.finish()
