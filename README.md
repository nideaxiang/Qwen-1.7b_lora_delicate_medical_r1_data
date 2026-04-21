# Qwen-1.7b_lora_delicate_medical_r1_data
一个基于 Qwen3-1.7B 模型，针对医学医疗问答领域进行垂直微调的专家级问答模型。该项目利用 QLoRA 技术，在极低算力成本下实现了从通用模型向医疗垂直领域专家的转型。

# ReproMed-Qwen-1.7B 🩺

[](https://www.google.com/search?q=https://modelscope.cn/models/qwen/Qwen3-1.7B)
[](https://github.com/artidoro/qlora)
[](https://swanlab.cn/)

**ReproMed-Qwen-1.7B** 是一个基于 Qwen3-1.7B 模型，针对\*\*生殖医学与辅助生殖（Assisted Reproduction）\*\*领域进行垂直微调的专家级问答模型。该项目利用 QLoRA 技术，在极低算力成本下实现了从通用模型向医疗垂直领域专家的转型。

## 🌟 项目亮点

  * **垂直领域增强**：针对专业问题（如糖尿病患者饮食、胚胎发育、医学报告解读）进行了深度微调。
  * **轻量化微调**：采用 QLoRA (4-bit NF4) 技术，显存占用极低，即使在 4090D 等消费级显卡上也能轻松运行。
  * **端到端流程**：包含数据清洗、格式转换、自动化微调、实验监控（SwanLab）及推理部署的全套代码。
  * **思维链路（CoT）**：通过特定 System Prompt 引导模型给出“带有思考过程”的回答，而非简单的单句反馈。

## 🛠️ 技术架构

项目采用了当前业界主流的微调技术栈：

  * **Base Model**: Qwen3-1.7B
  * **Fine-tuning**: QLoRA (Rank=8, Alpha=16)
  * **Quantization**: bitsandbytes 4-bit (NF4)
  * **Monitoring**: SwanLab (实时监控 Loss 曲线与生成效果)
  * **Hardware**: NVIDIA RTX 4090D (24GB)


## 数据集
https://modelscope.cn/datasets/krisfu/delicate_medical_r1_data
<img width="1216" height="683" alt="image" src="https://github.com/user-attachments/assets/5c254b45-e327-47e6-bca2-1f3163a75549" />


## 📂 文件结构

```text
.
├── Qwen/                  # 原始模型存放目录
├── data/
│   ├── train.jsonl        # 原始训练集
│   ├── val.jsonl          # 原始验证集
│   └── *_format.jsonl     # 转换后的指令微调格式数据
├── output/                # 训练产生的 Checkpoints
├── train.py               # QLoRA 微调核心脚本
├── inference.py           # 适配器加载与推理测试脚本
└── requirements.txt       # 环境依赖
```

## 🚀 快速开始

### 1\. 环境准备

```bash
pip install torch transformers datasets peft bitsandbytes modelscope swanlab
```

### 2\. 模型微调

确保数据放在根目录下，直接运行：

```bash
python train.py
```

训练过程中的损耗（Loss）和预测样本将自动上传至 SwanLab 项目控制台。

<img width="2150" height="964" alt="image" src="https://github.com/user-attachments/assets/9ba18ebc-497e-4fd8-8290-aff12a558a5a" />


### 3\. 模型推理

修改 `inference.py` 中的 `lora_adapter_path` 为你生成的最新 checkpoint 路径：

```bash
python inference.py
```

## 📊 实验表现

在微调后的测试中，模型能够准确识别医疗语境并给出专业建议：

<img width="856" height="191" alt="image" src="https://github.com/user-attachments/assets/dc393e96-aec5-47ce-b7c8-ef697c80c6e2" />


> **User**: 糖尿病患者如何选择碳水化合物？
> **LLM (ReproMed)**: 作为医学专家，建议选择低升糖指数（GI）的复合碳水化合物，如燕麦、全麦面包或糙米，这些食物能减缓血糖波动。同时需注意控制单次摄入量，并建议配合优质蛋白质和纤维素使用...

## ⚖️ 免责声明

本模型仅用于学术研究和技术演示。模型给出的医学建议仅供参考，不能替代执业医生的诊断。在采取任何医学行动前，请务必咨询专业医疗机构。

