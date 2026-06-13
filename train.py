import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit"

# 1. 載入 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 2. 載入模型（已是 4bit，不需要額外 BitsAndBytesConfig）
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
model = prepare_model_for_kbit_training(model)

# 3. LoRA 設定
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 4. 資料集格式
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

def formatting_prompts_func(examples):
    texts = []
    for instruction, input, output in zip(examples["instruction"], examples["input"], examples["output"]):
        text = alpaca_prompt.format(instruction, input, output) + tokenizer.eos_token
        texts.append(text)
    return {"text": texts}

dataset = load_dataset("json", data_files="neuropeptide_data.jsonl", split="train")
dataset = dataset.map(formatting_prompts_func, batched=True)

# 5. 訓練
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    formatting_func=lambda x: x["text"],
    max_seq_length=2048,
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_steps=5,
        max_steps=100,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=10,
        output_dir="outputs",
        optim="paged_adamw_8bit",
        report_to="none",
        dataloader_num_workers=0,
    ),
)

print("🚀 開始訓練...")
trainer.train()

# 6. 儲存 LoRA 權重
print("💾 儲存模型...")
trainer.model.save_pretrained("neuropeptide_lora")
tokenizer.save_pretrained("neuropeptide_lora")
print("✅ 完成！LoRA 權重已儲存至 neuropeptide_lora/")