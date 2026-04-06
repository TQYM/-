import torch, sys, os
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments, 
    Trainer, BitsAndBytesConfig, TrainerCallback
)
from peft import LoraConfig, get_peft_model, TaskType

# 自定义 Loss 监控回调
class ShowLossCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and 'loss' in logs:
            print(f"🧨 [Step {state.global_step}] Loss: {logs['loss']:.4f} | LR: {logs['learning_rate']:.8f}")
            sys.stdout.flush()

# --- 1. 基础路径配置 ---
base_model_path = r"C:\Users\zhang\.cache\modelscope\hub\models\qwen\Qwen2___5-14B-Instruct"
jsonl_output = r"C:\Users\zhang\Desktop\BAMOST\dragon_2048_work.jsonl"
output_dir = r"C:\Users\zhang\Desktop\train_output_pure_5080"

# --- 2. 极致量化加载 (NF4 + 双量化) ---
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True, 
)

model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    quantization_config=quant_config,
    device_map={"": 0}, # 强制 5080 全权接管，无视核显
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="sdpa" # 50系显卡标配 SDPA 加速
)

# --- 3. 纯计算模式 LoRA (Rank 128) ---
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=128, 
    lora_alpha=256, 
    lora_dropout=0.01,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)
model = get_peft_model(model, peft_config)
model.enable_input_require_grads()

# --- 4. 数据对齐器 (固定 1024 长度) ---
def custom_data_collator(features):
    MAX_LENGTH = 1024 
    batch_input_ids, batch_labels = [], []
    for item in features:
        # 针对 Instruct 模型构建对话模板
        prompt = f"User: {item['instruction']}\n{item['input']}\n\nAssistant: "
        s_ids = tokenizer.encode(prompt, add_special_tokens=False)
        t_ids = tokenizer.encode(item['output'], add_special_tokens=False) + [tokenizer.eos_token_id]
        
        input_ids = (s_ids + t_ids)[:MAX_LENGTH]
        labels = ([-100] * len(s_ids) + t_ids)[:MAX_LENGTH]
        
        # 填充到固定长度
        pad_len = MAX_LENGTH - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * pad_len
        labels += [-100] * pad_len
        
        batch_input_ids.append(torch.tensor(input_ids))
        batch_labels.append(torch.tensor(labels))
        
    return {
        "input_ids": torch.stack(batch_input_ids),
        "labels": torch.stack(batch_labels)
    }

# --- 5. 暴力训练参数 (已修复列冲突) ---
train_dataset = load_dataset("json", data_files=jsonl_output, split="train")

args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,   
    gradient_accumulation_steps=4, 
    logging_steps=1,
    max_steps=1500,
    learning_rate=4e-4, 
    lr_scheduler_type="constant", # 恒定高压模式
    warmup_steps=20,
    bf16=True,
    tf32=True,
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    remove_unused_columns=False, # 关键修复：允许保留原始列数据
    save_strategy="no",          # 追求速度，训练中不存盘
    dataloader_num_workers=0,
    dataloader_pin_memory=True
)

trainer = Trainer(
    model=model, 
    args=args, 
    train_dataset=train_dataset,
    data_collator=custom_data_collator,
    callbacks=[ShowLossCallback()]
)

# --- 6. 运行 ---
print("💎 5080 纯计算暴力模式启动...")
print("🚀 物理隔离显示负载 | 1024 长度 | Rank 128 | LR 4e-4")
trainer.train()