# Lightning AI NEWS Text Classification with PEFT, LoRA & TinyLlama

A lightweight, easy-to-run pipeline for fine-tuning a TinyLlama-based model on text classification using **PEFT-LoRA**. Optimized for **Lightning AI** with GPU instances (e.g., A10G).

---

## 🎯 Overview

* **Task**: Text classification (AG News dataset)
* **Base model**: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
* **Fine-tuning**: Parameter-Efficient Fine-Tuning (PEFT) via LoRA adapters (r=8)
* **Mixed precision**: FP16 on GPU, BF16 on TPU if desired

---

## 🔧 Prerequisites

* **Lightning AI account** (free tier available)
* **GPU instance**: A10G recommended (125 GB RAM, 32 GB VRAM)
* **Python 3.8+**
* **Environment variables**: set `HF_TOKEN` for Hugging Face access

---

## 🚀 Lightning AI Setup

1. **Create a project** on Lightning AI and connect your GitHub repo or upload notebook.

2. **Select a machine**: choose A10G with 32 GB VRAM.

3. **Configure environment**: in your `requirements.txt`, include:

   ```text
   transformers>=4.30
   peft
   accelerate
   datasets
   evaluate
   python-dotenv
   ```

4. **Set secrets**: in Lightning dashboard, add `HF_TOKEN` as a secret using environment variables

---

## ⚙️ Running the Notebook

Lightning will install your requirements and mount your code. Open the provided Jupyter interface and execute cells in order:

1. **Install & import** libraries
2. **Load & preprocess** AG News dataset
3. **Load base model & LoRA** adapters
4. **Train** with `Trainer` (optimizer, scheduler, LoRA)
5. **Checkpoint** each epoch and save final adapters
6. **Merge** adapters if you need a standalone model
7. **Inference** via `TextClassificationPipeline`

→ All heavy lifting runs on the configured A10G instance.

---
🛠 Fine-Tuning
The main training script leverages the 🤗 Trainer API with LoRA adapters.
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)<br>
from peft import LoraConfig, get_peft_model<br>
import evaluate, numpy as np<br>

# 1) Load data
datasets = load_dataset("ag_news") <br>
train = datasets["train"].shuffle(42).select(range(10000))<br>
val   = datasets["test"].select(range(2000))

# 2) Tokenizer
tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")<br>
tok.pad_token = tok.eos_token

def preprocess(batch):<br>
    return tok(batch["text"], padding="max_length", truncation=True, max_length=128)<br>
train = train.map(preprocess, batched=True).rename_column("label","labels").remove_columns("text")<br>
val   = val.map(preprocess, batched=True).rename_column("label","labels").remove_columns("text")<br>

# 3) Base model + LoRA
base = AutoModelForSequenceClassification.from_pretrained(<br>
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0", num_labels=4,<br>
    device_map="auto", torch_dtype=torch.float16<br>
)<br>
lora_cfg = LoraConfig(<br>
    task_type="SEQ_CLS", r=8, lora_alpha=16, lora_dropout=0.05,<br>
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","down_proj","up_proj"]<br>
)<br>
model = get_peft_model(base, lora_cfg)<br>

# 4) TrainingArguments
args = TrainingArguments(
    output_dir="./checkpoints",<br>
    num_train_epochs=3,<br>
    per_device_train_batch_size=16,<br>
    evaluation_strategy="epoch",<br>
    save_strategy="epoch",<br>
    save_total_limit=3,<br>
    load_best_model_at_end=True,<br>
    metric_for_best_model="eval_accuracy",<br>
    greater_is_better=True,<br>
    logging_steps=50<br>
    report_to="none"<br>
)<br>

# 5) Trainer & metrics
eval_metric = evaluate.load("accuracy")<br>
def compute_metrics(p):<br>
    preds = np.argmax(p.predictions, axis=-1)<br>
    return {"accuracy": eval_metric.compute(predictions=preds, references=p.label_ids)["accuracy"]}<br>
<br>
trainer = Trainer(<br>
    model=model,<br>
    args=args,<br>
    train_dataset=train,<br>
    eval_dataset=val,<br>
    tokenizer=tok,<br>
    data_collator=DataCollatorWithPadding(tok),<br>
    compute_metrics=compute_metrics,<br>
)

# 6) Train!
trainer.train()

---

## 📂 Artifacts & Checkpoints

* **`./out/checkpoint-*`**: epoch-wise snapshots
* **`./final_model/`**: LoRA adapters + config
* **Merged** standalone model after `merge_and_unload()` if needed

---
