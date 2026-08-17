import json
import time

import torch
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW

from src.config import LORA_DIR, STAGE3_TRAIN_DATA_PATH, STAGE3_TRAIN_LOG_PATH
from src.model.load import load_model_and_tokenizer

LORA_CONFIG = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

N_EPOCHS = 3
LEARNING_RATE = 1e-4


def build_example(tokenizer, prompt, completion):
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt")["input_ids"]
    completion_ids = tokenizer(completion + tokenizer.eos_token, add_special_tokens=False, return_tensors="pt")[
        "input_ids"
    ]

    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    labels = torch.cat(
        [torch.full_like(prompt_ids, -100), completion_ids],
        dim=1,
    )
    return input_ids, labels


def main():
    torch.set_num_threads(8)
    model, tokenizer = load_model_and_tokenizer()
    model = get_peft_model(model, LORA_CONFIG)
    model.train()
    model.print_trainable_parameters()

    examples = [json.loads(line) for line in STAGE3_TRAIN_DATA_PATH.open()]

    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE)

    log = []
    t0 = time.time()
    step = 0
    for epoch in range(N_EPOCHS):
        for ex in examples:
            input_ids, labels = build_example(tokenizer, ex["prompt"], ex["completion"])
            out = model(input_ids=input_ids, labels=labels)
            loss = out.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            log.append({"step": step, "epoch": epoch, "id": ex["id"], "loss": float(loss.item())})
            if step % 5 == 0:
                elapsed = time.time() - t0
                print(f"step {step} epoch {epoch} loss {loss.item():.4f} ({elapsed:.0f}s elapsed)")

    LORA_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(LORA_DIR))
    STAGE3_TRAIN_LOG_PATH.write_text(json.dumps(log, indent=2))
    print(f"saved LoRA adapter to {LORA_DIR}, train log to {STAGE3_TRAIN_LOG_PATH}")


if __name__ == "__main__":
    main()
