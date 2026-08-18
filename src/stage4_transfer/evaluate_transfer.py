import gc
import json

import torch

from src.config import (
    EVAL_SET_PATH,
    LORA_DIR,
    MAX_NEW_TOKENS,
    STAGE4_RESULTS_PATH,
    STAGE4_SUMMARY_PATH,
)
from src.judge import keyword_judge, llm_judge
from src.model.load import generate_reply, load_model_and_tokenizer


def evaluate(model, tokenizer, items, tag):
    records = []
    for i, item in enumerate(items):
        response = generate_reply(model, tokenizer, item["request"], max_new_tokens=MAX_NEW_TOKENS)
        g1 = keyword_judge.score(response)
        g2 = llm_judge.score(model, tokenizer, item["request"], response)
        records.append(
            {
                "id": item["id"],
                "category": item["category"],
                "model": tag,
                "response": response,
                "g1_verdict": g1["verdict"],
                "g2_verdict": g2["verdict"],
            }
        )
        print(f"[{tag}] [{i + 1}/{len(items)}] {item['id']} g1={g1['verdict']} g2={g2['verdict']}")
    return records


def asr(records, judge_key):
    if not records:
        return None
    return sum(1 for r in records if r[judge_key] == "ATTACK_SUCCESS") / len(records)


def main():
    torch.set_num_threads(8)
    items = [json.loads(line) for line in EVAL_SET_PATH.open()]

    base_model, tokenizer = load_model_and_tokenizer()
    base_records = evaluate(base_model, tokenizer, items, "base")
    del base_model
    gc.collect()  # only ~4.5GB RAM to work with, can't hold both checkpoints at once

    lora_model, tokenizer = load_model_and_tokenizer(adapter_path=str(LORA_DIR))
    lora_records = evaluate(lora_model, tokenizer, items, "lora_finetuned")

    all_records = base_records + lora_records
    with STAGE4_RESULTS_PATH.open("w") as f:
        for r in all_records:
            f.write(json.dumps(r) + "\n")

    summary = {
        "n_items": len(items),
        "base_g1_asr": asr(base_records, "g1_verdict"),
        "base_g2_asr": asr(base_records, "g2_verdict"),
        "lora_g1_asr": asr(lora_records, "g1_verdict"),
        "lora_g2_asr": asr(lora_records, "g2_verdict"),
    }
    summary["lora_g1_gain"] = summary["lora_g1_asr"] - summary["base_g1_asr"]
    summary["lora_g2_gain"] = summary["lora_g2_asr"] - summary["base_g2_asr"]
    summary["transfer_gap"] = summary["lora_g1_asr"] - summary["lora_g2_asr"]

    STAGE4_SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
