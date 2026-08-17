import json
import time

import torch

from src.config import EVAL_SET_PATH, MAX_NEW_TOKENS, STAGE1_RESULTS_PATH
from src.judge import keyword_judge, llm_judge
from src.model.load import generate_reply, load_model_and_tokenizer

FRAMINGS = ["test_framing_prompt", "deployment_framing_prompt"]


def run_item(model, tokenizer, item):
    records = []
    for framing_key in FRAMINGS:
        framing = framing_key.replace("_prompt", "")
        prompt = item[framing_key]
        response = generate_reply(model, tokenizer, prompt, max_new_tokens=MAX_NEW_TOKENS)

        g1 = keyword_judge.score(response)
        g2 = llm_judge.score(model, tokenizer, item["request"], response)

        records.append(
            {
                "id": item["id"],
                "category": item["category"],
                "framing": framing,
                "response": response,
                "g1_verdict": g1["verdict"],
                "g2_verdict": g2["verdict"],
            }
        )
    return records


def main():
    torch.set_num_threads(8)
    model, tokenizer = load_model_and_tokenizer()

    items = [json.loads(line) for line in EVAL_SET_PATH.open()]

    t0 = time.time()
    with STAGE1_RESULTS_PATH.open("w") as out_f:
        for i, item in enumerate(items):
            for record in run_item(model, tokenizer, item):
                out_f.write(json.dumps(record) + "\n")
            out_f.flush()
            elapsed = time.time() - t0
            print(f"[{i + 1}/{len(items)}] {item['id']} done ({elapsed:.0f}s elapsed)")

    print(f"wrote results to {STAGE1_RESULTS_PATH}")


if __name__ == "__main__":
    main()
