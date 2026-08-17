import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import LOCAL_MODEL_PATH


def load_model_and_tokenizer(adapter_path=None):
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL_PATH, dtype=torch.bfloat16, low_cpu_mem_usage=True
    )

    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate_reply(model, tokenizer, prompt, max_new_tokens=200, temperature=0.0):
    messages = [{"role": "user", "content": prompt}]
    encoded = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    prompt_len = encoded["input_ids"].shape[1]

    output_ids = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = output_ids[0, prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
