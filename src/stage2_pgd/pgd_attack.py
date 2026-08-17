import torch

ADV_INIT_TOKEN = "!"


def _segment_embeds(model, tokenizer, text, add_special_tokens=False):
    ids = tokenizer(text, add_special_tokens=add_special_tokens, return_tensors="pt")["input_ids"]
    embed_matrix = model.get_input_embeddings().weight
    embeds = embed_matrix[ids[0]].unsqueeze(0)
    return ids, embeds


def build_attack_inputs(model, tokenizer, request, target, n_adv_tokens=20):
    """Lay out prefix / adversarial suffix / target as one embedding sequence.

    prefix = chat-templated user turn up to the request, adv = a block of
    free-floating embedding vectors we optimize, target = the compliance
    string we're teacher-forcing the model toward. Loss is next-token CE
    restricted to the target span, so gradients only flow back into the
    adversarial block.
    """
    prefix_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": request + " "}],
        add_generation_prompt=True,
        tokenize=False,
    )
    prefix_ids, prefix_embeds = _segment_embeds(model, tokenizer, prefix_text, add_special_tokens=False)
    target_ids, target_embeds = _segment_embeds(model, tokenizer, target, add_special_tokens=False)

    embed_matrix = model.get_input_embeddings().weight
    init_ids = tokenizer(ADV_INIT_TOKEN, add_special_tokens=False)["input_ids"][0]
    init_adv_embeds = embed_matrix[init_ids].detach().clone().repeat(n_adv_tokens, 1).unsqueeze(0)

    return {
        "prefix_embeds": prefix_embeds.detach(),
        "init_adv_embeds": init_adv_embeds,
        "target_ids": target_ids,
        "target_embeds": target_embeds.detach(),
    }


def pgd_step(model, prefix_embeds, adv_embeds, target_embeds, target_ids, step_size, epsilon, delta):
    adv = (adv_embeds + delta).detach().requires_grad_(True)

    full_embeds = torch.cat([prefix_embeds, adv, target_embeds[:, :-1, :]], dim=1)
    n_prefix_adv = prefix_embeds.shape[1] + adv.shape[1]

    labels = torch.full((1, full_embeds.shape[1]), -100, dtype=torch.long)
    labels[0, n_prefix_adv:] = target_ids[0, 1:]

    out = model(inputs_embeds=full_embeds, labels=labels)
    loss = out.loss
    loss.backward()

    with torch.no_grad():
        grad_sign = adv.grad.sign()
        new_delta = delta - step_size * grad_sign
        new_delta = new_delta.clamp(-epsilon, epsilon)

    return float(loss.item()), new_delta


def run_pgd(model, tokenizer, request, target, n_adv_tokens=20, n_steps=40, step_size=0.01, epsilon=0.3):
    inputs = build_attack_inputs(model, tokenizer, request, target, n_adv_tokens)
    delta = torch.zeros_like(inputs["init_adv_embeds"])

    loss_curve = []
    for step in range(n_steps):
        loss, delta = pgd_step(
            model,
            inputs["prefix_embeds"],
            inputs["init_adv_embeds"],
            inputs["target_embeds"],
            inputs["target_ids"],
            step_size,
            epsilon,
            delta,
        )
        loss_curve.append(loss)

    return {"loss_curve": loss_curve, "final_adv_embeds": inputs["init_adv_embeds"] + delta}
