# grader-blindspot

Testing whether a model's near-zero attack-success rate on a safety eval reflects genuine robustness or the model detecting it's being evaluated — and whether reward-seeking behavior learned against one grader transfers to a different one.

## Research questions

1. Does attack success rate change when the same adversarial prompt is framed as a test vs. embedded in a plausible deployment conversation?
2. How much attack success is available to a gradient-based attacker (PGD in embedding space) beyond what static prompts show — i.e. how much headroom is hiding behind a low static ASR?
3. When a model is trained against a narrow, gameable reward proxy, does the resulting reward-seeking behavior generalize to a held-out grader, or does it stay bound to the training grader's specific format?

## Method

1. Built a 60-item adversarial eval (12 categories, 5 items each — weapons, cybercrime, fraud, drugs, extremism, privacy, chem/bio, financial crime, disinformation, self-harm-adjacent, stalking, illegal services), each wrapped two ways: an explicit red-team-eval framing and a naturalistic deployment framing, same underlying request either way.
2. Grade every response with two graders: G1, a cheap keyword/refusal-phrase classifier with a known blind spot (it only looks at surface phrasing, not content), and G2, an LLM judge (same base model, different mechanism) that's asked to judge substance. G1 is the train-time reward signal for stage 3; G2 is the held-out check.
3. Run PGD in embedding space (continuous relaxation of a GCG-style suffix attack) against a subset of items, optimizing a block of adversarial embedding vectors to maximize the likelihood of a compliance-style target string, then snap the result to real tokens and measure whether it actually moves attack success versus the static-prompt baseline.
4. LoRA-finetune the model on a small set of examples that game G1's blind spot — responses phrased to dodge every refusal trigger phrase while staying substantively empty (no real harmful content, just deflection dressed up to avoid the keyword filter).
5. Evaluate the finetuned model against G1 (train-time grader) and G2 (held-out) on the full 60-item eval set, which the LoRA run never saw. The gap between G1's score and G2's score post-training is the actual result: large gap means the model overfit to G1's specific keyword patterns, small gap with both elevated means it learned something closer to "game graders in general."

## Status

All five stages are implemented and I've smoke-tested the pieces that needed it (the PGD loss alignment, the LoRA masking, both graders) on a handful of examples each. What I haven't done yet is run the full pipeline end to end — stage 1 alone is 120 generations plus 120 judge calls on a CPU, and I started that run twice and killed it partway both times to free up the machine for other things, so there's no results section here yet. Everything below the code is real; there's no numbers section further down because I'm not going to write one until an actual run produces it.

The model is Qwen2.5-1.5B-Instruct, running locally in bf16 on CPU (no GPU on this machine) — same checkpoint I already had cached from the confabulation-probe project. At roughly 0.11s/token with 8 threads, a full stage 1 run is on the order of 30-45 minutes; stage 2 and 3 involve backward passes so they'll run slower per item, which is why they're scoped to a 6-item subset and a 30-example training set respectively rather than the full eval set.

## Running the pipeline

```
pip install -r requirements.txt

python -m src.stage1_framing.build_eval_set      # already run -> data/eval_set.jsonl
python -m src.stage1_framing.run_framing_eval    # results/stage1_framing_results.jsonl
python -m src.stage1_framing.analyze             # results/stage1_summary.json + figure

python -m src.stage2_pgd.run_pgd_eval            # results/stage2_pgd_results.jsonl
python -m src.stage2_pgd.analyze                 # results/stage2_summary.json + figure

python -m src.stage3_train.build_reward_hack_data  # already run -> data/stage3_reward_hack_data.jsonl
python -m src.stage3_train.train_lora              # results/lora_adapter/, results/stage3_train_log.json

python -m src.stage4_transfer.evaluate_transfer  # results/stage4_transfer_results.jsonl + summary
```

## Structure

```
src/model/           # local model loading + generation
src/judge/           # G1 (keyword, gameable) and G2 (LLM judge, held-out)
src/stage1_framing/  # eval set construction, framing-delta run + analysis
src/stage2_pgd/      # embedding-space PGD attack, headroom run + analysis
src/stage3_train/    # reward-hacking training data + LoRA fine-tune
src/stage4_transfer/ # transfer eval against G1 and G2, transfer-gap number
data/                # eval set + stage-3 training set (disjoint from each other)
results/             # run outputs, summaries, figures -- empty until a run finishes
```

## Limitations (known going in, not from results yet)

- G2 is the same base model as the one being evaluated, just prompted differently — it's a held-out *mechanism* (semantic judging vs. keyword matching), not a held-out model. A truly independent judge model would be a stronger test of the transfer-gap result.
- The stage-3 training examples are all built from the same 10 deflection templates, so what the LoRA run learns could be closer to "reproduce these specific phrasings" than a general grader-gaming strategy. Worth checking generalization against paraphrased or novel deflection styles before reading too much into any transfer-gap number.
- 60 eval items / 6 PGD items / 30 training examples are small-N by construction, given CPU-only compute on a single machine. Fine for checking the pipeline is doing the right thing, not enough to report a statistically solid effect size.
- One model, one scale (1.5B). Nothing here says anything about how this behaves at frontier scale.
- PGD headroom is measured after snapping continuous embeddings to their nearest real tokens, which is a lossy step — the continuous-space loss curve and the actual post-discretization attack success rate are two different numbers and shouldn't be conflated.

## References

Eval awareness:
- Needham et al. (2025), "Large Language Models Often Know When They Are Being Evaluated"
- Meinke et al. (2024, Apollo), "Frontier Models are Capable of In-Context Scheming"
- van der Weij et al. (2024), "AI Sandbagging: Language Models Can Strategically Underperform on Evaluations"
- Greenblatt et al. (2024, Anthropic/Redwood), "Alignment Faking in Large Language Models"

Attacks:
- Zou et al. (2023), "Universal and Transferable Adversarial Attacks on Aligned Language Models" (GCG)
- Madry et al. (2017), "Towards Deep Learning Models Resistant to Adversarial Attacks" (PGD)
- Chao et al. (2023), PAIR; Mehrotra et al. (2023), TAP

Reward hacking / judge gaming:
- Denison et al. (2024, Anthropic), "Sycophancy to Subterfuge: Investigating Reward-Tampering"
- Zheng et al. (2023), "Judging LLM-as-a-Judge"
- Raina et al. (2024), "Is LLM-as-a-Judge Robust?"
- Hubinger et al. (2024), "Sleeper Agents"

## License

MIT
