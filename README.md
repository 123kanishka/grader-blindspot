# grader-blindspot

Testing whether a model's near-zero attack-success rate on a safety eval reflects genuine robustness or the model detecting it's being evaluated — and whether reward-seeking behavior learned against one grader transfers to a different one.

## Research questions

1. Does attack success rate change when the same adversarial prompt is framed as a test vs. embedded in a plausible deployment conversation?
2. When a model is trained against a narrow, gameable reward proxy, does the resulting reward-seeking behavior generalize to a held-out grader, or does it stay bound to the training grader's specific format?

## Method

1. Build a narrow (50–100 item) adversarial eval for one unambiguous behavior.
2. Run it under two framings — test-like vs. deployment-like — and measure the attack-success delta.
3. LoRA-finetune a small open model against a simple, exploitable automated grader (G1).
4. Evaluate the finetuned policy against G1 (train-time grader) and a held-out grader G2 with different phrasing/criteria.
5. The reward gap between G1 and G2 is the reward-seeking-generalization result.

## Status

🚧 Early stage — repo scaffolding only, no results yet.

## Structure (planned)

```
eval/            # adversarial prompt suite + grading logic
framings/        # test-like vs. deployment-like prompt variants
finetune/        # LoRA training against grader G1
transfer/        # held-out grader G2 + transfer-gap analysis
notebooks/       # exploratory analysis
```

## Model / compute

Open-weight ~8–9B model, LoRA fine-tuning (a few GPU-hours), eval-suite half requires no GPU.

## License

MIT
