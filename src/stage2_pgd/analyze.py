import json

from src.config import FIGURES_DIR, STAGE1_RESULTS_PATH, STAGE2_RESULTS_PATH, STAGE2_SUMMARY_PATH


def deployment_baseline_for(item_id):
    for line in STAGE1_RESULTS_PATH.open():
        r = json.loads(line)
        if r["id"] == item_id and r["framing"] == "deployment_framing":
            return r["g2_verdict"]
    return None


def main():
    results = [json.loads(line) for line in STAGE2_RESULTS_PATH.open()]

    n = len(results)
    pgd_g2_success = sum(1 for r in results if r["g2_verdict"] == "ATTACK_SUCCESS")
    pgd_g1_success = sum(1 for r in results if r["g1_verdict"] == "ATTACK_SUCCESS")

    baseline_g2_success = 0
    for r in results:
        baseline = deployment_baseline_for(r["id"])
        r["baseline_g2_verdict"] = baseline
        if baseline == "ATTACK_SUCCESS":
            baseline_g2_success += 1

    avg_loss_reduction = sum(r["loss_curve"][0] - r["loss_curve"][-1] for r in results) / n

    summary = {
        "n_items": n,
        "baseline_g2_asr": baseline_g2_success / n,
        "pgd_g2_asr": pgd_g2_success / n,
        "pgd_g1_asr": pgd_g1_success / n,
        "g2_headroom": pgd_g2_success / n - baseline_g2_success / n,
        "avg_target_loss_reduction": avg_loss_reduction,
        "per_item": [
            {
                "id": r["id"],
                "category": r["category"],
                "loss_start": r["loss_curve"][0],
                "loss_end": r["loss_curve"][-1],
                "baseline_g2_verdict": r["baseline_g2_verdict"],
                "pgd_g2_verdict": r["g2_verdict"],
            }
            for r in results
        ],
    }

    STAGE2_SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    plot_loss_curves(results)


def plot_loss_curves(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for r in results:
        ax.plot(r["loss_curve"], label=r["id"], alpha=0.8)
    ax.set_xlabel("PGD step")
    ax.set_ylabel("target compliance loss")
    ax.set_title("Stage 2: PGD headroom curves")
    ax.legend(fontsize=7)
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "stage2_pgd_headroom.png", dpi=150)
    print(f"saved figure to {FIGURES_DIR / 'stage2_pgd_headroom.png'}")


if __name__ == "__main__":
    main()
