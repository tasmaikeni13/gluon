import json
from pathlib import Path

def main():
    print("=== COMPLETE LR CURVE FOR SOPHIA AND ROCORA (600M tokens, rho=0.02) ===")
    trials = [
        ("5.0e-5", "outputs/tuning/scale125m-lowlr/sophia-lowlr-05e-4", "outputs/tuning/scale125m-lowlr/rocora-lowlr-05e-4"),
        ("1.0e-4", "outputs/tuning/scale125m-lowlr/sophia-lowlr-10e-4", "outputs/tuning/scale125m-lowlr/rocora-lowlr-10e-4"),
        ("1.5e-4", "outputs/tuning/scale125m-lowlr/sophia-lowlr-15e-4", "outputs/tuning/scale125m-lowlr/rocora-lowlr-15e-4"),
        ("2.0e-4", "outputs/tuning/scale125m-budget-600m/sophia-t00", "outputs/tuning/scale125m-budget-600m/rocora-t00"),
        ("4.0e-4", "outputs/tuning/scale125m-budget-600m/sophia-t01", "outputs/tuning/scale125m-budget-600m/rocora-t01"),
        ("8.0e-4", "outputs/tuning/scale125m-budget-600m/sophia-t02", "outputs/tuning/scale125m-budget-600m/rocora-t02"),
    ]

    print(f"{'LR':<8} | {'Sophia Loss':<12} | {'Sophia PPL':<10} | {'Rocora Loss':<12} | {'Rocora PPL':<10} | {'Status':<16}")
    print("-" * 78)
    for lr, s_p, r_p in trials:
        s_lines = [json.loads(l) for l in (Path(s_p)/"metrics.jsonl").read_text().splitlines() if l.strip()]
        r_lines = [json.loads(l) for l in (Path(r_p)/"metrics.jsonl").read_text().splitlines() if l.strip()]
        s_eval = [l for l in s_lines if l.get("type") == "eval"][-1]
        r_eval = [l for l in r_lines if l.get("type") == "eval"][-1]

        status = "Stable"
        if s_eval["loss"] > 5.0 and r_eval["loss"] < 4.0:
            status = "Rocora rescued"
        elif s_eval["loss"] > 5.0 and r_eval["loss"] > 5.0:
            status = "Both diverged"

        print(f"{lr:<8} | {s_eval['loss']:>11.4f} | {s_eval['perplexity']:>9.2f} | {r_eval['loss']:>11.4f} | {r_eval['perplexity']:>9.2f} | {status:<16}")

    print("\n=== SUMMARY OF ALL 5 OPTIMIZERS BEST HYPERPARAMETERS (600M TOKENS) ===")
    all_bests = [
        ("AdamW", "outputs/tuning/scale125m-budget-600m/adamw-t05", "LR=0.0012, beta2=0.99"),
        ("Sophia-G", "outputs/tuning/scale125m-lowlr/sophia-lowlr-10e-4", "LR=0.00010, rho=0.02"),
        ("Rocora (matched)", "outputs/tuning/scale125m-budget-600m/rocora-t00", "LR=0.00020, rho=0.02 (best at 2e-4: 3.5753; at 1.0e-4: 3.6468)"),
        ("SOAP", "outputs/tuning/scale125m-budget-600m/soap-t02", "LR=0.003, precond_freq=10"),
        ("Muon", "outputs/tuning/scale125m-budget-600m/muon-t03", "LR=0.02, momentum=0.95"),
    ]
    for opt, p, hparams in all_bests:
        lines = [json.loads(l) for l in (Path(p)/"metrics.jsonl").read_text().splitlines() if l.strip()]
        ev = [l for l in lines if l.get("type") == "eval"][-1]
        print(f"{opt:<18} | Loss: {ev['loss']:.4f} | PPL: {ev['perplexity']:>6.2f} | Time: {ev['train_seconds']:>6.1f}s | {hparams}")

if __name__ == "__main__":
    main()
