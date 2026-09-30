#!/usr/bin/env python3
from __future__ import annotations
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts" / "run_frozen_untargeted_defense_v320b1.py"
OUTPUT = ROOT / "scripts" / "run_reviewer_reconstruction_ablation_v4340c.py"
EXPECTED = "5f3852fc13959301b31adf47f57df3b65456fb028a647978b355f976e5aee951"

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()

def one(s, old, new, label):
    n = s.count(old)
    if n != 1:
        raise RuntimeError(f"{label}: expected one match, found {n}")
    return s.replace(old, new, 1)

def main():
    got = sha(SOURCE)
    if got != EXPECTED:
        raise RuntimeError(f"base hash mismatch: {got}")
    s = SOURCE.read_text(encoding="utf-8")

    old = '''    parser.add_argument(
        "--replacement-policy",
        choices=["plain_fedavg", "trusted_reconstruction"],
        required=True,
    )
'''
    new = old + '''    parser.add_argument(
        "--ablation-arm",
        choices=["center_only", "hard_rejection"],
        required=True,
    )
'''
    s = one(s, old, new, "parser")

    start = s.index('    ema_threshold_q99 = float(threshold)')
    end = s.index('    class_weights = sqrt_class_weights(y_train, args.max_class_weight)', start)
    repl = '''    ema_threshold_q99 = float(threshold)
    instant_threshold_q95 = float("nan")
    selected_reconstruction_policy = ""
    residual_profiles: Dict[int, Dict[str, torch.Tensor]] = {}
    warmup_update_scale = float("nan")
    scale_lower = float("nan")
    scale_upper = float("nan")
    norm_clip_multiplier = float("nan")
    reconstruction_checkpoint_path: Path | None = None

    if args.replacement_policy == "trusted_reconstruction":
        warmup_scores_path = (
            warmup_dir / "calibration"
            / "clean_leave_one_round_out_scores.csv"
        )
        if not warmup_scores_path.exists():
            raise FileNotFoundError(warmup_scores_path)
        warmup_scores = pd.read_csv(warmup_scores_path)
        ema_threshold_q99 = quantile_higher(
            warmup_scores[SCORE_COLUMN].to_numpy(dtype=np.float64),
            FROZEN_EMA_QUANTILE,
        )
        instant_threshold_q95 = quantile_higher(
            warmup_scores[CANDIDATE].to_numpy(dtype=np.float64),
            FROZEN_INSTANT_QUANTILE,
        )

        if args.ablation_arm == "center_only":
            selected_reconstruction_policy = "center_only"
            residual_profiles = {client_id: {} for client_id in range(args.num_clients)}
        elif args.ablation_arm == "hard_rejection":
            selected_reconstruction_policy = "hard_rejection"
            residual_profiles = {client_id: {} for client_id in range(args.num_clients)}
        else:
            raise RuntimeError(f"Unsupported ablation arm: {args.ablation_arm}")

'''
    s = s[:start] + repl + s[end:]

    old = '''                replaced = bool(
                    flags_for_reconstruction[client_id]
                )
                reconstruction_meta = {
'''
    new = '''                flagged_for_mitigation = bool(
                    flags_for_reconstruction[client_id]
                )
                replaced = bool(
                    flagged_for_mitigation
                    and args.ablation_arm == "center_only"
                )
                rejected = bool(
                    flagged_for_mitigation
                    and args.ablation_arm == "hard_rejection"
                )
                reconstruction_meta = {
'''
    s = one(s, old, new, "mitigation flags")

    old = '''                    "update_replaced": replaced,
                    "selected_reconstruction_policy": (
'''
    new = '''                    "update_replaced": replaced,
                    "update_rejected": rejected,
                    "ablation_arm": args.ablation_arm,
                    "selected_reconstruction_policy": (
'''
    s = one(s, old, new, "audit rows")

    old = '''        model.load_state_dict(
            weighted_average_states(
                replacement_states,
                sample_counts,
                reference_state,
            )
        )
'''
    new = '''        if args.ablation_arm == "hard_rejection":
            aggregation_states = [
                replacement_states[client_id]
                for client_id in trusted_ids
            ]
            aggregation_sample_counts = [
                sample_counts[client_id]
                for client_id in trusted_ids
            ]
        else:
            aggregation_states = replacement_states
            aggregation_sample_counts = sample_counts

        model.load_state_dict(
            weighted_average_states(
                aggregation_states,
                aggregation_sample_counts,
                reference_state,
            )
        )
'''
    s = one(s, old, new, "aggregation")

    old = '''            "replacement_policy": args.replacement_policy,
            "participating_samples": int(sum(sample_counts)),
            "replaced_clients": int(replaced_clients),
'''
    new = '''            "replacement_policy": args.replacement_policy,
            "ablation_arm": args.ablation_arm,
            "mitigation_action": (
                "coordinate_median_center_reconstruction"
                if args.ablation_arm == "center_only"
                else "hard_rejection_and_weight_renormalization"
            ),
            "participating_samples": int(sum(sample_counts)),
            "replaced_clients": int(replaced_clients),
            "rejected_clients": int(
                flags.sum() if args.ablation_arm == "hard_rejection" else 0
            ),
            "aggregation_retained_clients": int(
                len(trusted_ids)
                if args.ablation_arm == "hard_rejection"
                else args.num_clients
            ),
'''
    s = one(s, old, new, "round metadata")

    old = '''                "replacement_policy": args.replacement_policy,
                "monitoring_round": int(monitoring_round),
'''
    new = '''                "replacement_policy": args.replacement_policy,
                "ablation_arm": args.ablation_arm,
                "monitoring_round": int(monitoring_round),
'''
    s = one(s, old, new, "checkpoint metadata")

    old = '''        "replacement_policy": args.replacement_policy,
        "model_seed": int(args.model_seed),
'''
    new = '''        "replacement_policy": args.replacement_policy,
        "ablation_arm": args.ablation_arm,
        "mitigation_action": (
            "coordinate_median_center_reconstruction"
            if args.ablation_arm == "center_only"
            else "hard_rejection_and_weight_renormalization"
        ),
        "reconstruction_calibration_required": False,
        "reconstruction_calibration_used": False,
        "historical_residual_used": False,
        "hard_rejection_used": bool(args.ablation_arm == "hard_rejection"),
        "center_only_used": bool(args.ablation_arm == "center_only"),
        "detector_algorithm_changed": False,
        "detector_thresholds_changed": False,
        "model_seed": int(args.model_seed),
'''
    s = one(s, old, new, "final metadata")

    s = s.replace('"experiment_version": "3.20B.1"', '"experiment_version": "4.34.0c-RECONSTRUCTION-ABLATION"')
    s = s.replace('"phase": "exact_derived_frozen_untargeted_defense"', '"phase": "reviewer_reconstruction_mitigation_ablation"')

    old = '''    print("Attack type:", args.attack_type)
    print("Mean validation macro F1:", f"{round_table['val_macro_f1'].mean():.6f}")
'''
    new = '''    print("Attack type:", args.attack_type)
    print("Ablation arm:", args.ablation_arm)
    print("Reconstruction calibration used: False")
    print("Historical residual used: False")
    print("Mean validation macro F1:", f"{round_table['val_macro_f1'].mean():.6f}")
'''
    s = one(s, old, new, "completion print")

    header = '''# REVIEWER-DERIVED FILE v4.34.0c
# Derived from frozen V3.20B.1 SHA256 5F3852FC13959301B31ADF47F57DF3B65456FB028A647978B355F976E5AEE951
# No reconstruction-calibration bundle is used because neither ablation requires historical residuals.
'''
    OUTPUT.write_text(header + s, encoding="utf-8", newline="\n")
    print("DERIVED ABLATION RUNNER CREATED")
    print("BASE SHA256:", got.upper())
    print("DERIVED SHA256:", sha(OUTPUT).upper())
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
