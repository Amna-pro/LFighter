
from pathlib import Path
import csv
import hashlib

ROOT = Path.home() / "LFighter-research"
ATTACKS = [
    "all_to_one_benign",
    "cyclic_shift",
    "multiclass_partial_cycle",
    "pairwise_swap",
    "random_flip",
]
SEEDS = [7, 99, 123, 2026]

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def sha256_tree(paths) -> str:
    h = hashlib.sha256()
    found = False
    for path in sorted(paths, key=lambda p: str(p).lower()):
        found = True
        h.update(str(path.name).encode("utf-8"))
        h.update(sha256_file(path).encode("ascii"))
    return h.hexdigest() if found else "MISSING"

rows = []
for seed in SEEDS:
    for attack in ATTACKS:
        manifest_dir = (
            ROOT
            / "results"
            / "cic_iot_diad_untargeted_exact_qualification_v320a3"
            / "runs"
            / attack
            / f"seed_{seed}"
            / "plain_fedavg"
            / "attack_manifest"
        )
        defended_dir = (
            ROOT
            / "results"
            / "cic_iot_diad_frozen_untargeted_defense_v320b1"
            / "runs"
            / attack
            / f"seed_{seed}"
            / "trusted_reconstruction"
        )
        labels = manifest_dir / "poisoned_labels.npz"
        indices = manifest_dir / "poisoned_indices.npz"
        rows.append(
            {
                "seed": seed,
                "attack_type": attack,
                "labels_manifest_sha256": sha256_file(labels),
                "indices_manifest_sha256": sha256_file(indices),
                "defended_checkpoint_tree_sha256": sha256_tree(defended_dir.rglob("*.pt")),
                "defended_table_tree_sha256": sha256_tree(defended_dir.rglob("*.csv")),
            }
        )

out = (
    ROOT
    / "results"
    / "cic_iot_diad_frozen_untargeted_defense_v320b1"
    / "summary"
    / "tables"
    / "task40_integrity_audit.csv"
)
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

for row in rows:
    print(
        row["seed"],
        row["attack_type"],
        row["labels_manifest_sha256"][:12],
        row["defended_checkpoint_tree_sha256"][:12],
        row["defended_table_tree_sha256"][:12],
    )
print("WROTE:", out)
