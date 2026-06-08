"""``python -m citanet.data.inspect <scenario_id>`` -- parse a scenario from
its STKG, validate against the observations cache, and print a summary.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .loader import load_scenario, validate_consistency

DEFAULT_DATA_ROOT = Path("data/battlefield_stkg_dataset")


def inspect(scenario_id: str, data_root: Path) -> None:
    scenario_dir = data_root / "scenarios" / scenario_id
    ontology_dir = data_root / "ontology"
    scn = load_scenario(scenario_dir, ontology_dir)
    validate_consistency(scenario_dir, scn)

    gold = json.loads((scenario_dir / "labels" / "gold_identities.json").read_text(encoding="utf-8"))
    n_match = len(gold["identities"])
    n_dangling_obs = len(gold["dangling_observations"])
    n_dangling_ent = len(gold["dangling_local_entities"])
    a = scn.n_entities("A")
    b = scn.n_entities("B")
    type_counts = Counter(o.type for o in scn.observations)
    src_counts = Counter(o.source for o in scn.observations)

    print(f"scenario        : {scenario_id}")
    print(f"epoch           : {scn.epoch_iso}")
    print(f"observations    : {len(scn.observations)}")
    print(f"entities        : A={a} / B={b}")
    print(f"matched ids     : {n_match}")
    print(f"dangling        : {n_dangling_ent} entities / {n_dangling_obs} observations")
    print(f"obs types       : {dict(type_counts)}")
    print(f"obs sources     : {dict(src_counts)}")
    print(f"landmarks       : {list(scn.landmarks)}")
    print(f"events          : {list(scn.events)}")
    print("STKG <-> observations.jsonl consistency: OK")
    print()
    print("matched identities:")
    for ident in gold["identities"]:
        members = ident["member_local_entities"]
        print(f"  {ident['gold_identity_id']:10s} type={ident['true_type']:10s} "
              f"members={members} n_obs={len(ident['member_observations'])}")
    print("dangling (must abstain):")
    for d in gold["dangling_local_entities"]:
        print(f"  {d['kg_id']}:{d['local_entity_id']} type={d['true_type']} obs={d['obs_ids']}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect a CITA-Net scenario")
    ap.add_argument("scenario_id")
    ap.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    args = ap.parse_args()
    inspect(args.scenario_id, Path(args.data_root))


if __name__ == "__main__":
    main()
