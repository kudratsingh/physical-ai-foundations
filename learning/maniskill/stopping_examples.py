"""Illustrative activation videos selected after the declared diagnostic.

The aggregate comparison is unchanged. These examples show the first activation
per policy, not a success-rate sample or predeclared performance evaluation.
"""

import argparse
import json
from pathlib import Path
import shutil

from stable_baselines3 import PPO
from .experiment import save_json
from .normalized_ppo import FrozenPolicy, preserve_rng, sha256
from .stopping_diagnostic import rollout


def export(output, evidence, raw_checkpoint, normalized_snapshot):
    output, evidence = Path(output), Path(evidence)
    output.mkdir(parents=True, exist_ok=False)
    selection = json.loads((evidence / "plot_examples.json").read_text())
    protocol = dict(
        scope=__doc__,
        selection=selection,
        raw_checkpoint_sha256=sha256(raw_checkpoint),
        normalized_snapshot=json.loads(
            (Path(normalized_snapshot) / "snapshot.json").read_text()
        ),
        source_sha256=sha256(__file__),
    )
    save_json(output / "protocol.json", protocol)
    policies = {
        "raw_final": PPO.load(raw_checkpoint, device="cpu"),
        "normalized_final": FrozenPolicy.load(normalized_snapshot),
    }
    exports = []
    try:
        with preserve_rng():
            for name, policy in policies.items():
                seed = selection["seeds"][name]
                for mode in ["normal", "override"]:
                    filename = f"activation_{name}_{mode}"
                    result = rollout(
                        policy,
                        [seed],
                        override=(mode == "override"),
                        output=output / f"{filename}.json",
                        video=True,
                    )
                    reference = json.loads(
                        (evidence / f"{name}_{mode}.json").read_text()
                    )
                    expected = next(
                        e for e in reference["episodes"] if e["seed"] == seed
                    )
                    if result["episodes"][0] != expected:
                        raise ValueError(
                            "Activation video changed a recorded diagnostic trajectory"
                        )
                    video = output / f"{filename}_seed{seed}.mp4"
                    shutil.copy2(video, evidence / video.name)
                    exports.append(
                        dict(
                            policy=name,
                            seed=seed,
                            mode=mode,
                            video=video.name,
                            video_sha256=sha256(video),
                            all_recorded_fields_match=True,
                            reference=f"{name}_{mode}.json",
                            active_steps=expected["active_steps"],
                            success_at_end=expected["success_at_end"],
                        )
                    )
    finally:
        policies["normalized_final"].close()
    save_json(
        evidence / "activation_examples.json", dict(protocol=protocol, videos=exports)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ["output", "evidence", "raw-checkpoint", "normalized-snapshot"]:
        parser.add_argument("--" + arg, type=Path, required=True)
    args = parser.parse_args()
    export(args.output, args.evidence, args.raw_checkpoint, args.normalized_snapshot)


if __name__ == "__main__":
    main()
