"""Fetch pinned official demonstrations and invoke ManiSkill's CPU replay tool.

No expert checkpoint is loaded. Only HDF5 trajectories are replayed. The raw
archive remains intact; a working metadata copy selects CPU-compatible rendering.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from urllib.request import urlopen
import zipfile
from .bc_data import sha256_file
from .experiment import save_json

ARCHIVE_SHA256 = "1ce24408bf93658501faaa0ef164a0cdc29539cc412612a413fac746697eeab2"
URL = "https://huggingface.co/datasets/haosulab/ManiSkill_Demonstrations/resolve/main/demos/PushCube-v1.zip?download=true"
ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-id", choices=["PushCube-v1", "PickCube-v1"], default="PushCube-v1"
    )
    parser.add_argument("--count", type=int, default=512)
    parser.add_argument("--output", type=Path, default=ROOT / ".assets/bc_replay")
    args = parser.parse_args()
    limit = 997 if args.env_id == "PickCube-v1" else 1023
    url = URL.replace("PushCube-v1", args.env_id)
    checksum = (
        "b2d4afb30fa309755862b98c342e6ee18918253c93f3bbac16ed6670748f26d8"
        if args.env_id == "PickCube-v1"
        else ARCHIVE_SHA256
    )
    if not 1 <= args.count <= limit:
        parser.error(f"count must be 1 to {limit}")
    raw = ROOT / ".assets/demos"
    raw.mkdir(parents=True, exist_ok=True)
    archive = raw / f"{args.env_id}.zip"
    if not archive.exists():
        temporary = archive.with_suffix(".zip.tmp")
        with urlopen(url, timeout=120) as response, temporary.open("wb") as destination:
            shutil.copyfileobj(response, destination)
        temporary.replace(archive)
    if sha256_file(archive) != checksum:
        raise ValueError("Official demonstration archive checksum mismatch")
    with zipfile.ZipFile(archive) as package:
        for member in package.infolist():
            target = (raw / member.filename).resolve()
            if not target.is_relative_to(raw.resolve()):
                raise ValueError("Archive path escapes data directory")
        package.extractall(raw)
    source = raw / args.env_id / "rl/trajectory.none.pd_joint_delta_pos.physx_cuda.h5"
    metadata = source.with_suffix(".json")
    args.output.mkdir(parents=True, exist_ok=False)
    working = args.output / source.name
    shutil.copy2(source, working)
    data = json.loads(metadata.read_text())
    data["env_info"]["env_kwargs"]["render_backend"] = "cpu"
    save_json(working.with_suffix(".json"), data)
    environment = os.environ.copy()
    environment["MS_ASSET_DIR"] = str(ROOT / ".assets")
    icd = ROOT / ".runtime/MoltenVK_icd.json"
    if sys.platform == "darwin" and icd.exists():
        environment.setdefault("VK_ICD_FILENAMES", str(icd))
    command = [
        sys.executable,
        "-m",
        "mani_skill.trajectory.replay_trajectory",
        "--traj-path",
        str(working),
        "--sim-backend",
        "physx_cpu",
        "--obs-mode",
        "state",
        "--target-control-mode",
        "pd_joint_delta_pos",
        "--use-first-env-state",
        "--save-traj",
        "--count",
        str(args.count),
        "--num-envs",
        "1",
        "--reward-mode",
        "normalized_dense",
    ]
    subprocess.run(command, env=environment, check=True)
    result = args.output / "trajectory.state.pd_joint_delta_pos.physx_cpu.h5"
    replayed = json.loads(result.with_suffix(".json").read_text())
    save_json(
        args.output / "provenance.json",
        dict(
            url=url,
            archive_sha256=checksum,
            source_hdf5_sha256=sha256_file(source),
            source_metadata_sha256=sha256_file(metadata),
            replayed_hdf5_sha256=sha256_file(result),
            replayed_metadata_sha256=sha256_file(result.with_suffix(".json")),
            attempted=args.count,
            retained=len(replayed["episodes"]),
            preparation="working metadata copy sets render_backend=cpu; first source state restored; actual actions replayed; successful CPU replays retained",
            command=command,
        ),
    )
    print("CPU demonstration dataset:", result, flush=True)


if __name__ == "__main__":
    main()
