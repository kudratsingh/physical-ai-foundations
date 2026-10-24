"""The coefficient choice, reset partitions and historical models stay frozen."""

import json

import pytest

from learning.maniskill.experiment import save_json
from learning.maniskill.normalized_ppo import sha256
from learning.maniskill.strength_experiment import (
    COEFFICIENTS,
    SEEDS,
    TASK,
    check_sources,
    declare,
    train,
)


def reference_fixture(tmp_path):
    teacher = tmp_path / "teacher.pt"
    teacher.write_bytes(b"fixture teacher; protocol tests never load a policy")
    reference = tmp_path / "reference"
    save_json(
        reference / "protocol.json",
        {
            "teacher_sha256": sha256(teacher),
            "test_seeds": list(range(240000, 240050)),
        },
    )
    for seed in SEEDS:
        directory = reference / f"seed{seed}" / "anchored"
        point = {
            "timesteps": 196608,
            "success_at_end_rate": 0.75,
            "success_once_rate": 0.8,
            "return_mean": 30.0,
        }
        save_json(directory / "selected.json", point)
        save_json(directory / "validation.json", [point])
        (directory / "best.zip").write_bytes(f"fixture seed{seed}".encode())
    return teacher, reference


def test_declaration_preserves_selected_models_and_freezes_only_fixed_strengths(
    tmp_path,
):
    teacher, reference = reference_fixture(tmp_path)
    run = tmp_path / "run"
    declare(run, teacher, reference)
    p = check_sources(run, teacher)
    assert p["coefficients"] == COEFFICIENTS == {"lambda10": 10.0, "lambda25": 25.0}
    assert p["training_seeds"] == SEEDS and TASK.total_timesteps == 262144
    assert set(p["test_seeds"]).isdisjoint(range(240000, 240050))
    assert p["validation_seeds"] == list(range(230000, 230040))
    manifest = json.loads((run / "reference_pr18_selected/manifest.json").read_text())
    assert manifest["evaluation_in_this_study"] is False
    assert manifest["models"] == p["historical_reference"]["selected_models"]
    for row in manifest["models"]:
        seed = row["training_seed"]
        original = reference / f"seed{seed}/anchored/best.zip"
        copied = run / f"reference_pr18_selected/seed{seed}_best.zip"
        assert copied.read_bytes() == original.read_bytes()
        assert sha256(copied) == row["checkpoint_sha256"]
    with pytest.raises(FileExistsError):
        declare(run, teacher, reference)
    with pytest.raises(ValueError, match="Undeclared training condition"):
        train(run, teacher, 42, "lambda50")
    assert not (run / "seed42").exists()


@pytest.mark.parametrize("mutation", ["coefficient", "teacher", "source", "protocol"])
def test_tampering_is_rejected_before_worker_construction(tmp_path, mutation):
    teacher, reference = reference_fixture(tmp_path)
    run = tmp_path / "run"
    declare(run, teacher, reference)
    if mutation in ("coefficient", "protocol"):
        path = run / "protocol.json"
        p = json.loads(path.read_text())
        if mutation == "coefficient":
            p["coefficients"]["lambda25"] = 50.0
        else:
            p["test_seeds"][0] = 240000
        save_json(path, p)
    elif mutation == "teacher":
        teacher.write_bytes(b"changed")
    else:
        (run / "launch_sources/retention_ppo.py").write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        train(run, teacher, 42, "lambda25")
    assert not (run / "seed42").exists()
