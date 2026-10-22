"""Evaluation math: distinguish temporary success from success at episode end."""

import pytest
from learning.maniskill.metrics import summarize, wilson_interval


def test_success_definitions_and_uncertainty():
    episodes = [
        dict(
            seed=10000,
            **{"return": 10},
            steps=50,
            success_once=True,
            success_at_end=False,
        ),
        dict(
            seed=10001,
            **{"return": 4},
            steps=50,
            success_once=False,
            success_at_end=False,
        ),
    ]
    s = summarize(episodes)
    assert s["success_once_rate"] == 0.5
    assert s["success_at_end_rate"] == 0
    assert s["return_mean"] == 7
    assert s["success_once_95pct_wilson"][0] < 0.5 < s["success_once_95pct_wilson"][1]
    assert wilson_interval(0, 20)[1] > 0
    assert wilson_interval(20, 20)[0] < 1


def test_rejects_empty_duplicate_or_nonfinite_evaluation():
    with pytest.raises(ValueError):
        summarize([])
    e = dict(
        seed=1, **{"return": 1}, steps=50, success_once=False, success_at_end=False
    )
    with pytest.raises(ValueError, match="unique"):
        summarize([e, e])
    with pytest.raises(ValueError, match="finite"):
        summarize([{**e, "return": float("nan")}])


@pytest.mark.parametrize(
    "folder,names",
    [
        ("first_run", ["ppo", "random", "zero"]),
        ("week4_bc", ["bc_32", "bc_256", "ppo", "random", "zero"]),
    ],
)
def test_committed_results_agree_with_episode_evidence(folder, names):
    import json
    import math
    from pathlib import Path

    evidence = (
        Path(__file__).resolve().parents[2] / "learning/maniskill/evidence" / folder
    )
    comparison = json.loads((evidence / "comparison.json").read_text())
    for name in names:
        result = json.loads((evidence / f"test_{name}.json").read_text())
        assert summarize(result["episodes"]) == comparison[name]
        for episode in result["episodes"]:
            cube, goal = episode["final_cube_position"], episode["final_goal_position"]
            geometric_success = math.dist(cube[:2], goal[:2]) < 0.1 and cube[2] < 0.025
            assert geometric_success == episode["success_at_end"]
