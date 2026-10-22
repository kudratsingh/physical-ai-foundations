"""Real simulator contract tests; run in the isolated ManiSkill environment."""

import numpy as np
from learning.maniskill.experiment import Config, env_thunk, evaluate


def test_monitor_only_emits_sb3_episode_metadata_at_truncation():
    # ManiSkill's optional per-step info['episode'] conflicts with SB3 Monitor.
    # The experiment aggregates success manually and leaves Monitor owning this key.
    env = env_thunk(Config())()
    try:
        obs, _ = env.reset(seed=999)
        assert obs.shape == (35,)
        assert env.action_space.shape == (8,)
        for step in range(50):
            obs, reward, terminated, truncated, info = env.step(
                np.zeros(8, dtype=np.float32)
            )
            assert np.all(np.isfinite(obs)) and np.isfinite(reward)
            assert not terminated
            if step < 49:
                assert not truncated and "episode" not in info
            else:
                assert truncated and {"r", "l", "t"} <= info["episode"].keys()
                assert info["episode"]["l"] == 50
    finally:
        env.close()


def test_seeded_random_rollouts_repeat_and_use_full_horizon(tmp_path):
    first = evaluate(
        Config(), "random", [10000, 10001], tmp_path / "first.json", traces=True
    )
    second = evaluate(Config(), "random", [10000, 10001])
    assert first["summary"] == second["summary"]
    for a, b in zip(first["episodes"], second["episodes"]):
        assert a["return"] == b["return"]
        assert a["steps"] == b["steps"] == 50
        assert len(a["trajectory"]) == 51
        assert a["final_cube_position"] == a["trajectory"][-1]["cube"]
        assert a["final_cube_position"] == b["final_cube_position"]
