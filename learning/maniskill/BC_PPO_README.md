# PickCube: placement diagnostics and BC-initialized PPO

This experiment separates observation-only stopping, precision placement, and
learning from an already useful behavior-cloned policy. The original PickCube
reward, control interface, horizon and success rule stay unchanged.

- [Observation-only stop and placement results](PLACEMENT_RESULTS.md): hybrid
  recoveries remain explicitly separate from autonomous PPO success.
- [Paired BC/PPO protocol](BC_PPO_PROTOCOL.md): teacher transfer, actual saved
  initial-model fingerprints, isolated evaluation RNG, and matched control.
- [Paired learning results](BC_PPO_RESULTS.md): fixed-budget final performance,
  secondary validation-selected checkpoint, training telemetry and limitations.
- [Placement visualization](evidence/pickcube_placement/index.html).
- [BC/PPO visualization](evidence/bc_ppo_paired/index.html).

Use the separate ManiSkill Python 3.11 environment from the
[setup instructions](README.md). The MuJoCo Python environment is separate.
Teacher/checkpoint files under `learning/runs/` are local artifacts, excluded
from Git. Reproducing the training requires the existing BC-256 teacher or
rebuilding it using [the PickCube baseline commands](PICKCUBE_README.md).

```bash
# These commands are run from the repository root with the ManiSkill venv.
# A new output directory is required. Preflight declares settings before training.
.venv/bin/python -m learning.maniskill.bc_initialized_ppo preflight \
  --output learning/runs/bc_ppo_fresh \
  --teacher learning/runs/pickcube_bc/bc_256/best.pt

.venv/bin/python -m learning.maniskill.bc_initialized_ppo train \
  --output learning/runs/bc_ppo_fresh \
  --teacher learning/runs/pickcube_bc/bc_256/best.pt --arm warm \
  > learning/runs/bc_ppo_fresh/warm_training.log 2>&1
.venv/bin/python -m learning.maniskill.bc_initialized_ppo train \
  --output learning/runs/bc_ppo_fresh \
  --teacher learning/runs/pickcube_bc/bc_256/best.pt --arm control \
  > learning/runs/bc_ppo_fresh/control_training.log 2>&1

# Both arms must complete before touching the declared test resets.
.venv/bin/python -m learning.maniskill.bc_ppo_verify \
  --run learning/runs/bc_ppo_fresh \
  --teacher learning/runs/pickcube_bc/bc_256/best.pt
.venv/bin/python -m learning.maniskill.bc_initialized_ppo compare \
  --output learning/runs/bc_ppo_fresh \
  --teacher learning/runs/pickcube_bc/bc_256/best.pt
.venv/bin/python -m learning.maniskill.bc_ppo_verify \
  --run learning/runs/bc_ppo_fresh \
  --teacher learning/runs/pickcube_bc/bc_256/best.pt --complete
.venv/bin/python -m learning.maniskill.bc_ppo_report \
  --run learning/runs/bc_ppo_fresh \
  --output learning/runs/bc_ppo_fresh/report

# Serve the committed report without simulator dependencies.
python3 -m http.server 8775 --bind 127.0.0.1 \
  --directory learning/maniskill/evidence/bc_ppo_paired
```

Open http://localhost:8775 for paired native videos, validation and stochastic
training curves, final test outcomes and optimizer traces. To view the P0 report,
serve `learning/maniskill/evidence/pickcube_placement` on another port.

Evaluation preserves Python, NumPy and CPU Torch RNG state across environment
construction, inference and cleanup. The guard covers the CPU training recipe;
it makes no claim about CUDA RNG or simulator-internal random state in an
unrelated live environment. Telemetry is read-only and counts every stochastic
training transition, excluding validation/test episodes.

The critic-path regression test uses a tiny fake observation/action contract to
check actual policy inputs in rollout, value/bootstrap and likelihood paths.
Real simulator tests additionally check transfer and stochastic updates. CI
validates this implementation and saved evidence; it does not independently
repeat the million-transition experiment or establish learned task success.
