# PickCube critic investigation

[Results](CRITIC_RESULTS.md) · [Protocol](CRITIC_PROTOCOL.md) ·
[Native visual dashboard](evidence/critic_bound/index.html)

Use the separate ManiSkill Python 3.11 environment from [setup](README.md).
Full checkpoints and the expert dataset remain ignored local artifacts.

The historical results were produced by source commit `4f48931`. Use that commit
in an isolated worktree when reproducing or running strict source guards on the
committed historical experiment. The current commands below declare a **new**
worker-seeded condition: Python/NumPy/Torch seeds per worker plus ManiSkill
`enhanced_determinism=True`. It uses new validation seeds 200000–200009 and test
seeds 210000–210049. No policy has been trained or evaluated on those partitions
in this investigation. Current source guards intentionally reject the historical
protocol after the repair; its archived snapshots and verification remain intact.


```bash
.venv/bin/python -m learning.maniskill.critic_experiment declare \
  --output learning/runs/critic_fresh \
  --teacher learning/runs/pickcube_bc/bc_256/best.pt
.venv/bin/python -m learning.maniskill.critic_experiment preflight \
  --output learning/runs/critic_fresh \
  --teacher learning/runs/pickcube_bc/bc_256/best.pt
# Train each arm freshly; these may run sequentially or concurrently.
.venv/bin/python -m learning.maniskill.critic_experiment train \
  --output learning/runs/critic_fresh --arm raw \
  --teacher learning/runs/pickcube_bc/bc_256/best.pt
.venv/bin/python -m learning.maniskill.critic_experiment train \
  --output learning/runs/critic_fresh --arm bounded \
  --teacher learning/runs/pickcube_bc/bc_256/best.pt
# Require actual complete checkpoints before using the fresh final test seeds.
.venv/bin/python -m learning.maniskill.critic_verify \
  --run learning/runs/critic_fresh --teacher learning/runs/pickcube_bc/bc_256/best.pt
.venv/bin/python -m learning.maniskill.critic_experiment compare \
  --output learning/runs/critic_fresh --teacher learning/runs/pickcube_bc/bc_256/best.pt
.venv/bin/python -m learning.maniskill.critic_verify \
  --run learning/runs/critic_fresh --teacher learning/runs/pickcube_bc/bc_256/best.pt --complete

# The offline historical audit requires the previous BC/PPO run and expert data.
.venv/bin/python -m learning.maniskill.critic_audit \
  --run learning/runs/bc_ppo_paired42 \
  --dataset learning/.assets/pickcube_cpu_full/trajectory.state.pd_joint_delta_pos.physx_cpu.h5 \
  --split-protocol learning/maniskill/evidence/stage1_pickcube/protocol.json \
  --output learning/runs/critic_audit/audit.json
.venv/bin/python -m learning.maniskill.critic_report \
  --run learning/runs/critic_fresh --audit learning/runs/critic_audit/audit.json \
  --output learning/runs/critic_fresh/report

# The committed dashboard needs no simulator or training dependencies.
python3 -m http.server 8777 --bind 127.0.0.1 \
  --directory learning/maniskill/evidence/critic_bound
```

Open http://localhost:8777 for value/return plots, measured gradient and Adam
update plots, fixed-budget versus selected outcomes, and native videos.
On a restricted macOS session, set `MPLCONFIGDIR` to a writable directory with a
prebuilt font cache before rendering plots. This does not affect training.
