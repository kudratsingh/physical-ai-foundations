# PickCube action-distribution ablation: predeclared protocol

## Question and treatment

Does training with a tanh-transformed Gaussian instead of an externally clipped
Gaussian improve accurate placement and stopping? This changes the action
probability distribution, not just commands at evaluation. Use SB3 2.9.0's
`SquashedDiagGaussianDistribution`, including the Jacobian in log probabilities.
The actor/critic, learned latent log standard deviations and Adam optimizer stay
the same. Initial parameter fingerprints must match a raw-Gaussian constructor.
The deterministic command becomes tanh(latent mean). Tanh can itself saturate;
bounded support does not guarantee small commands or task success.

## Fixed recipe and data split

- Upstream PickCube-v1, raw 42-component state, pd_joint_delta_pos eight actions,
  original normalized-dense reward, 50-step horizon, all success thresholds unchanged.
- Seed 42, four CPU workers, exactly 1,048,576 transitions, rollout 256, batch 256,
  epochs 8, actor/critic256x256, learning rate 0.0003, gamma 0.99, lambda 0.95,
  PPO ratio clip0.2, entropy coefficient 0. No observation/reward normalization.
- Existing original raw final checkpoint is the primary control; verify its actual
  budget/task/spaces and checksum. It was trained earlier on the same Mac; runtime
  comparisons are descriptive. New validation seeds do not affect training weights.
- Validate every 131,072 transitions on100000–100009. Rank by final success, then
  any-step success, then raw return. Preserve training RNG streams during validation.
- Primary outcome: final official success on50 fresh paired resets110000–110049,
  fixed-budget final versus final. Report any-step success, raw return, final 3D
  distance and maximum seven-arm speed. Continue all 50 steps after first success.
- Selected bounded checkpoint is secondary; original raw selection used different
  validation seeds. No test-based checkpoint/budget selection.
- Native videos show the first two declared test seeds, regardless of outcome.
  Save synchronized action/mean/std, goal error and arm speed traces for validation
  diagnosis. Existing tests90000–90049 remain historical, not this new test set.

## Decision and limits

This is one training seed, not an algorithm ranking or proof that clipping caused
past failures. If the bounded final policy reaches at least 5/50 final successes
and improves on the paired raw control, run an independent paired replication at
seeds43 and44 with the same budget and recipe before claiming a robust improvement.
Use fresh test resets 120000–120049 for both replication seeds and validation seeds
121000–121009. If not promising, report the negative result and inspect the stopping
diagnostic before spending more samples. Do not change reward/controller/horizon
mid-experiment. Keep privileged stopping overrides separate from learned success.

Historical normalization evidence retains its original any-first selection order.
The final-first correction applies to future runs and does not rewrite old evidence.

Implementation reference: [SB3 probability distributions](https://stable-baselines3.readthedocs.io/en/master/common/distributions.html).
The installed pinned SB3 2.9.0 implementation is the source used for this run.
