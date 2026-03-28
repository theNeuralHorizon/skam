# Reinforcement Learning for Ensemble Weight Optimization — Feasibility Analysis

## Problem Formulation

The goal is to dynamically adjust ensemble weights (e.g., IF weight, LSTM weight,
decision threshold) in response to the current system state, rather than using
static weights like the current 0.4/0.6 split in `CombinedIFLSTMEnsemble`.

### State Space

- **Current feature values**: 5-dimensional vector (request_rate, error_rate,
  p99_latency, cpu_usage, memory_usage).
- **Recent history**: sliding window of the last N feature vectors (e.g., N=10),
  capturing short-term trends. Flattened, this is 50 dimensions.
- **Current ensemble sub-scores**: individual anomaly scores from each component
  (IF score, LSTM proxy score, XGBoost score, etc.). Adds 2-4 dimensions.
- **Total state dimensionality**: ~55-60 continuous features.

### Action Space

- **Continuous**: 3-4 values in [0, 1]:
  - Weight for Isolation Forest component
  - Weight for LSTM/temporal component
  - Weight for XGBoost component (if using the new ensemble)
  - Decision threshold adjustment (+/- 0.1 around base 0.7)
- Actions must satisfy: weights sum to 1.0 (simplex constraint).

### Reward Function

```
R(t) = +1.0 * (true_positive)       # correctly detected anomaly
     + +0.1 * (true_negative)       # correctly passed normal sample
     + -0.5 * (false_positive)      # false alarm penalty
     + -2.0 * (false_negative)      # missed anomaly (heavily penalized)
     + -0.01 * |w(t) - w(t-1)|      # small penalty for weight instability
```

The asymmetric penalties reflect operational priorities: missing a real anomaly is
worse than a false alarm, but false alarms degrade operator trust.

An alternative reward based on continuous metrics:

```
R(t) = F1(window) - 0.3 * FPR(window)
```

where F1 and FPR are computed over a rolling window of recent decisions.

## Algorithm Selection

**PPO (Proximal Policy Optimization)** is the most suitable algorithm because:
- Handles continuous action spaces natively
- Stable training with clipped objective (important for small datasets)
- Works well with relatively few environment interactions
- Widely supported (stable-baselines3, RLlib)

**Alternative**: DQN with discretized weights (e.g., weights in {0.0, 0.1, ..., 1.0})
would simplify training but loses fine-grained control.

## Dataset Size Assessment

**Current dataset: 125 scenarios from RCAEval Online Boutique.**

This is the critical constraint. Each scenario yields ~280 time steps (70 minutes
at 15-second intervals). Total available transitions:

- 125 scenarios x 280 steps = ~35,000 transitions

RL requirements for convergence (rough estimates):
- **Tabular Q-learning**: ~100K transitions for simple MDPs
- **PPO with neural network**: ~500K-1M transitions for moderate state spaces
- **Our state space (55-60 dims)**: likely needs 1-2M transitions minimum

**Verdict: We have ~35K transitions, need ~1M. The dataset is 30x too small.**

### Possible mitigations

1. **Synthetic data augmentation**: Add Gaussian noise to existing scenarios to
   generate more transitions. Risk: may teach the RL agent to respond to noise
   rather than real anomaly patterns.
2. **Offline RL (Conservative Q-Learning)**: Train from a fixed dataset without
   exploration. More sample-efficient, but still likely insufficient at 35K.
3. **Sim-to-real**: Build a simple simulator of the microservice metrics. High
   engineering effort, and fidelity is questionable.

## Estimated Training Time

- **Environment step**: ~0.1ms (feature lookup + ensemble scoring)
- **1M steps with PPO**: ~100 seconds of environment time + ~10 minutes of
  gradient updates = ~15 minutes total on a single GPU
- **Hyperparameter search** (50 configs): ~12 hours
- **Wall-clock to production**: 2-3 days of engineering + tuning

Training time itself is not the bottleneck; data scarcity is.

## Honest Assessment

### Will RL actually help?

**Almost certainly not, for this use case. Here is why:**

1. **Static weights work well enough.** The current 0.4/0.6 split in
   `CombinedIFLSTMEnsemble` achieves competitive AUC-ROC. A grid search over
   static weights (which takes minutes) would capture most of the gains that
   dynamic weighting could provide.

2. **The problem is not sequential.** Each anomaly detection decision is largely
   independent of previous decisions. The ensemble does not need to "plan ahead"
   or consider long-term consequences — this is where RL excels, and it is not
   needed here.

3. **Insufficient data for generalization.** With only 125 scenarios and 5 fault
   types, an RL agent would overfit to the training distribution. It would learn
   scenario-specific weight adjustments rather than general principles.

4. **Simpler alternatives exist.** Bayesian optimization of static weights,
   or even a simple lookup table mapping fault-type signatures to pre-tuned
   weights, would achieve similar or better results with far less complexity.

5. **Operational risk.** An RL agent that adjusts weights at inference time
   introduces non-determinism and makes debugging harder. If the agent enters
   a bad state, it could suppress real anomaly detection.

### When RL *would* make sense

- If we had 10,000+ diverse scenarios (not 125)
- If the system dynamics changed significantly over time (concept drift)
- If there were sequential decision-making involved (e.g., "investigate this
  anomaly vs. wait for more evidence" with a cost model)
- If we had a high-fidelity simulator to train in

### Recommendation

**Do not pursue RL for ensemble weight optimization.** Instead:

1. Run a grid search over weight combinations (5 minutes of compute)
2. Optionally use Bayesian optimization (scikit-optimize) for the 3-4 weight
   parameters — this is far more sample-efficient than RL
3. If dynamic weighting is truly desired, consider a simple rule-based system:
   e.g., increase LSTM weight when temporal variance is high, increase IF weight
   when point anomalies are sharp

The engineering effort for RL (2-3 days minimum) would be better spent on
improving the feature engineering or collecting more training data.
