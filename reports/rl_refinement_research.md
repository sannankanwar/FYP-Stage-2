# Research Report: RL & Non-Differentiable Refinement for Metalens Inversion

**Goal**: Refine `Experiment9` output parameters to minimize residual phase, given a **Non-Differentiable (Black-Box) Forward Model**.
**Constraints**: < 30GB VRAM, Single GPU.

---

### 1) Concept Card: Black-Box Refinement

*   **Core Mechanism**: Since we cannot backpropagate through the Forward Model, we must treat the simulator as an "Environment" or "Oracle". The Refiner Agent observes the current state (parameters + residual) and takes an action (update parameters) to maximize the Reward (negative reconstruction error).
*   **Baselines**:
    *   **Random Search**: Simple perturbation.
    *   **Nelder-Mead**: Standard derivative-free optimization.
    *   **CMA-ES**: Evolutionary strategy (Covariance Matrix Adaptation) - Gold standard for continuous black-box problems.
*   **Approaches**:
    1.  **Evolutionary / Genetic Algorithms (GA)**: Maintain a population of solutions. Evolve them by selecting the best performers (lowest residual) and mutating them.
    2.  **Reinforcement Learning (RL)**: Train an agent (Policy Network) to predict the *optimal update step* given the current parameters and residual.
        *   **State**: Current Parameters, Residual Image.
        *   **Action**: $\Delta \theta$ (Change in parameters).
        *   **Reward**: Reduction in MSE of residual.
    3.  **Bayesian Optimization**: Build a probabilistic surrogate model of the Forward Model to guide the search (usually expensive for high dimensions/pixels, but good for few parameters).

---

### 2) Experiment Plans

#### Plan A: Evolutionary Algorithm (CMA-ES) [Recommended First Step]
**Hypothesis**: Evolutionary strategies are generally more sample-efficient and stable than RL for direct parameter optimization where no "sequential decision making" is strictly required (i.e., we just want the best parameters, not necessarily a trajectory).
*   **Setup**:
    *   **Tool**: `cma` (Python library) or `deap`.
    *   **Initialization**: Start center at `Experiment9` prediction. Sigma = 0.05.
    *   **Population**: 32-64 candidates per generation (Parallelizable).
*   **Steps**:
    1.  Generate 64 candidate parameter sets.
    2.  Run Forward Model (Black Box) on all 64.
    3.  Compute Loss (MSE of Residual).
    4.  Update CMA distribution.
    5.  Repeat for 50-100 generations.
*   **Pros**: Extremely robust, no training required, handles non-differentiable landscapes well.
*   **Cons**: Slower inference (requires many forward passes).

#### Plan B: Reinforcement Learning (PPO/DDPG)
**Hypothesis**: An RL agent can learn a "search policy" that is much faster than CMA-ES. Instead of evolving a population, the agent looks at the residual and says "Increase Focal Length by 0.1".
*   **Setup**:
    *   **Environment**: Custom generic Gym/Farama environment wrapping your Forward Model.
    *   **Observation**: `[Current_Params (5), Residual_Map_Downsampled (32x32)]`.
    *   **Action**: Continuous vector `Delta_Params (5)`.
    *   **Algorithm**: **PPO** (Proximal Policy Optimization) or **TD3** (Twin Delayed DDPG) via `stable-baselines3`.
*   **Training Loop**:
    1.  Reset Env: Sample random metalens parameters $\theta_{gt}$, simulate image $I$.
    2.  Agent guesses $\theta_0$ (or use Expr9 predictor).
    3.  Loop 10 steps:
        *   Agent sees residual -> outputs $\Delta \theta$.
        *   New $\theta = \theta + \Delta \theta$.
        *   Reward = $MSE_{prev} - MSE_{new}$.
*   **Pros**: Once trained, inference is instant (1 forward pass of Policy Net).
*   **Cons**: Hard to train. RL is unstable. Requires defining a good reward function.

#### Plan C: "Unsupervised" Learned Refiner (Simulated Annealing inspired)
**Hypothesis**: A specialized Neural Network can be trained to simply "clean up" the parameters using a self-paced curriculum, essentially distilling the CMA-ES process into a network.
*   **Setup**:
    *   Generate a dataset of (Perturbed Params, GT Params).
    *   Train a network `Refiner(Perturbed, Residual) -> GT`.
    *   This effectively learns the inverse of the local error surface.
*   **Pros**: Supervised training (stable).
*   **Cons**: Requires generating a massive dataset from the slow black-box forward model beforehand.

---

### 3) Uncertainties & Resolution

*   **Forward Model Speed**: If the Forward Model takes >1 second, RL training (millions of steps) will be infeasible.
    *   *Resolution*: Use a "Surrogate Model" (Fast Neural Proxy) for RL training, then fine-tune on the real Black Box.
*   **State Representation**: Is the Residual Map necessary? Or just the scalar error?
    *   *Resolution*: The Residual Map contains spatial information (e.g., "center is focused, edges are blurry") which guides *which* parameter to tune.

### Recommendation
1.  **Immediate**: Implement **CMA-ES (Plan A)**. It requires no training, just scripting. It will tell you the "upper bound" of performance.
2.  **Secondary**: If CMA-ES is too slow for your final application, use the data generated by CMA-ES runs to train an **RL Agent (Plan B)** or **Refiner (Plan C)** (Behavior Cloning).
