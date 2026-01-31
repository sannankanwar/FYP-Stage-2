# Research Report: Refinement Strategies for Metalens Inversion

**Goal**: Refine the output parameters of "Experiment 9" to minimize the residual between the input phase map and the forward-modeled phase map.
**Constraint**: < 30GB VRAM, Single GPU.
**Context**: You have a differentiable Forward Model.

---

### 1) Concept Card: Test-Time Optimization (TTO) & Learned Refinement

*   **Core Mechanism**: Instead of trusting the one-shot prediction of the neural network $f(x) \to \theta$, we treat the prediction as an *initialization* $\theta_0$. We then refine $\theta$ to minimize the physics-consistency loss: $\mathcal{L} = \| x - \text{Forward}(\theta) \|^2$.
*   **Baselines**:
    *   **Direct Regression**: Standard ResNet/UNet (what you have).
    *   **Random Search**: Perturbing $\theta$ randomly (inefficient).
*   **Approaches**:
    1.  **Gradient-Based TTO**: Use backpropagation through the fixed Forward Model to update $\theta$ directly using Adam/SGD.
    2.  **Learned Optimizer**: Train a secondary network (RNN/U-Net) that learns *how* to update $\theta$ based on the gradient or residual.
    3.  **Evolutionary Algorithms**: Evolve a population of $\theta$ values to minimize loss without gradients.
*   **Failure Modes**:
    *   **Local Minima**: TTO might get stuck if the landscape is non-convex.
    *   **Run-time**: TTO requires many forward passes per sample (slow).
    *   **Gradient Explosion**: Backpropagating through complex physics simulators can be unstable.

---

### 2) Experiment Plans

#### Plan A: Gradient-Based Test-Time Optimization (Recommended)
**Hypothesis**: Since the Forward Model is differentiable, we can directly optimize the parameters for each specific test sample to minimize the residual, achieving near-perfect consistency.
*   **Setup**:
    *   Take `Experiment9` prediction $\theta_{pred}$.
    *   Freeze the model weights. Set $\theta = \theta_{pred}$ as a leaf variable with `requires_grad=True`.
    *   Define Loss: `MSE(InputPhase, ForwardModel(theta))`.
    *   Optimizer: `Adam(params=[theta], lr=1e-2)`.
*   **Steps**:
    1.  For each test sample attempt ~100-500 iterations of gradient descent.
    2.  Track the reduction in Residual RMSE.
*   **Pros**: No new training required. Mathematically precise.
*   **Cons**: Slow at inference time (seconds per sample).

#### Plan B: Genetic Algorithm (GA) / CMA-ES
**Hypothesis**: If the loss landscape has many local minima or high-frequency noise that confuses gradients, a population-based search will find better global solutions.
*   **Setup**:
    *   Use a library like `deap` or `pygad`.
    *   **Population**: Generate 50 variants of $\theta$ around the `Experiment9` prediction (Gaussian noise).
    *   **Fitness**: Negative RMSE of the residual.
*   **Steps**:
    1.  Run Forward Model on batch of 50 candidates (efficient on GPU).
    2.  Select top k, crossover, mutate.
    3.  Repeat for 10-20 generations.
*   **Pros**: Robust to bad gradients. Easy to parallelize on GPU.
*   **Cons**: Approximate. Can be slower than gradient descent if population is large.

#### Plan C: Learned Optimizer (The "Refiner" V2)
**Hypothesis**: An "Optimizer Model" can learn to jump straight to the solution or take efficient steps, speeding up TTO significantly.
*   **Setup**:
    *   **Architecture**: A holistic "Update Network" (e.g., small ResNet or LSTM) that takes `[CurrentParams, Gradient_of_Loss]` as input and outputs `Delta_Params`.
    *   **Training**: Train this network on the *trajectory* of optimization.
*   **Steps**:
    1.  Generate a dataset of (Initial Params, Gradient at Initial, Optimal Step).
    2.  Train network to predict the optimal step.
*   **Pros**: Extremely fast inference (1-3 steps vs 500 TTO steps).
*   **Cons**: Complex to implement and train.

---

### 3) Paper & Code References

*   **Deep Image Prior (Ulyanov et al.)**: Foundational paper on optimizing network/inputs at test time.
*   **CMA-ES (Covariance Matrix Adaptation Evolution Strategy)**: The gold standard for continuous parameter optimization without gradients.
*   **Learning to Learn via Gradient Descent (Andrychowicz et al.)**: The classic "Learned Optimizer" paper.

### Recommendation
**Start with Plan A (Gradient TTO)**. It requires **zero new training** and will tell you the *theoretical limit* of how good your results can get. If TTO achieves near-zero residual, you know the perfect parameters exist and are reachable. If TTO fails, your Forward Model or Parameter Space might be deficient.
