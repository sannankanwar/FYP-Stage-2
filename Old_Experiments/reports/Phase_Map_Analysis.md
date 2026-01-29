# Hyperbolic Phase Map Generation & Reconstruction Analysis

This document details the mechanisms for generating hyperbolic phase maps, the structure of model inputs and outputs, and the process for reconstructing residual phase maps as implemented in the codebase.

## 1. Hyperbolic Phase Map Generation

The core logic for generating phase maps resides in `src/inversion/forward_model.py` and `data/loaders/simulation.py`.

### The Physics Model
The ideal hyperbolic phase profile, designed to focus light at a specific focal length, is governed by the following formula:

$$
\phi(x, y) = \frac{2\pi}{\lambda} \left( \sqrt{x^2 + y^2 + f^2} - f \right)
$$

Where:
*   $x, y$: Spatial coordinates on the lens surface (in micrometers).
*   $f$: Focal length (in micrometers).
*   $\lambda$: Wavelength of the incident light (in micrometers).
*   $\sqrt{x^2 + y^2 + f^2} - f$: The optical path difference required to constructively interfere at the focal point.

### Implementation Details
*   **Coordinate Grid Creation**: In `data/loaders/simulation.py`, a spatial grid is generated based on the center coordinates ($x_c, y_c$) and the window size ($S$).
    ```python
    x_coords = np.linspace(xc - S/2, xc + S/2, N)
    y_coords = np.linspace(yc - S/2, yc + S/2, N)
    ```
*   **Phase Computation**: The `compute_hyperbolic_phase` function in `src/inversion/forward_model.py` implements the formula above using either NumPy or PyTorch operations.
*   **Phase Wrapping**: The raw accumulated phase is wrapped to the range $[-\pi, \pi]$ using the complex exponential method to simulate the physical properties of a metasurface (which can only impart phase shifts modulo $2\pi$).
    ```python
    phi_wrapped = angle(exp(1j * phi_unwrapped))
    ```

## 2. Model Inputs

The neural networks do not consume the raw phase values directly due to the discontinuity at the wrap boundary ($-\pi / \pi$). Instead, the system uses a **2-channel trigonometric representation**.

*   **Format**: A tensor of shape `(Batch, 2, Height, Width)`.
    *   **Channel 0**: $\cos(\phi_{wrapped})$
    *   **Channel 1**: $\sin(\phi_{wrapped})$
*   **Purpose**: This representation is continuous and differentiable everywhere, preventing the model from struggling with the artificial discontinuities introduced by phase wrapping.

## 3. Model Outputs

The models (e.g., FNO-ResNet, standard ResNet) are regressors designed to predict the physical parameters that generated the input phase map.

*   **Format**: A tensor of shape `(Batch, 5)`.
*   **Parameters**:
    1.  **$x_c$**: Center x-coordinate of the observation window.
    2.  **$y_c$**: Center y-coordinate of the observation window.
    3.  **$S$**: Scaling factor / Window size (the physical width of the field of view).
    4.  **$\lambda$ (Wavelength)**: The design wavelength of the lens.
    5.  **$f$ (Focal Length)**: The focal length of the lens.

## 4. Residual Phase Map Reconstruction

The "residual phase map" concept appears in the loss functions (e.g., `PhysicsConsistencyLoss` in `src/training/loss.py`). It serves as a physics-informed consistency check. The model's predicted parameters are used to *reconstruct* a synthetic phase map, which is then compared to the original input.

### Reconstruction Process
1.  **Prediction**: The model outputs prediction parameters $\theta_{pred} = [x_c, y_c, S, \lambda, f]$.
2.  **Dynamic Grid Generation**: A normalized coordinate grid `grid_x, grid_y` (range $[-0.5, 0.5]$) is created on the GPU matching the input resolution.
3.  **Physical Mapping**: The grid is scaled and shifted using the predicted spatial parameters:
    *   $X_{phys} = x_{c, pred} + S_{pred} \cdot grid_x$
    *   $Y_{phys} = y_{c, pred} + S_{pred} \cdot grid_y$
4.  **Forward Simulation**: The `compute_hyperbolic_phase` function acts on this reconstructed physical grid using the predicted $\lambda_{pred}$ and $f_{pred}$.
5.  **Transformation**: The resulting phase is wrapped and converted into the same 2-channel $[\cos, \sin]$ format used for the input.
    *   $\text{Reconstructed Image} = [\cos(\phi_{pred}), \sin(\phi_{pred})]$

### Calculating the Difference (Residual)
The residual is effectively the difference between the **Reconstructed Image** (derived from predictions) and the **Original Input Image** (ground truth phase map).

*   **Loss Calculation**: The `PhysicsConsistencyLoss` computes the Mean Squared Error (MSE) between these two maps:
    $$ L_{physics} = || \text{Input}_{[\cos, \sin]} - \text{Reconstructed}(\theta_{pred})_{[\cos, \sin]} ||^2 $$

This mechanism forces the model to predict parameters that are not just numerically close to the labels (Parameter Loss) but also physically consistent enough to regenerate the visual phase pattern observed in the input.
