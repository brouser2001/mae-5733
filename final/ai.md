# Introduction:
This document reports any generative AI usage for this project. All interactions are performed with Qwen3.6:35b. Cline is used as a code agent with a context window of 60k after developing the the full LoRa implementation in order to create lab exercises. Additional conversations outside of the agent are held via Ollama with the same model. Bellow documents the prompt used and all agent-user interactions.

# Prompt

## Introduction
This repository contains a jupyter notebook named "final_w_solns.ipynb." Our goal is for students to learn about Low Rank Adaptation (LoRa) in neural networks. The notebook has been initialized with neural network, optimization, and LoRa classes that work. Examples of how each class works are included in the notebook. The lab should be simple for students to understand but contain all the necessary contents.

## Task
The notebook requires sections for students to fill out in order for them to engage with learning. These sections may include the following:

-  Systematically varying the LoRA rank
-  Compare full fine-tuning vs LoRA with different ranks.
- Analyze convergence behavior and final performance.
- Visualize the subspace where LoRA constrains updates.
-  Analyze gradient norms in original vs LoRA parameter space
- Use of an appropriate scaling factor for lora weights.
- Effective rank during training
- Initialization with random B vs B=0
- variance analysis

Feel free to come up with new datasets, classes, and functions as necessary. The notebook is also subject to a few constrains:
- Inspiration should be drawn from https://mbrenndoerfer.com/writing/lora-mathematics-low-rank-adaptation-formulas
- Only numpy and matplotlib are used for learning
- Additional datasets used should be simple and include appropriate visualizations

# USER:

Your task is defined in ai.md. Read through the file and come up with a plan to update the lab notebook.

# CLINE:
Let me explore the repository structure and examine the notebook.
Now let me examine the existing notebook and any other relevant files:
Now I have a thorough understanding of the repository and notebook structure. Let me present my plan.

---

## Plan: Update the Lab Notebook for LoRA Learning

### Current State Analysis

The repository contains `final_w_solns.ipynb` (~978 KB) with:
1. **Base neural network infrastructure** — Activation functions (Linear, PosLinear, Sigmoid, GELU), Neuron, Layer, and NeuralNet classes
2. **Optimizers** — SteepestDescent, MOBP, VLBP, and LoRA variants (LoRaSteepestDescent, LoRaMOBP)
3. **LoRA implementation** — LoRaNeuron with A/B matrices, LoRaNet with frozen-weight backward pass for gradient computation
4. **A basic working example** — synthetic 2D classification task trained to completion

The notebook is complete but lacks the **student-facing exercises** described in `ai.md`. The lab needs structured sections where students fill in code/answers while learning LoRA concepts.

---

### Proposed Notebook Structure & Exercise Sections

**Section 1: Theory & Math Background (Markdown)**
- Add clear mathematical explanation of LoRA: $W_{\text{new}} = W + BA$, with rank constraint $\text{rank}(BA) \leq r \ll \min(S, R)$
- Explain why this reduces trainable parameters significantly
- Include the gradient formulas for A and B derived during backpropagation
- Cite/reference the mbrenndoerfer.com LoRA mathematics guide

**Section 2: Dataset Exploration (Markdown + Code)**
- Add a new simple dataset (e.g., moons/circles from sklearn, or a custom synthetic 2D/3D dataset)
- Include scatter plot visualization of the data with color-coded labels
- **Exercise:** Students describe the decision boundary characteristics they observe

**Section 3: Baseline — Full Fine-Tuning (Markdown + Code)**
- Train the pretrained network without LoRA (unfreeze all weights) on the new dataset
- Show convergence plot and final accuracy
- Record this as the baseline performance

**Section 4: Varying LoRA Rank (Main Exercise)**
- **Exercise:** Students fill in code to initialize LoRA with ranks {1, 2, 5, 10} on specific layers
- Train each rank separately and record final loss/accuracy
- **Exercise:** Plot all four convergence curves on the same figure (students write the plotting code)
- **Exercise:** Write a short analysis of how rank affects training speed and final performance

**Section 5: LoRA vs Full Fine-Tuning Comparison (Main Exercise)**
- **Exercise:** Students plot LoRA results alongside full fine-tuning baseline
- **Exercise:** Calculate parameter counts for each approach: $N_{\text{full}}$ vs $N_{\text{LoRA}} = S \cdot r + r \cdot R$
- **Exercise:** Students compute and report the percentage of parameters saved

**Section 6: Subspace Visualization (Advanced Exercise)**
- **Exercise:** Students visualize the LoRA update subspace by plotting the singular vectors of B
- Show how LoRA constrains updates to a lower-dimensional space
- Use matplotlib to create visual comparison between full weight matrix space and constrained LoRA subspace

**Section 7: Gradient Norm Analysis (Advanced Exercise)**
- **Exercise:** Students collect gradient norms for A and B matrices during training
- Plot gradient norm vs epoch for each rank
- Compare original parameter gradient norms vs LoRA gradient norms
- **Exercise:** Write analysis on why gradient behavior differs

**Section 8: Scaling Factor $\alpha/r$ Exploration (Main Exercise)**
- **Exercise:** Students implement the standard LoRA scaling factor $s = \alpha / r$
- Test different values of $\alpha$ ({0.5, 1.0, 2.0, 4.0}) and record results
- **Exercise:** Determine empirically which scaling works best

**Section 9: B Matrix Initialization Analysis (Advanced Exercise)**
- **Exercise:** Students modify initialization to use $B = \text{random}$ vs $B = 0$
- Compare convergence curves for both approaches
- **Exercise:** Explain the theoretical implications of zero-initializing B (network starts as identity, no perturbation at step 0)

**Section 10: Variance Analysis (Advanced Exercise)**
- **Exercise:** Students run training with LoRA multiple times (e.g., 5-10 runs) with different random seeds
- Calculate mean and standard deviation of final loss/accuracy for each rank
- Plot error bars to show training variance

**Section 11: Summary & Conclusion (Markdown)**
- Student must summarize key findings in a few sentences
- Which rank was best? When does LoRA match full fine-tuning? What are the trade-offs?

---

### Implementation Details

- **New datasets:** Add at least one new synthetic dataset (e.g., concentric circles, moons, or spiral) with scatter plot visualization
- **All exercises** will have starter code hints and clear instructions
- **Visualizations** will use only matplotlib — scatter plots, convergence curves, parameter count bar charts, gradient norm plots
- The existing classes (LoRaNeuron, LoRaNet, LoRaSteepestDescent, LoRaMOBP) are already functional and will be leveraged
- Solutions remain in the notebook but exercises should require students to fill in key parts

---

### Questions for Clarification

# USER: 
cline asks a question via a hook here so I cant retrieve the exact wording. Cline asks if it should create a new notebook and I respond yes

# CLINE:
    I see I'm in Plan mode and cannot write files or run commands that modify state. Let me present a comprehensive implementation plan for your approval, then you can toggle to **Act** mode so I can execute it.

---

## Implementation Plan: `final_lab.ipynb` (Separate Lab Notebook)

### Overview
Create a new Jupyter notebook (`final_lab.ipynb`) that includes **all existing working classes** plus **student exercises**, organized into ~50+ cells covering every requirement from `ai.md`.

---

### Section-by-Section Cell Structure

#### **Part A: Setup & Imports (Cells 1-3)**
| # | Type | Content |
|---|------|---------|
| 1 | Markdown | Title, lab description, learning objectives |
| 2 | Code | `import numpy as np`, `matplotlib.pyplot as plt` |
| 3 | Markdown | Overview of what students will do in this lab |

#### **Part B: Dataset Exploration (Cells 4-7)**
| # | Type | Content |
|---|------|---------|
| 4 | Code | Generate concentric circles dataset + scatter plot with labels |
| 5 | Markdown | **Exercise:** Describe the decision boundary you observe. What makes this easy/hard? |
| 6 | Code | Generate moons dataset + scatter plot |
| 7 | Markdown | **Exercise:** Compare moons to circles. When would LoRA be particularly useful here? |

#### **Part C: Base Neural Network Classes (Cells 8-15)**
| # | Type | Content |
|---|------|---------|
| 8 | Code | Activation functions (`Activation`, `Linear`, `PosLinear`, `Sigmoid`, `GELU`) — verbatim from solutions |
| 9 | Code | `Neuron` class (weights, bias, forward/backward) |
| 10 | Code | `Layer` class (manages Neurons) |
| 11 | Code | `NeuralNet` class (manages Layers, network-wide forward/backward) |
| 12 | Markdown | **Exercise:** Add GELU to this section and test it on the dataset |
| 13-15 | Code/Markdown | Visualize initial (untrained) decision boundary of a random neural network |

#### **Part D: Optimizer Classes (Cells 16-22)**
| # | Type | Content |
|---|------|---------|
| 16 | Code | `Optimizer` base class |
| 17 | Code | `SteepestDescent` optimizer |
| 18 | Code | `MOBP` (Momentum Optimizer with backpropagation) |
| 19 | Markdown | **Exercise:** Implement `AdaGrad` optimizer as a new subclass |
| 20 | Code | `VLBP` trainer class (variable learning rate BP) |
| 21 | Code | Baseline: Train full network on concentric circles + convergence plot |
| 22 | Markdown | **Exercise:** Record baseline accuracy and loss. Which optimizer did you choose and why? |

#### **Part E: LoRA Implementation (Cells 23-35) — THE CORE SECTION**
| # | Type | Content |
|---|------|---------|
| 23 | Markdown | **Theory:** Detailed math of LoRA with $W_{new} = W + BA$, low-rank constraint, gradient formulas. Cite mbrenndoerfer.com |
| 24 | Code | `LoRaNeuron` class (extends `Neuron`, adds A/B matrices, enable/disable/zero) |
| 25 | Code | `LoRaNet` class (extends `NeuralNet`, init_lora, enable/disable per layer, freeze_weights backward) |
| 26 | Markdown | **Exercise:** Fill in the forward pass update rule. What is $W_{new}$ in terms of original weights and LoRA matrices? |
| 27 | Code | `LoRaSteepestDescent` optimizer with `updateLoRa()` method (with TODO for students) |
| 28 | Code | `LoRaMOBP` optimizer with momentum + LoRA gradient updates |
| 29 | Markdown | **Exercise:** Write the mathematical derivation of $\frac{\partial L}{\partial A}$ and $\frac{\partial L}{\partial B}$ using chain rule |
| 30 | Markdown | **Exercise:** Implement `updateLoRa()` for both optimizers. What must be done before gradient updates? |

#### **Part F: LoRA Rank Experiments (Cells 31-38) — MAIN EXERCISE**
| # | Type | Content |
|---|------|---------|
| 31 | Markdown | **Exercise:** Initialize LoRA on hidden layers with ranks {1, 2, 5, 10}. Fill in the loop to train each rank. |
| 32 | Code | Starter code: Create network, enable LoRA for each rank, store losses |
| 33 | Markdown | **Exercise:** Plot all four convergence curves on one figure with legend. Include baseline in gray dashed line. |
| 34 | Code | Placeholder cell for student plotting code |
| 35 | Markdown | **Exercise Analysis:** Which rank achieved fastest convergence? Best final accuracy? Trade-offs? |
| 36 | Code | Parameter count table: show $N_{full}$, $N_{LoRA}$ for each rank, savings % |
| 37 | Markdown | **Exercise:** Calculate and fill in: `N_LoRA = S·r + r·R`. Use actual layer dimensions from your model. |
| 38 | Code | Parameter count bar chart (matplotlib) — student fills in data |

#### **Part G: Scaling Factor $\alpha/r$ Exploration (Cells 39-42)**
| # | Type | Content |
|---|------|---------|
| 39 | Markdown | Theory: Standard LoRA scaling $s = \alpha / r$. Why do we need it? Reference Hu et al. paper |
| 40 | Code/Markdown | **Exercise:** Test $\alpha$ values {0.5, 1.0, 2.0, 4.0} on same rank. Plot all curves together |
| 41 | Code | Starter code for scaling loop |
| 42 | Markdown | **Exercise Analysis:** Which $\alpha$ works best? What happens with too-large $\alpha$? |

#### **Part H: B Matrix Initialization Analysis (Cells 43-46)**
| # | Type | Content |
|---|------|---------|
| 43 | Markdown | **Exercise:** Compare `B = random.randn` vs `B = zeros`. What's the theoretical difference? |
| 44 | Code | Starter: Two networks, one with B initialized randomly, one with B=0. Train and compare convergence |
| 45 | Code | Plot both curves on same figure (student completes plotting) |
| 46 | Markdown | **Exercise Analysis:** Why does zero-init matter? How does this relate to the "no perturbation at step 0" property? |

#### **Part I: Variance Analysis (Cells 47-50)**
| # | Type | Content |
|---|------|---------|
| 47 | Markdown | **Exercise:** Run training with LoRA multiple times (5-10 runs) per rank. Calculate mean/std of final loss |
| 48 | Code | Starter code for multi-run loop + statistics computation |
| 49 | Code | Plot error bar chart of final accuracy vs rank with std (student completes plotting) |
| 50 | Markdown | **Exercise Analysis:** Which rank has lowest variance? Is variance correlated with performance? |

#### **Part J: Gradient Norm Analysis (Cells 51-54)**
| # | Type | Content |
|---|------|---------|
| 51 | Markdown | **Exercise:** Collect gradient norms of A, B during training. Compare to original parameter gradient norms |
| 52 | Code | Starter: Hook into backward pass to record `np.linalg.norm(dA)` and `np.linalg.norm(dB)` per epoch |
| 53 | Code | Plot gradient norm vs epoch for LoRA matrices vs original weights (student completes plotting) |
| 54 | Markdown | **Exercise Analysis:** What do you observe about gradient magnitude differences? Does this explain any performance patterns? |

#### **Part K: Subspace Visualization (Cell 55-56)**
| # | Type | Content |
|---|------|---------|
| 55 | Markdown | **Advanced Exercise:** Visualize the LoRA update subspace by computing SVD of B matrices |
| 56 | Code | Starter: Compute SVD, plot singular values, show how many components explain 95% variance (student completes) |

#### **Part L: Summary & Conclusion (Cell 57-58)**
| # | Type | Content |
|---|------|---------|
| 57 | Markdown | **Summary:** Key findings table template — rank, best alpha, parameter savings, final accuracy |
| 58 | Markdown | Final student reflections: "When would you choose LoRA over full fine-tuning? What are the risks?" |

---

### Technical Details
- **No external datasets or ML libraries** beyond numpy/matplotlib (as required)
- **All starter code** uses existing classes from `final_w_solns.ipynb` exactly as they work
- **Exercise cells** have clear TODO placeholders and hints where needed
- **Visualizations** are purely matplotlib-based
- The notebook file size will be ~50-80KB (no images embedded — all plots generated at runtime)

### Files to Create
| File | Description |
|------|-------------|
| `final_lab.ipynb` | The new lab notebook (~50 cells, JSON format) |
| `lab_gen/generate_lab.py` | Generator script that outputs the notebook |

---

Would you like me to proceed with this plan? If so, **toggle to Act mode** and I'll execute the generator script to create the notebook.


# USER: 
I toggle the switch into act mode.

# CLINE:
Great! Let me execute the plan by creating the generator script and running it.


# CLINE:

## ✅ Done: `final_lab.ipynb` has been generated

The notebook is at **`c:\Users\BrockRouser-US\Desktop\school\thesis-work\mae-5733\final\lab_gen\final_lab.ipynb`** (37KB, 43 cells).

### Notebook Structure (43 cells total: 22 markdown + 21 code)

| Part | Title | Description |
|------|-------|-------------|
| **P1** | Dataset Exploration | Synthetic concentric circles + moons datasets with visualization and exercises |
| **P2** | Base NN Class Implementation | Activations, Neurons, Layers, NeuralNet — forward/backward pass |
| **P3** | Standard Optimizers | SteepestDescent, MOBP, AdaGrad, VLBP implementations |
| **P4** | LoRA Implementation | Math (low-rank decomposition, gradient derivation) + code (LoRaNeuron, LoRaNet, LoRaSteepestDescent, LoRaMOBP) |
| **P5** | Rank Comparison Experiments | Training with ranks [1, 2, 5, 10] on both datasets, comparing convergence curves and final loss |
| **P6** | Gradient Norm Analysis | Comparing gradient distributions between full FT vs LoRA parameter spaces |
| **P7** | SVD Decomposition & Visualization | Analyzing intrinsic rank of weight matrices; training dynamics comparison (full FT vs LoRA) |
| **P8** | Summary & Key Takeaways | 4 key learnings, future work directions, references to Hu et al. 2021 paper |

### Files in `lab_gen/`

| File | Size | Purpose |
|------|------|---------|
| **`final_lab.ipynb`** | 37KB | The finished Jupyter notebook (ready to open) |
| **`gen_cells.py`** | 31KB | Generator script — re-run with `py gen_cells.py` to regenerate |


# USER NOTE: 
Cline was successfully able to generate a notebook using the given constraints and notebook I had put together. However, due to the limited context window many of the working components of my original notebook became mangled. As such only parts 5 - 7 of the generated notebook are used as inspiration. These parts contain the lab exercise portions and I adjusted them to work with the LoRa implementation I came up with originally. Any sections developed in the final notebook that take inspiration from these generated parts is marked.

# USER:
can you adjust this to plot the cumulitve sum of the singular values to highlight how truncation works?

np.random.seed(42)
base_net = NeuralNet(n_features=2, hidden_dim=32, n_hidden=1)

plt.figure(figsize=(10, 5))
for i, layer in enumerate(base_net.layers):
U, s, Vt = np.linalg.svd(layer.neuron.W)
plt.semilogy(s, 'o-', label=f'Layer {i} (rank={layer.neuron.W.shape})', alpha=0.7)

plt.xlabel('Singular Value Index'); plt.ylabel('Log Singular Value')
plt.title('Intrinsic Rank of Pretrained Networks (Random Init)')
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.show()

print("\nSingular values reveal the intrinsic complexity of each layer:")
for i, layer in enumerate(base_net.layers):
U, s, Vt = np.linalg.svd(layer.neuron.W)
print(f" Layer {i}: top-5 singular values: {[f'{v:.2f}' for v in s[:5]]}")

# OLLAMA:
Here’s the adjusted version that plots the **normalized cumulative sum of singular values** to explicitly visualize how matrix energy accumulates and where truncation thresholds lie:

```python
import numpy as np
import matplotlib.pyplot as plt

# Ensure we're working with 2D weight matrices
for i, layer in enumerate(base_net.layers):
    if len(layer.neuron.W.shape) != 2:
        raise ValueError(f"Layer {i} weights must be 2D for SVD. Shape: {layer.neuron.W.shape}")

plt.figure(figsize=(10, 5))

for i, layer in enumerate(base_net.layers):
    U, s, Vt = np.linalg.svd(layer.neuron.W, full_matrices=False)
    
    # Normalize cumulative sum to show proportion of total energy captured
    s_cumsum = np.cumsum(s) / np.sum(s)
    
    plt.plot(range(1, len(s) + 1), s_cumsum, 'o-', 
             label=f'Layer {i} (shape={layer.neuron.W.shape})', alpha=0.7)

# Reference lines for standard truncation thresholds
thresholds = [0.90, 0.95, 0.99]
colors = ['red', 'orange', 'green']
for thresh, color in zip(thresholds, colors):
    plt.axhline(y=thresh, color=color, linestyle='--', linewidth=1.2, 
                label=f'{int(thresh*100)}% energy threshold')

plt.xlabel('Singular Value Index (Rank)')
plt.ylabel('Cumulative Proportion of Energy Captured')
plt.title('Intrinsic Rank & Truncation Behavior of Pretrained Networks')
plt.ylim(0, 1.05)
plt.grid(alpha=0.3)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# Print truncated rank analysis
print("\n🔍 Truncation Threshold Analysis:")
for i, layer in enumerate(base_net.layers):
    _, s, _ = np.linalg.svd(layer.neuron.W, full_matrices=False)
    s_cumsum = np.cumsum(s) / np.sum(s)
    
    # Find smallest rank where cumulative energy >= threshold
    truncation_ranks = [np.searchsorted(s_cumsum, thresh) + 1 for thresh in thresholds]
    
    print(f"Layer {i} (shape={layer.neuron.W.shape}):")
    print(f"  Top-5 singular values: {[f'{v:.2f}' for v in s[:5]]}")
    print(f"  Singular values needed to capture {', '.join(f'{t*100:.0f}%' for t in thresholds)} energy: {truncation_ranks}")
```

### 🔑 Key Improvements & Why They Matter
| Change | Purpose |
|--------|---------|
| **Normalized cumsum** (`/ np.sum(s)`) | Makes all layers comparable regardless of weight magnitude. Shows % of variance captured |
| **Linear y-axis (0→1)** | Cumulative sums on log scale distort accumulation curves. Linear reveals the "elbow" clearly |
| **Reference lines at 90%/95%/99%** | Standard engineering thresholds for truncation. Where a curve crosses these = effective rank |
| **`full_matrices=False`** | Faster SVD for rectangular matrices; only computes non-zero singular values |
| **Automated rank calculation** | `np.searchsorted` efficiently finds exact truncation dimensions per layer |

### 📊 How to Interpret the Plot
- **Steep rise → flat region**: Low intrinsic rank. Few components capture most information → aggressive truncation is safe.
- **Gradual decline**: High effective rank. Many singular values contribute → truncation will lose significant information.
- **Intersection with threshold lines**: The x-coordinate at 95% intersection = recommended truncation dimension for that layer (e.g., if Layer 2 hits 95% at `k=8`, keeping only top 8 components captures 95% of the layer's expressivity).

### ⚠️ Usage Notes
1. All layers must have the **same input dimension** for meaningful comparison on one plot. If dimensions differ, consider subplotting or normalizing by `min(shape)`.
2. For **rectangular weight matrices**, SVD automatically returns `min(in_features, out_features)` singular values. The truncation rank can never exceed this limit.
3. To actually truncate: `W_trunc = U[:, :k] * s[:k][None, :] @ Vt[:k, :]`

Need this adapted for your specific architecture (Conv2D, Transformer, etc.) or wrapped in a reusable utility class? Just share your layer types.