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