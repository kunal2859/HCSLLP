# Fine-Tuning Analysis: DevOps AI Assistant

## Dataset Choice
The dataset consists of **200 high-quality instruction-response pairs** curated specifically for DevOps engineering. It covers areas such as Docker configuration, Kubernetes orchestration, CI/CD pipeline optimization, and infrastructure-as-code (Terraform/Ansible). I chose this domain because DevOps often involves deterministic but complex syntax (like YAML or Shell) where base models frequently hallucinate specific flags or version-dependent configurations. By fine-tuning on this target domain, the model learns the "shorthand" and best practices specific to modern platform engineering.

## LoRA Decisions (Rank & Alpha)
I selected a **LoRA Rank (r) of 8** and an **Alpha of 16**. 
- **Rank 8**: Provides a balance between parameter efficiency and model capacity. For a 3.8B parameter model like Phi-3-mini, a rank of 8 adds roughly 0.1% more parameters, which is sufficient to capture domain-specific terminology without overfitting.
- **Alpha 16**: Using a 2:1 ratio for Alpha to Rank is a standard heuristic that provides a stable starting point for the learning rate scaling. This ensures that the adapter weights have a meaningful impact on the base model's activations.
- **Target Modules**: I targeted `all-linear` layers to ensure that the adaptation occurs across the entire transformer block, not just the attention layers.

## Evaluation Results & Limitations
The fine-tuned model achieved a **measurable improvement in ROUGE-L scores** on the held-out 20 samples. 
- **Base Model**: Often gave generic, long-winded answers that missed specific DevOps technicalities.
- **Fine-Tuned Model**: Showed higher "Exact Match" qualities for command-line flags and a more concise, technical tone.

**Limitations**: 
- **Context Window**: Despite the 4k capacity of Phi-3, the LoRA adaptation was done with a `max_length` of 128 to save memory on 16GB RAM devices. This means the model might struggle with very long logs.
- **Dataset Size**: While 200 samples meet the baseline, the model still exhibits occasional "catastrophic forgetting" of general knowledge when pressed on non-DevOps topics.
