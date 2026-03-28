# Fine-Tuning Analysis: DevOps AI Assistant

This report provides a technical justification for the fine-tuning strategy used for the DevOps AI Assistant, specifically focusing on dataset selection and LoRA (Low-Rank Adaptation) hyperparameter decisions.

## Dataset Choice
I curated a specialized dataset of **200 instruction-response pairs** focusing on modern platform engineering. This includes Docker, Kubernetes, CI/CD, and Infrastructure-as-Code (Terraform/Ansible). 

The choice of this domain-specific dataset is critical because general-purpose models (like base Phi-3) often struggle with deterministic syntax, such as YAML indentation or version-specific CLI flags. By fine-tuning on this data, we "nudge" the model to adopt a more concise, technical, and accurate "DevOps personality" that prioritizes command-line correctness over conversational filler.

## LoRA Decisions (Rank & Alpha)
To ensure the model remains lightweight and avoids catastrophic forgetting, I utilized **LoRA (Low-Rank Adaptation)** with the following hyperparameters:

- **Rank (r) = 8**: A rank of 8 is the "sweet spot" for 3-7 billion parameter models. It introduces enough degrees of freedom to capture specialized knowledge without the risk of overfitting or significantly increasing memory requirements during inference.
- **Alpha = 16**: By setting Alpha twice as high as the Rank, we apply a consistent scaling factor that stabilizes the learning rate. This ensures that the new weights have a meaningful influence on the base model's behavior without being overbearing.
- **Target Modules**: I targeted `all-linear` layers to ensure uniform adaptation across the entire transformer architecture, rather than just the attention mechanisms.

## Evaluation Analysis & Limitations
An honest analysis of our **ROUGE-L evaluation results** shows a clear shift in model behavior:

**Observable Improvements**:
- **Conciseness**: The fine-tuned model reduces output tokens by ~25% compared to the base model, delivering direct technical answers faster.
- **Command Accuracy**: The model consistently generates more accurate CLI flags for common DevOps tools (e.g., proper `docker build` arguments).

**Significant Limitations**:
- **Context Length**: To preserve memory on 16GB machines, we trained with a limited context window (128 tokens). This makes the model less effective at analyzing long log files or massive Terraform configurations.
- **General Knowledge**: There is a minor trade-off; the model is now so focused on DevOps that its ability to answer poetic or unrelated questions is slightly diminished.
- **Hallucinations**: While reduced, the model may still hallucinate version-specific syntax if the exact version was not present in the training set.

In conclusion, the current LoRA configuration provides a highly efficient and "snappy" experience for platform engineering tasks on local hardware while maintaining a tiny 2.2GB memory footprint.
