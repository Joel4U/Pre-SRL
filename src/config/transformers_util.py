from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW

import torch.nn as nn

from logger import get_logger

logger = get_logger()
def get_huggingface_optimizer_and_scheduler(model: nn.Module,
                                            pretr_lr: float,
                                            other_lr: float,
                                            num_training_steps: int,
                                            weight_decay: float = 0.0,
                                            eps: float = 1e-8,
                                            warmup_step: int = 0):
    """
    Copying the optimizer code from HuggingFace.
    """
    logger.info(f"Using AdamW optimizer with BERT/RoBerta lr: {pretr_lr}, other lr: {other_lr}, "
                f"eps: {eps}, weight decay: {weight_decay}, warmup_step: {warmup_step}")

    no_decay = ["bias", "LayerNorm.weight"]
    # optimizer_grouped_parameters = [
    #     {
    #         "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
    #         "weight_decay": weight_decay,
    #     },
    #     {
    #         "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
    #         "weight_decay": 0.0,
    #     },
    # ]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if "word_embedder" in n and not any(nd in n for nd in no_decay)],
            "lr": pretr_lr,
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if "word_embedder" in n and any(nd in n for nd in no_decay)],
            "lr": pretr_lr,
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in model.named_parameters() if "word_embedder" not in n and not any(nd in n for nd in no_decay)],
            "lr": other_lr,
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if "word_embedder" not in n and any(nd in n for nd in no_decay)],
            "lr": other_lr,
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, eps=eps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_step, num_training_steps=num_training_steps
    )
    return optimizer, scheduler

