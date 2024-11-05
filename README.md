# Pre-SRL
使用RoBERTa的SRL-LSTM/Transformer-Biaffine-CRF的SRL任务

# Parameters

embedder_type: roberta-base
optimizer: adamw
pretr_lr: 2e-05
other_lr: 0.001
max_grad_norm: 1.0
pred_embed_dim: 48
enc_dim: 400
mlp_dim: 300
LSTM(1)-Transformers(1)

# Performance
**CoNLL 2005 WSJ**

| Models  | P. | R. | F1 |
| ------------- | ------------- |------------- |------------- |
| BiLSTM-Span + ELMo + Ensemble | 89.2  |  87.9   |  88.5
| Pre-SRL + RoBERTa Base | 89.05  |  89.17  | 89.11
| SRL-MM + XLNet Large | -  |  -  | 89.80
| CRF2o + RoBERTa Large | 89.45  | 89.63 |  89.54
| LG-HeSyFu + RoBERTa Large | 88.86    | 89.28 |  89.04
| MRC-SRL(SOTA)-RoBERTa Large  | 90.4  | 89.7 | **90.0**

**CoNLL 2005 Brown**

| Models  | P. | R. | F1 |
| ------------- | ------------- |------------- |------------- |
| BiLSTM-Span + ELMo + Ensemble| 81.0   |  78.4   | 79.6
| Pre-SRL + RoBERTa Base|  80.51   |  79.78 | 80.15
| SRL-MM + XLNet Large | -  |  -  | 85.02
| CRF2o + RoBERTa Large | 83.89  | 83.39 |   83.64
| LG-HeSyFu + RoBERTa Large | 83.52   | 83.75 |   83.67
| MRC-SRL(SOTA) + RoBERTa Large  | 86.4  |  83.8 |  **85.1**

**CoNLL 2012 EN**

| Models  | P. | R. | F1 |
| ------------- | ------------- |------------- |------------- |
| BiLSTM-Span + ELMo + Ensemble |  -    |  -    | 87.0
| Pre-SRL + RoBERTa Base |  87.27    | 88.12   | 87.69
| SRL-MM + XLNet Large | -  |  -  |  87.67
| CRF2o + RoBERTa Large  | 88.11  | 88.53 |  88.32
| LG-HeSyFu + RoBERTa Large | 88.09    | 88.83 |  88.59
| MRC-SRL(SOTA) + RoBERTa Large  | 88.6  |   87.9  |  **88.3**
