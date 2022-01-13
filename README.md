# GoEmotions

A fineturned BERT on [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions) with [Huggingface Transformers](https://github.com/huggingface/transformers)

### Description
1. Based on the uncased BERT pretrained model with a linear output layer.
2. Added several commonly-used emoji and tokens to the special token list of the tokenizer.
3. Used weighted loss and focal loss with label smoothing while training.

### Training Parameters
Pretrained Model: BERT-Base, Uncased (12-layer, 768-hidden, 12-heads)
| Parameter         |      |
| ----------------- | ---: |
| Learning rate     | 5e-6 |
| Warmup proportion |  0.1 |
| Epochs (Original) |   20 |
| Epochs (Else)     |   10 |
| Max Seq Length    |   50 |
| Batch size        |   16 |

## Results

Best Results of `Macro F1`

| Macro F1 (%) |  Dev  | Test  |
| ------------ | :---: | :---: |
| original     | 51.75 | 53.11 |
| group        | 70.09 | 70.06 |
| ekman        | 64.12 | 63.76 |

## Reference

- [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions)
- [GoEmotions Github](https://github.com/google-research/google-research/tree/master/goemotions)
- [Huggingface Transformers](https://github.com/huggingface/transformers)
- [GoEmotions-PyTorch](https://github.com/monologg/GoEmotions-pytorch)
