# knowledge-distillation
### Introduction

A Pytorch implement of knowledge-distillation for fitnet and softmax-T.

Inspired by [knowledge-distillation-PyTorch](https://github.com/peterliht/knowledge-distillation-pytorch), however , presenting a more common and direct implement.

### Use

If you use a offline dataset, please put the offline package in the `/data` folder.

You can modify the train config in the file  `/experiments/params.json` . 

Train: `python3 main.py`

### Experiment

| 实验时间    | student | student Acc | teacher  | teacher Acc | KD Acc | loss function | epoch | 注释                               |
| ----------- | ------- | ----------- | -------- | ----------- | ------ | ------------- | ----- | ---------------------------------- |
| 2021.3.8.11 | CNN     | 0.7511      | DenseNet | 0.8194      | 0.7511 | FitNet        | 30    | initial version                    |
| 2021.3.8.13 | CNN     | 0.8412      | DenseNet | 0.9273      | 0.8600 | FitNet        | 30    | common version of overfit DenseNet |
| 2021.3.9.09 | CNN     | 0.8412      | DenseNet | 0.9470      | 0.8667 | FitNet        | 30    | common DenseNet                    |
| 2021.3.9.10 | CNN     | 0.8412      | DenseNet | 0.9470      | 0.8831 | softmaxT      | 100   | softmaxT loss function             |



