# knowledge-distillation
### Introduction

A Pytorch implement of knowledge-distillation for fitnet and softmax-T.

Inspired by [knowledge-distillation-PyTorch](https://github.com/peterliht/knowledge-distillation-pytorch), however , presenting a more common and direct implement.

### Use

If you use a offline dataset, please put the offline package in the `/data` folder.

You can modify the train config in the file  `/experiments/params.json` . 

Train: `python3 main.py`

### Experiment

| TimeLine    | student net | student acc | teacher net | teacher acc | kd acc | loss function | epoch | Comments                         |
| ----------- | ----------- | ----------- | ----------- | ----------- | ------ | ------------- | ----- | -------------------------------- |
| 2021.3.8.11 | cnn         | 0.7511      | densenet    | 0.8194      | 0.7511 | fitnet        | 30    | init version                     |
| 2021.3.8.13 | cnn         | 0.8412      | densenet    | 0.9273      | 0.8600 | fitnet        | 30    | common version, overfit densenet |
| 2021.3.9.09 | cnn         | 0.8412      | densenet    | 0.9470      | 0.8667 | fitnet        | 30    | common densenet                  |
| 2021.3.9.10 | cnn         | 0.8412      | densenet    | 0.9470      | 0.8831 | softmaxT      | 100   | softmaxT loss function           |
| 2021.3.9.15 | cnn         | 0.8412      | densenet    | 0.9470      | 0.8805 | fitnet        | 100   | enlarge epoch number             |
| 2021.3.9.17 | cnn         | 0.8650      | densenet    | 0.9470      | 0.8841 | fitnet        | 100   | improve cnn acc                  |
| 2021.3.9.18 | cnn         | 0.8650      | densenet    | 0.9470      | 0.8754 | fitnet        | 100   | change T from 20 to 4            |

