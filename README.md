# Prototypical Networks for Few-shot Learning
University of Cambridge MLMI4 Team 4's replication of the paper [Prototypical Networks for Few-shot Learning](https://arxiv.org/abs/1703.05175) by Jake Snell, Kevin Swersky, Richard S. Zemel. Our implementation is inspired by [this repo](https://github.com/orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch). We implement few-shot classification for the Omniglot and MiniImageNet datasets as well as zero-shot learning for the CUB dataset.

## Replication Results
### Omniglot
| Model | 1-shot (5-way Acc.) | 5-shot (5-way Acc.) | 1 -shot (20-way Acc.) | 5-shot (20-way Acc.)|
| --- | --- | --- | --- | --- |
| Paper | 98.8% | 99.7% | 96.0% | 98.9%|
| Our replication | 98.6% | 99.6%| 95.3% | 98.7% |

### Visualization of Embedding Space (Omniglot, 5-way-5-shot, 1 test episode)
![Visualization of Embedding SpaceMLP for MNIST training curve](https://github.com/Wenlin-Chen/Prototypical-Networks-for-Few-shot-Learning/blob/main/src/visualization.png)

### MiniImageNet
| Model | 1-shot (5-way Acc.) | 5-shot (5-way Acc.) |
| --- | --- | --- |
| Paper | 48.37% | 65.37% |
| Our replication | 49.42 &pm 0.78% | 68.20 &pm 0.66%|
