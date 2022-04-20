# PSL-MOCO
Code for ICLR2022 Paper: [Pareto Set Learning for Neural Multi-objective Combinatorial Optimization](https://openreview.net/forum?id=QuObT9BTWo)

It contains the training and testing codes for three multi-objective combinatorial optimization (MOCO) problems:

- Multi-Objective Travelling Salesman Problem (MOTSP)
- Multi-Objective Capacitated Vehicle Routing Problem (MOCVRP)
- Multi-Objective Knapsack Problem (MOKP)

This code is heavily based on the [POMO repository](https://github.com/yd-kwon/POMO), and it has been reorganized accroding to the [new POMO version](https://github.com/yd-kwon/POMO/tree/master/NEW_py_ver). The main changes include:

- Graph embedding has been removed. 
- BatchNorm has been replaced by InstanceNorm.

**Quick Start**

- To train a model, such as MOTSP with 20 nodes, run *train_motsp_n20.py* in the corresponding folder.
- To test a model, such as MOTSP with 20 nodes, run *test_motsp_n20.py* in the corresponding folder.
- Pretrained models for each problem can be found in the *result* folder.

**Reference**

If our work is helpful for your research, please cite our paper:
```
@inproceedings{lin2022pareto,
  title={Pareto Set Learning for Neural Multi-Objective Combinatorial Optimization},
  author={Xi Lin, Zhiyuan Yang, Qingfu Zhang},
  booktitle={International Conference on Learning Representations},
  year={2022},
  url={https://openreview.net/forum?id=QuObT9BTWo}
}
```

If you find our code useful, please also consider citing the POMO paper:
```
@inproceedings{Kwon2020pomo,
  author = {Yeong-Dae Kwon, Jinho Choo, Byoungjip Kim, Iljoo Yoon, Youngjune Gwon, Seungjai Min},
  booktitle = {Advances in Neural Information Processing Systems},
  pages = {21188--21198},
  title = {POMO: Policy Optimization with Multiple Optima for Reinforcement Learning},
  volume = {33},
  year = {2020}
}
```
