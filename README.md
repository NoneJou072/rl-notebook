我的强化学习笔记，持续更新中...  
---
目标是介绍每一种RL算法的
* 算法原理 ✔️
* Pytorch实现 ✔️
* 调参方法❌

单智能体强化学习算法：  

| 算法                                        | Policy     | Based        |
|-------------------------------------------|------------|--------------|
| 👉 [Sarsa](Sarsa/Sarsa.md)                | on-policy  | value-based  |
| 👉 [Q-Learning](Q-learning/Q_learning.md) | off-policy | value-based  |
| 👉 [DQN](DQN/DQN.md)                      | off-policy | value-based  |
| 👉 [Rainbow-DQN]() ❌                    | off-policy | value-based  |
| 👉 [REINFORCE](REINFORCE/REINFORCE.md)    | on-policy  | policy-based |
| 👉 [actor-critic](actor-critic/actor-critic.md) | on-policy  | policy-based |
| 👉 [DDPG](DDPG/DDPG.md)                   | off-policy | Actor-Critic |
| 👉 [TD3](TD3/TD3.md)                      | off-policy | Actor-Critic |
| 👉 [TRPO]() ❌                           | on-policy  | Actor-Critic |
| 👉 [PPO-Continuous](PPO/PPO.md)           | on-policy  | Actor-Critic |
| 👉 [SAC](SAC/SAC.md)                      | off-policy | Actor-Critic |

多智能体强化学习算法：  
Coming soon

运行示例
---

> 运行环境：  
python(in Pycharm)- 3.9.13  
gymnasium-0.28.1  
numpy-1.24.3  
torch-1.12.0  

建议使用 Pycharm 运行，vscode或终端启动会有路径问题。建议更新 `gymnasium` 和 `pytorch` 到最新版本。算法原理请参考每个算法文件夹内的 markdown 文件，内部实现参考以算法名称命名的 `.py` 脚本。想要训练可以运行 `train.py` 脚本。

在部分算法中，添加了 tensorboard 模块，在对应算法文件夹内启动训练后会生成 log 文件夹，通过下面的终端命令可以打开网页查看训练日志：
```shell
tensorboard --logdir .
```

参考文献
---
* [OpenAI-Spinningup](https://spinningup.openai.com/en/latest/algorithms/sac.html)
* [Easy-RL (蘑菇书)]()
* 深度强化学习-王树森等
* [Github:Lizhi-sjtu/DRL-code-pytorch](https://github.com/Lizhi-sjtu/DRL-code-pytorch/blob/main/2.Actor-Critic/README.md)  
* [动手学强化学习](https://hrl.boyuai.com/chapter/2/dqn%E7%AE%97%E6%B3%95)

* [博客园: 强化学习调参技巧二：DDPG、TD3、SAC算法为例](https://www.cnblogs.com/ting1/p/16984892.html)
