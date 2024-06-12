## 一些笔记
* [RObotic MAnipulation Network (ROMAN) – Hybrid Hierarchical Learning for Solving Complex Sequential Tasks](https://arxiv.org/pdf/2307.00125)
使用 BC（warm-starting）+ GAIL（PPO）来训练操作任务
* 使用 BC 作 warm-starting 看似是一个合理的策略，但在 [Augmenting GAIL with BC for sample efficient imitation learning](https://arxiv.org/pdf/2001.07798) 与 [Sample efficient imitation learning for continuous control]() 中指出，与从头开始训练的 GAIL 相比，使用行为克隆进行预训练并没有帮助，代理学习的是次优策略。
