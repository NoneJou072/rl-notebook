# Hindsight Experience Replay

## 问题背景

传统的强化学习算法都局限在单个目标上，训练好的算法只能完成这个目标，不能完成其他目标。当面对较复杂的复合任务时，需要训练多个智能体，每个智能体完成一个子任务，最后将这些子任务组合起来完成复合任务，这样会增加训练的复杂度。
我们以机械臂操作任务为例。在使用传统的强化学习算法时，采用单一策略只能训练抓取同一个位置的物体。对于不同的目标位置，要训练多个策略。另外这类环境通常依赖于人工进行奖励塑形，而设计一个好的奖励函数通常很困难。

因此在最近的研究中，人们更希望使用一个二进制信号来设置奖励，即稀疏奖励，只有当成功完成任务后才能够获得奖励，在其他的时刻将不会得到奖励。然而这种情况下，会产生一个样本有效性的问题。即智能体在训练的探索过程中，大部分时间内它都无法完成任务，或仅有几次完成了任务，导致有价值的样本数目非常少，智能体需要更多的时间进行探索，导致指数级的样本复杂度。

为了解决上面的问题，下面介绍 Hindsight Experience Reply (HER) 方法。

## 简单介绍

HER 的原理是回放每个具有任意目标的轨迹。当智能体在一个回合中没有达成目标时，可以使用另外的目标来替代原来的目标，之后根据这个新的目标重新计算奖励，来为智能体提供经验。

具体地，在 HER 方法下的马尔可夫决策过程表述为一个五元组，其中的状态 $S$ 不仅包含观测 $S_o$，还包括一个期望目标 $S_{dg}$， 且 $S_{dg} \in S_o$。

HER改善了稀疏奖励 DRL 的采样效率，可以将该方法与任意的 off-policy 算法结合。当使用了 HER 后，传统 RL 就变成了 Goal-Conditioned RL，一般译为基于目标导向的强化学习。


## 实现过程

实现 HER 算法的关键是构建 replay buffer，包含以下两步：

1. 初始化一个固定长度的队列，每次进入一个五元组 $(s_t,a_t,r_t,s_{t+1},g)$，定义初始状态 $s_0$ 和目标 $g$, 智能体根据当前状态 $s_t$ 和目标 $g$ 来采取动作。
   奖励的计算如下：

$$
r_t \gets r(a_t,s_t||g)
$$

采样出的 $(s_t,a_t,r_t,s_{t+1},g)$ 将被存放到 replay buffer 中。之后，每次都会基于实际目标 $g$ 采样一段完整的智能体经验序列。
2. 使用新的目标 $g'$ 重新计算奖励：

$$
r' \gets r(a_t,s_t||g')
$$

我们构建新的 transitions $(s_t, a_t, r′, s_{t+1}, g′)$ ，也存放到 replay buffer 中。

原文中给出了四种方法去获得新的目标 $g'$:

* final: 将每个回合最后的状态作为新目标
* future: 随机选择 $k$ 个在这个轨迹上并且在当前transition之后的状态作为新目标
* episode: 每次选择 $k$ 个在这个轨迹上的状态作为新目标
* random: 每次在所有出现过的状态里面选择 $k$ 个状态作为新目标

一般我们使用 future 方法，因为它能够更好地利用经验。

## 局限性

*这里引用[4]中的文字*。

HER 方法可以通过目标重标记策略，产生足够数量的非负奖励样本，即使智能体实际上没有完成任务。然而，在具有稀疏奖励的复杂顺序物体操作任务中(智能体必须按顺序成功完成每个子任务，以达到期望的最终目标)，由于隐含的 **一致非负奖励（Identical Non-Negative Reward，INNR）** 问题，智能体仍然会受到样本效率低下的困扰。当智能体在探索过程中无法影响已实现目标 (achieved-goal)时，就会出现INNR问题。在这种情况下，智能体无法从相同负奖励的原始样本，或一致非负的虚拟样本中，区分哪个动作更好。换句话说，实际探索的样本都是负样本，目标重标记的虚拟样本都是正样本，因此并不能从这些样本中区分好坏。因此，来自HER的INNR对于策略改进几乎没有帮助，甚至会降低策略的探索能力。这个隐含的INNR问题是HER在标准操作任务中样本效率低下的原因。

例如，在Push任务中，智能体必须先接近物体，然后将其推到期望的位置。当智能体在一个episode中未能改变物体的位置时，实现目标（achieved-goal）在整个episode中是相同的，所有的 $s_{ag}$ 都是相同的。然后，在经验回放过程中，所有的事后目标 $s_{dg^h}$ 也将与 $s_{ag}$ 具有相同的值。当 $s_{ag}$ 和 $s_{dg^h}$ 相同时，所有的事后样本都是成功的。然而，这样的成功并不是由智能体的行动造成的。这些样本对智能体的策略改进没有帮助，甚至阻碍了学习。我们将这种现象称为INNR问题。

## References

1. [Hindsight Experience Replay (neurips.cc)](https://proceedings.neurips.cc/paper_files/paper/2017/file/453fadbd8a1a3af50a9df4df899537b5-Paper.pdf)
2. [【强化学习算法 34】HER - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/51357496)
3. [[强化学习5] HER（Hindsight Experience Replay） - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/403527126)
4. [[2208.00843] Relay Hindsight Experience Replay: Self-Guided Continual Reinforcement Learning for Sequential Object Manipulation Tasks with Sparse Rewards (arxiv.org)](https://arxiv.org/abs/2208.00843)
5. [baselines/baselines/her at master · openai/baselines (github.com)](https://github.com/openai/baselines/tree/master/baselines/her)
