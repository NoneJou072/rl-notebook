# A2C

在[带基线的策略梯度](../REINFORCE/REINFORCE.md)中，我们得到了策略梯度的一个无偏估计：

$$g(s,a;\theta):=[Q_\pi(s,a)-V_\pi(s)]\cdot\bigtriangledown_\theta\ln\pi(a\mid s;\theta)$$

其中，$Q_\pi(s,a)-V_\pi(s)$ 称为优势函数(advantage function)。因此，基于该式得到的 [actor-critic](../actor-critic/actor-critic.md) 被称为 advantage actor-critic（A2C）。  
根据 actor-critic ，我们将 $Q_\pi(s,a)$ 替换为时序差分目标，再把价值函数替换为价值网络，策略梯度近似为

$$g(s,a;\theta):=[r_t+\gamma\cdot v(s_{t+1};\omega)-v(s_{t};\omega)]\cdot\bigtriangledown_\theta\ln\pi(a\mid s;\theta)$$

我们用上式来更新策略网络的参数 $\theta$，以增大目标函数 $J(\theta)$