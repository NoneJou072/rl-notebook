## 策略梯度
假设目标策略是一个随机性策略，并且处处可微，我们一般用神经网络去近似策略函数 $\pi(a\mid s)$，称为策略网络 $\pi(a\mid s;\theta)$，其中  $\theta$ 表示神经网络的参数，初始时是随机的。
如果一个策略很好，那么状态价值 $V_\pi(S)$ 的均值应该很大，因此我们定义目标函数：

$$J(\theta)=\mathbb E_S\left[V_\pi(S) \right]$$

我们要让目标函数值越大越好，求解最大化问题，可以使用 **梯度上升**（gradient ascent）更新 $\theta$, 使 $J(\theta)$ 增大，表示为：

$$\theta_{new}\gets \theta_{new}+\beta\cdot\bigtriangledown_\theta J(\theta_{now})$$

其中，$\beta$ 表示学习率，是一个超参数。其中的梯度 $\bigtriangledown_\theta J(\theta_{now})$ 称为策略梯度，可以写成

$$\bigtriangledown_\theta J(\theta)=\mathbb{E}_S\left[\mathbb{E}_{A\sim\pi(\cdot\mid S;\theta)}\left[Q_\pi(S,A)\cdot\bigtriangledown_\theta\ln\pi(A\mid S;\theta)\right]\right]$$

由于上式的求解较为复杂，我们使用近似策略梯度的方法。我们每次观测到 S 中的一个状态 s，可以根据当前的策略网络随机抽样出一个动作 a，计算随机梯度

$$g(s,a;\theta):=Q_\pi(s,a)\cdot\bigtriangledown_\theta\ln\pi(a\mid s;\theta)$$

显然 $g(s,a;\theta)$ 是策略梯度 $\bigtriangledown_\theta J(\theta)$ 的无偏估计：
$$\bigtriangledown_\theta J(\theta)=\mathbb{E}_S\left[\mathbb{E}_{A\sim\pi(\cdot\mid S;\theta)}[g(S,A;\theta)]\right]$$

我们使用随机梯度上升来更新 $\theta$:
$$\theta\gets \theta+\beta\cdot g(s,a;\theta)$$

但由于我们不知道 $Q_\pi(s,a)$，所以无法计算出 $g(s,a;\theta)$ ，解决方法是对 $Q_\pi(s,a)$ 作近似。一种方法是 REINFORCE，用实际观测的回报 u
 近似，另一种方法是 actor-critic，用神经网络 $q(s,a;\omega)$ 近似。

## REINFORCE
由于动作价值定义为折扣回报 $U_t$ 的条件期望：
$$Q_\pi(s_t,a_t)=\mathbb E[U_t\mid S_t=s_t,A_t=a_t]$$

我们可以用蒙特卡洛方法近似上面的条件期望。从时刻 t 开始，智能体完成一个回合后，可以计算出 $u_t=\sum_{k=t}^n\gamma^{k-t}\cdot r_k$。因为 $u_t$ 是 随机变量 $U_t$ 的观测值，所以 $u_t$ 是上式中期望的蒙特卡洛方法近似。我们用 $u_t$ 代替 $Q_\pi(s_t,a_t)$，随机梯度就可以近似成

$$g(s_t,a_t;\theta)=u_t\cdot\bigtriangledown_\theta\ln\pi(a_t\mid s_t;\theta)$$

这样我们得到了训练策略网络的方法，即 REINFORCE。REINFORCE 是同策略的算法，要求行为策略和目标策略相同。


REINFORCE 算法的具体算法流程如下：

* 初始化策略参数 $\theta$
* for 序列 $e=1 \to E$ do :
    * 用当前策略采样轨迹 $\{s_1,a_1,r_1,s_2,a_2,r_2,\dots,s_T,a_T,r_T\}$
    * 计算当前轨迹每个时刻往后的回报 $u_t=\sum_{k=t}^n\gamma^{k-t}\cdot r_k$
    * 对 $\theta$ 进行更新，$\theta\gets \theta+\beta\cdot g(s,a;\theta)$
* end for