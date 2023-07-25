# Sarsa：同策略时序差分控制

时序差分方法是给定一个策略，然后我们去估计它的价值函数。接着我们要考虑怎么**使用时序差分方法的框架来估计 Q 函数**，也就是 **Sarsa 算法**。Sarsa 的更新公式与时序差分方法的公式是类似的，它将原本时序差分方法更新 V 的过程，变成了更新 Q，即

$$ Q(s_t,a_t)\gets Q(s_t,a_t) + \alpha(r_{t}+ \gamma Q(s_{t+1},a_{t+1})-Q(s_t,a_t))$$

上式是指我们可以用下一步的 Q 值来更新这一步的 Q 值，不断地强化每一个 Q 值。

具体地，我们用 $Q(s_t,a_t)$ 来逼近 $G_t$，那么 $Q(s_{t+1},a_{t+1})$ 其实就是近似 $G_{t+1}$，我们就可以先把 $r_t+\gamma Q(s_{t+1},a_{t+1})$ 当作 $Q(s_t,a_t)$ 想要逼近的目标值，即时序差分目标，我们想要计算的就是 $Q(s_t,a_t)$ 。因为最开始 Q 值都是随机初始化或者是初始化为 0，所以它需要不断地去逼近它理想中真实的 Q 值（时序差分目标），$r_t+\gamma Q(s_{t+1},a_{t+1})-Q(s_t,a_t)$ 就是时序差分误差。我们用软更新的方式来逼近, 即每次我们只更新一点点，α 类似于学习率，最终 Q 值慢慢地逼近真实的目标值。  
该算法由于每次更新值函数时需要知道 $(s_t, a_t, r_t, s_{t+1}, a_{t+1})$这几个值，因此得名 Sarsa 算法。

## n 步 Sarsa
Sarsa 属于单步更新算法，每执行一个动作，就会更新一次价值和策略。如果不进行单步更新，而是采取 n 步更新或者回合更新，即在执行 n 步之后再更新价值和策略，这样我们就得到了**n 步 Sarsa（n-step Sarsa）**
比如 2 步 Sarsa 就是执行两步后再来更新 Q 函数的值。对于 Sarsa，在 t 时刻的价值为

$$ Q_t=r_{t}+ \gamma Q(s_{t+1},a_{t+1})$$

而对于 n 步 Sarsa，它的 n 步 Q 回报为

$$Q^n_t=r_{t+1}+\gamma r_{t+2}+\dots+\gamma^{n-1}r_{t+n}+\gamma^nQ(s_{t+n},a_{t+n})$$

如果给 Q^n^​~t~ 加上**资格迹衰减参数（decay-rate parameter for eligibility traces）λ** 并进行求和，即可得到Sarsa(λ) 的 Q 回报

![image](assets/image-20220805151623-ea0q2tj.png)

因此，n 步 Sarsa(λ) 的更新策略为

![image](assets/image-20220805151834-hftujxb.png)

总之，Sarsa 和 Sarsa(λ) 的差别主要体现在价值的更新上。

Sarsa 的伪代码如下所示：

* 初始化 $Q(s,a)$
* for 序列 $e=1 \to E$ do：

  * 得到初始状态 $s$
  * 用$\epsilon$ -greedy 策略根据 Q 选择当前状态 s 下的动作 a
  * for 时间步 $t=1\to T$ do :

    * 得到环境反馈的 r, s'
    * 用$\epsilon$ -greedy 策略根据 Q 选择当前状态 s' 下的动作 a'
    * $Q(s,a)\gets Q(s,a)+\alpha[r+\gamma Q(s',a')-Q(s,a)]$
    * $s\gets s',a\gets a'$
  * end for
* end for
