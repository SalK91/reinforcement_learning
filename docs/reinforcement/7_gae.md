# Chapter 7: Advances in Policy Optimization – GAE, TRPO, and PPO

In the previous chapter, we improved the foundation of policy gradients by reducing variance (using baselines) and introducing actor–critic methods. We also noted that unrestricted policy updates can be unstable and sample-inefficient. In this chapter, we present modern advances in policy optimization that build on those ideas to achieve much better performance in practice. We focus on two main developments: Generalized Advantage Estimation (GAE), which refines how we estimate advantages to balance bias and variance, and trust-region methods (specifically Trust Region Policy Optimization (TRPO) and Proximal Policy Optimization (PPO)) that ensure updates do not destabilize the policy. These techniques enable more sample-efficient, stable learning by reusing data safely and preventing large detrimental policy shifts.


## Generalized Advantage Estimation (GAE)
Accurate and low-variance advantage estimates are crucial for effective policy gradient updates. Recall that the policy gradient update uses the term $\nabla_\theta \log \pi_\theta(a_t|s_t)\,A(s_t,a_t)$ – if $A(s_t,a_t)$ is noisy or biased, it can severely affect learning. Advantage can be estimated via:
- Monte Carlo returns: $A_t = G_t - V(s_t)$ using the full return $G_t$ (summing all future rewards until episode end). This is an unbiased estimator of the true advantage, but it has very high variance because it includes all random future outcomes.

- One-step TD returns: $A_t \approx r_t + \gamma V(s_{t+1}) - V(s_t)$, using the critic’s bootstrapped estimate of the future. This one-step advantage (equivalently the TD error $\delta_t$) has much lower variance (it relies on the learned value for the next state) but is biased by function approximation and by truncating the return after one step.

- n-Step returns:We can also use intermediate approaches, for example a 2-step return $R^{(2)}t = r_t + \gamma r{t+1} + \gamma^2 V(s_{t+2})$ giving an advantage $\hat{A}^{(2)}_t = R^{(2)}_t - V(s_t)$. In general, an n-step advantage estimator can be written as:

    $$A^{t}(n) = \sum_{i=0}^{n-1} \gamma^{i} r_{t+i+1} + \gamma^{n} V(s_{t+n}) -V(s_{t})$$


    which blends $n$ actual rewards with a bootstrap at time $t+n$. Smaller $n$ (like 1) means more bias (due to heavy reliance on $V$) but low variance; larger $n$ (approaching the episode length) reduces bias but increases variance.

The pattern becomes clearer if we express these in terms of the TD error $\delta_t$ (the one-step advantage at $t$):

$$ \delta_t \;=\; r_t + \gamma V(s_{t+1}) - V(s_t). $$
- For a 1-step return, $\hat{A}^{(1)}_t = \delta_t$.
- For a 2-step return, $\hat{A}^{(2)}t = \delta_t + \gamma\,\delta$.
- For an $n$-step return, $\hat{A}^{(n)}t = \delta_t + \gamma \delta_{t+1} + \gamma^2 \delta_{t+2} + \cdots + \gamma^{n-1}\delta_{t+n-1}$

Each additional term $\gamma^i \delta_{t+i}$ extends the return by one more step of real reward before bootstrapping, increasing bias a bit (since it assumes the later $\delta$ terms are based on an approximate $V$) but capturing more actual reward outcomes (reducing variance less).

Generalized Advantage Estimation (GAE) takes this idea to its logical conclusion by forming a weighted sum of all n-step advantages, with exponentially decreasing weights. Instead of picking a fixed $n$, GAE uses a parameter $0 \le \lambda \le 1$ to blend advantages of different lengths:

$$\hat{A}^{\text{GAE}(\gamma,\lambda)}_t \;=\; (1-\lambda)\Big(\hat{A}^{(1)}_t + \lambda\,\hat{A}^{(2)}_t + \lambda^2\,\hat{A}^{(3)}_t + \cdots\Big)$$

This infinite series can be shown to simplify to a very convenient form:

$$\hat{A}_t^{\text{GAE}(\gamma,\lambda)} = \sum_{i=0}^{\infty} (\gamma \lambda)^i\delta_{t+i}$$

which is an exponentially-weighted sum of the future TD errors. In practice, this is implemented with a simple recursion running backward through each trajectory (since it’s a sum of discounted TD errors).

Key intuition: $\lambda$ controls the bias–variance trade-off in advantage estimation:

- $\lambda = 0$ uses only the one-step TD error: $\hat{A}^{\text{GAE}(0)}_t = \delta_t$. This is the lowest-variance, highest-bias estimator (similar to TD(0) advantage)[21].

- $\lambda = 1$ uses an infinitely long sum of un-discounted TD errors, which in theory equals the full Monte Carlo return advantage (since all bootstrapping is deferred to the end). This is unbiased (in the limit of exact $V$) but highest variance – essentially Monte Carlo estimation.

- Intermediate $0<\lambda<1$ gives a mixture. A typical choice is $\lambda = 0.95$ in many applications, which provides a good balance (mostly long-horizon returns with a bit of bootstrapping to damp variance).

GAE is not introducing a new kind of return; rather, it generalizes existing returns. It smoothly interpolates between TD and Monte Carlo methods. When $\lambda$ is low, GAE trusts the critic more (using more bootstrapped estimates); when $\lambda$ is high, GAE leans toward actual returns over many steps.
In modern actor–critic algorithms (including TRPO and PPO), GAE is used to compute the advantage for each state-action in a batch. A typical implementation for each iteration is:

1.	Collect trajectories using the current policy $\pi_{\theta}$ (e.g. run $N$ episodes or $T$ time steps of experience).
2.	Compute state values $V(s_t)$ for each state visited (using the current value function estimate).
3.	Compute TD residuals $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ for each time step.
4.	Apply GAE formula: going from $t=T-1$ down to $0$, accumulate $\hat{A}_t = \delta_t + \gamma \lambda, \hat{A}{t+1}$, with $\hat{A}_{T} = 0$. This yields $\hat{A}_t \approx \sum{i\ge0} (\gamma \lambda)^i \delta{t+i}$.

5.	Use Advantages for Update: These $\hat{A}_t$ values serve as the advantage estimates in the policy gradient update. Simultaneously, you can compute proxy returns for the critic by adding $\hat{A}_t$ to the baseline $V(s_t)$ (i.e. $\hat{R}_t = \hat{A}_t + V(s_t)$, an estimate of the actual return) and use those to update the value function parameters.

The result of GAE is a much smoother, lower-variance advantage signal for the actor, without introducing too much bias. Empirically, this greatly stabilizes training: the policy doesn’t overreact to single high-return episodes, and it doesn’t ignore long-term outcomes either. GAE essentially bridges the gap between the high-variance Monte Carlo world of Chapter 5 and the low-variance TD world of Chapter 3–4, and it has become a standard component in virtually all modern policy optimization algorithms.

## KL Divergence Constraints and Surrogate Objectives
We now turn to the question of stable policy updates. As discussed, a major issue with vanilla policy gradient is that a single update can accidentally push the policy into a disastrous region (because the gradient is computed at the current policy but we might step too far). To make updates safer, we want to constrain how much the policy changes at each step. A natural way to measure change between the old policy $\pi_{\text{old}}$ and a new policy $\pi_{\text{new}}$ is to use the Kullback–Leibler (KL) divergence. For example, we can require:

$$\mathbb{E}_{s \sim d^{\pi_{\text{old}}}}
\left[
D_{\mathrm{KL}}\bigl(\pi_{\text{new}}(\cdot \mid s)\,\|\,\pi_{\text{old}}(\cdot \mid s)\bigr)
\right]
\le \delta$$

for some small $\delta$. This means that on average over states (under the old policy’s state distribution $d_{\pi_{\text{old}}}$), the new policy’s probability distribution is not too far from the old policy’s distribution. A small KL divergence ensures the policies behave similarly, limiting the “surprise” from one update.

But how do we optimize under such a constraint? We need an objective function that tells us whether $\pi_{\text{new}}$ is better than $\pi_{\text{old}}$. Fortunately, theory provides a useful tool: a surrogate objective that approximates the change in performance if the policy change is small. One version, derived from the policy performance difference lemma and monotonic improvement theorem, is:

$$L_{\pi_{\text{old}}}(\pi_{\text{new}})
=
\mathbb{E}_{s,a \sim \pi_{\text{old}}}
\left[
\frac{\pi_{\text{new}}(a \mid s)}{\pi_{\text{old}}(a \mid s)}
A_{\pi_{\text{old}}}(s,a)
\right]$$

This is an objective functional—it evaluates the new policy using samples from the old policy, weighting rewards by the importance ratio $r(s,a) = \pi_{\text{new}}(a|s)/\pi_{\text{old}}(a|s)$. Intuitively, $L_{\pi_{\text{old}}}(\pi_{\text{new}})$ is asking: if the old policy visited state $s$ and took action $a$, how good would that decision be under the new policy’s probabilities? Actions that the new policy wants to do more of ($r > 1$) will contribute their advantage (good or bad) proportionally more.

Critically, one can show that if $\pi_{\text{new}}$ is very close to $\pi_{\text{old}}$ (in KL terms), then improving this surrogate $L$ guarantees an improvement in the true return $J(\pi)$. Specifically, there is a bound such that:
 

$$J(\pi_{\text{new}})
\ge
J(\pi_{\text{old}})
+
L_{\pi_{\text{old}}}(\pi_{\text{new}})
-
C \,
\mathbb{E}_{s \sim d^{\pi_{\text{old}}}}
\left[
D_{\mathrm{KL}}\bigl(\pi_{\text{new}} \,\|\, \pi_{\text{old}}\bigr)[s]
\right]$$

for some constant $C$ related to horizon and policy support. When the KL divergence is small, the last term is second-order (negligible), so roughly we get $J(\pi_{\text{new}}) \gtrapprox J(\pi_{\text{old}}) + L_{\pi_{\text{old}}}(\pi_{\text{new}})$. In other words, maximizing $L$ while keeping KL small ensures monotonic improvement: each update should not reduce true performance.

This insight leads directly to a constrained optimization formulation for safe policy updates:

- Objective: Maximize the surrogate $L_{\pi_{\text{old}}}(\pi_{\text{new}})$ (i.e. maximize expected advantage-weighted probability ratios).
- Constraint: Limit the policy divergence via $D_{\mathrm{KL}}(\pi_{\text{new}}\Vert \pi_{\text{old}}) \le \delta$ (for some small $\delta$).
Algorithms that implement this idea are called trust-region methods, because they optimize the policy within a trust region of the old policy. Next, we discuss two prominent algorithms: TRPO, which tackles the constrained problem directly (with some approximations), and PPO, which simplifies it into an easier unconstrained loss function.


## Trust Region Policy Optimization 
Trust Region Policy Optimization (TRPO) is a seminal algorithm that explicitly embodies the constrained update approach. TRPO chooses a new policy by approximately solving:

$$\max_{\theta_{\text{new}}} \; L_{\theta_{\text{old}}}(\theta_{\text{new}})
\quad \text{s.t.} \quad
\mathbb{E}_{s \sim d^{\pi_{\theta_{\text{old}}}}}
\left[
D_{\mathrm{KL}}\bigl(\pi_{\theta_{\text{new}}} \,\|\, \pi_{\theta_{\text{old}}}\bigr)
\right]
\le \delta$$

where $L_{\theta_{\text{old}}}(\theta_{\text{new}})$ is the surrogate objective defined above, and $\delta$ is a small trust-region threshold. In practice, solving this exactly is difficult due to the infinite-dimensional policy space. TRPO makes it tractable by using a few key ideas:

- Approximating the constraint via a quadratic expansion of the KL divergence (which yields a Fisher Information Matrix). This turns the problem into something like a second-order update (a natural gradient step). In fact, TRPO’s solution can be shown to correspond to a natural gradient ascent:

      $\theta_{\text{new}} = \theta_{\text{old}} + \sqrt{\frac{2\delta}{g^T F^{-1} g}}\; F^{-1} g$$
 
      where $g = \nabla_\theta L$ and $F$ is the Fisher matrix. This ensures the KL constraint is satisfied approximately, and is equivalent to scaling the gradient by $F^{-1}$. In simpler terms, TRPO updates $\theta$ in a direction that accounts for the curvature of the policy space, so that the change in policy (KL) is proportional to the step size.

- Using a line search to ensure the new policy actually improves $J(\pi)$. TRPO will back off the step size if the updated policy violates the constraint or fails to achieve a performance improvement. This safeguard maintains the monotonic improvement guarantee in practice.


A simplified outline of TRPO is:

1. Collect trajectories with the current policy $\pi_{\theta_{\text{old}}}$.
3. Estimate advantages $\hat{A}_t$ for each time step (using GAE or another method for high-quality advantage estimates).
3. Compute surrogate objective $L(\theta) = \mathbb{E}[r_t(\theta), \hat{A}t]$ where $r_t(\theta) = \frac{\pi{\theta}(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$.
4. Approximate KL constraint: Compute the policy gradient $\nabla_\theta L$ and the Fisher matrix $F$ (via sample-based estimation of the Hessian of the KL divergence). Solve for the update direction $p \approx F^{-1} \nabla_\theta L$ (e.g. using conjugate gradient).
5. Line search: Scale and apply the update step $\theta \leftarrow \theta + p$ gradually, checking the KL and improvement. Stop when the KL constraint or improvement criterion is satisfied.

TRPO’s updates are therefore conservative by design – they will only take as large a step as can be trusted not to degrade performance. TRPO was influential because it demonstrated much more stable and reliable training on complex continuous control tasks than vanilla policy gradient.

Strengths and Weaknesses of TRPO: TRPO offers a theoretical guarantee of non-destructive updates – under certain assumptions, each iteration is guaranteed to improve or at least not decrease performance. It uses a natural gradient approach that respects the geometry of policy space, which is more effective than an arbitrary gradient in parameter space. However, TRPO comes at a cost: it requires calculating second-order information (the Fisher matrix), and implementing the conjugate gradient solver and line search adds complexity. The algorithm can be slower per iteration and is more complex to code and tune. In practice, TRPO, while effective, proved somewhat cumbersome for large-scale problems due to these complexities.


## Proximal Policy Optimization

Proximal Policy Optimization (PPO) was introduced as a simpler, more user-friendly variant of TRPO that achieves similar results with only first-order optimization. The core idea of PPO is to keep the spirit of trust-region updates (don’t move the policy too far in one go) but implement it via a relaxed objective that can be optimized with standard stochastic gradient descent.
There are two main variants of PPO:

### KL-Penalty Objective: 

One version of PPO adds the KL-divergence as a penalty to the objective rather than a hard constraint. The objective becomes:

$$J_{\text{PPO-KL}}(\theta)
=
\mathbb{E}\!\left[ r_t(\theta)\, \hat{A}^t \right]
-
\beta \,\mathbb{E}\!\left[
D_{\mathrm{KL}}\!\left(\pi_{\theta} \,\|\, \pi_{\theta_{\text{old}}}\right)
\right]$$

where $\beta$ is a coefficient determining how strongly to penalize deviation from the old policy. If the KL divergence in an update becomes too large, $\beta$ can be adjusted (increased) to enforce smaller steps in subsequent updates. This approach maintains a soft notion of a trust region.


#### Algorithm (PPO with KL Penalty)

1: Input: initial policy parameters $\theta_0$, initial KL penalty $\beta_0$, target KL-divergence $\delta$  
2: for $k = 0, 1, 2, \ldots$ do  
3: $\quad$ Collect set of partial trajectories $\mathcal{D}_k$ using policy $\pi_k = \pi(\theta_k)$  
4: $\quad$ Estimate advantages $\hat{A}^t_k$ using any advantage estimation algorithm  
5: $\quad$ Compute policy update by approximately solving  
$\quad\quad$ $\theta_{k+1} = \arg\max_\theta \; L_{\theta_k}(\theta) - \beta_k \hat{D}_{KL}(\theta \,\|\, \theta_k)$  
6: $\quad$ Implement this optimization with $K$ steps of minibatch SGD (e.g., Adam)  
7: $\quad$ Measure actual KL: $\hat{D}_{KL}(\theta_{k+1}\|\theta_k)$  
8: $\quad$ if $\hat{D}_{KL}(\theta_{k+1}\|\theta_k) \ge 1.5\delta$ then  
9: $\quad\quad$ Increase penalty: $\beta_{k+1} = 2\beta_k$  
10: $\quad$ else if $\hat{D}_{KL}(\theta_{k+1}\|\theta_k) \le \delta/1.5$ then  
11: $\quad\quad$ Decrease penalty: $\beta_{k+1} = \beta_k/2$  
12: $\quad$ end if  
13: end for  


### Clipped Surrogate Objective (PPO-Clip): 

The more popular variant of PPO uses a clipped surrogate objective to restrict policy updates:

$$L^\text{CLIP}(\theta)
=
\mathbb{E}_{t}\!\left[
\min\!\Big(
r_t(\theta)\,\hat{A}^t,\;
\text{clip}\!\big(r_t(\theta),\, 1-\epsilon,\, 1+\epsilon\big)\,\hat{A}^t
\Big)
\right]$$


where $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$ as before, and $\epsilon$ is a small hyperparameter (e.g. 0.1 or 0.2) that defines the clipping range. This objective says: if the new policy’s probability ratio $r_t(\theta)$ stays within $[1-\epsilon,\,1+\epsilon]$, we use the normal surrogate $r_t \hat{A}_t$. But if $r_t$ tries to go outside this range (meaning the policy probability for an action has changed dramatically), we clip $r_t$ to either $1+\epsilon$ or $1-\epsilon$ before multiplying by $\hat{A}_t$. Effectively, the advantage contribution is capped once the policy deviates too much from the old policy.

The clipped objective is not exactly the original constrained problem, but it serves a similar purpose: it removes the incentive for the optimizer to push $r_t$ outside of $[1-\epsilon,1+\epsilon]$. If increasing $|\theta|$ further doesn’t increase the objective (because the min() will select the clipped term), then overly large policy changes are discouraged.

Why Clipping Works: Clipping is a simple heuristic, but it has proven extremely effective:

- It enforces a soft trust region by preventing extreme updates for any single state-action probability. The policy can still change, but not so much that any one probability ratio blows up.

- It avoids the complexity of solving a constrained optimization or computing second-order derivatives – we can just do standard SGD on $L^{CLIP}(\theta)$.

- It keeps importance sampling ratios near 1, which means the algorithm can safely perform multiple epochs of updates on the same batch of data without the estimates drifting too far. This directly improves sample efficiency (unlike vanilla policy gradient, PPO typically updates each batch for several epochs).


#### PPO (Clipped) Algorithm

1: Input: initial policy parameters $\theta_0$, clipping threshold $\epsilon$  
2: for $k = 0, 1, 2, \ldots$ do  
3: $\quad$ Collect a set of partial trajectories $\mathcal{D}_k$ using policy $\pi_k = \pi(\theta_k)$  
4: $\quad$ Estimate advantages $\hat{A}^{\,t}_k$ using any advantage estimation algorithm (e.g., GAE)  
5: $\quad$ Define the clipped surrogate objective  
$\quad\quad$  
$$
\mathcal{L}^{\text{CLIP}}_{\theta_k}(\theta)
= 
\mathbb{E}_{\tau \sim \pi_{\theta_k}}
\left[
\sum_{t=0}^{T}
\min\!\left(
r_t(\theta)\,\hat{A}^t_k,\;
\operatorname{clip}\!\left(r_t(\theta),\, 1-\epsilon,\, 1+\epsilon\right)\hat{A}^t_k
\right)
\right]
$$
6: $\quad$ Update policy parameters with several epochs of minibatch SGD to approximately maximize $\mathcal{L}^{\text{CLIP}}_{\theta_k}(\theta)$  
7: $\quad$ Set $\theta_{k+1}$ to the resulting parameters  
8: end for  


In practice, PPO with clipping has become one of the most widely used RL algorithms because it strikes a good balance between performance and simplicity. It is relatively easy to implement (compared to TRPO) and has been found to be robust across many tasks and hyperparameters. While it doesn’t guarantee monotonic improvement in theory, in practice it achieves stable training behavior very similar to TRPO.

In modern practice, PPO is the dominant choice for policy optimization in deep RL, due to its relative simplicity and strong performance across many environments. TRPO is still important conceptually (and sometimes used in scenarios where theoretical guarantees are desired), but PPO’s convenience usually wins out.

## Putting It Together: Sample Efficiency, Stability, and Monotonic Improvement

The advances covered in this chapter are often used together in state-of-the-art algorithms:

- Generalized Advantage Estimation (GAE) provides high-quality advantage estimates that significantly reduce variance without too much bias. This means we can get away with smaller batch sizes or fewer episodes to get a good learning signal – improving sample efficiency.

- Trust-region update rules (TRPO/PPO) ensure that each policy update is safe and stable – the policy doesn’t change erratically, preventing the kind of catastrophic drops in reward that naive policy gradients can suffer. By keeping policy changes small (via KL constraints or clipping), these methods enable multiple updates on the same batch of data (improving data efficiency) and maintain policy monotonicity, i.e. each update is expected to improve or at least not significantly degrade performance.

- In practice, an algorithm like PPO with GAE is an actor–critic method that uses all these ideas: an actor policy updated with a clipped surrogate objective (making updates stable), a critic to approximate $V(s)$ (enabling advantage estimation), GAE to compute advantages (trading off bias/variance), and typically multiple gradient epochs per batch to squeeze more learning out of each sample. This combination has proven remarkably successful in domains from simulated control tasks to games.

By building on the foundational policy gradient framework and addressing its shortcomings, GAE and trust-region approaches have made deep reinforcement learning much more practical and reliable. They illustrate how theoretical insights (performance bounds, policy geometry) and practical tricks (advantage normalization, clipping) come together to yield algorithms that can solve challenging RL problems while using reasonable amounts of training data and maintaining stability throughout learning. Each component – be it advantage estimation or constrained updates – plays a role in ensuring that learning is as efficient, stable, and monotonic as possible. Together, they represent the state-of-the-art toolkit for policy optimization in reinforcement learning.


| Method                | Key Idea                                 | Pros                     | Cons                     |
|-----------------------|-------------------------------------------|--------------------------|--------------------------|
| REINFORCE             | MC return-based policy gradient          | Simple, unbiased         | Very high variance       |
| Actor–Critic          | TD baseline value function               | More sample-efficient    | Requires critic          |
| Advantage Actor–Critic| Uses $A(s,a)$ for updates                | Best bias–variance trade | Needs accurate value est.|
| TRPO                  | Trust-region with KL constraint          | Strong theory, stable    | Complex, second-order    |
| PPO                   | Clipped/penalized surrogate objective    | Simple, stable, popular  | Heuristic, tuning needed |

## Mental Map
```text
                  Advanced Policy Gradient Methods
     Goal: Fix limitations of vanilla PG (variance, stability, KL control)
                               │
                               ▼
             Core Challenges in Policy Gradient Methods
       ┌────────────────────────────────────────────────────────┐
       │ High variance (MC returns)                             │
       │ Poor sample efficiency (on-policy only)                │
       │ Sensitive to step size → catastrophic policy collapse  │
       │ Small θ change ≠ small policy change                   │
       │ Reusing old data is unstable                           │
       └────────────────────────────────────────────────────────┘
                               │
                               ▼
                     Variance Reduction (Baselines)
       ┌────────────────────────────────────────────────────────┐
       │ Introduce baseline b(s) → subtract expectation         │
       │ Keeps estimator unbiased                               │
       │ Good choice: b(s)= V(s) → yields Advantage A(s,a)      │
       │ Update based on: how much action outperformed expected │
       └────────────────────────────────────────────────────────┘
                               │
                               ▼
                       Advantage Function A(s,a)
       ┌────────────────────────────────────────────────────────┐
       │ A(s,a) = Q(s,a) – V(s)                                 │
       │ Measures how much BETTER the action was vs average     │
       │ Positive → increase πθ(a|s); Negative → decrease it    │
       │ Major variance reduction – foundation of Actor–Critic  │
       └────────────────────────────────────────────────────────┘
                               │
                               ▼
                         Actor–Critic Framework
       ┌────────────────────────────────────────────────────────┐
       │ Actor: policy πθ(a|s)                                  │
       │ Critic: value function V(s;w) estimates baseline       │
       │ TD error δt reduces variance (bootstrapping)           │
       │ Faster, more sample-efficient than REINFORCE           │
       └────────────────────────────────────────────────────────┘
                               │
                               ▼
                     Target Estimation for the Critic
       ┌────────────────────────────┬────────────────────────────┐
       │ Monte Carlo (∞-step)       │  TD (1-step)               │
       │ + Unbiased                 │  + Low variance            │
       │ – High variance            │  – Biased                  │
       ├────────────────────────────┴────────────────────────────┤
       │ n-Step Returns: Blend of TD and MC                      │
       │ Control bias–variance by choosing n                     │
       │ Larger n → MC-like; smaller n → TD-like                 │
       └─────────────────────────────────────────────────────────┘
                               │
                               ▼
             Fundamental Problems with Vanilla Policy Gradient
       ┌────────────────────────────────────────────────────────┐
       │ Uses each batch for ONE gradient step (on-policy)      │
       │ Step size is unstable → huge performance collapse      │
       │ Small changes in θ → large unintended policy changes   │
       │ Need mechanism to limit POLICY CHANGE, not θ change    │
       └────────────────────────────────────────────────────────┘
                               │
                               ▼
            Safe Policy Improvement Theory → TRPO & PPO
       ┌────────────────────────────────────────────────────────┐
       │ Policy Performance Difference Lemma                    │
       │   J(π') − J(π) = Eπ' [Aπ(s,a)]                         │
       │ KL Divergence as policy distance metric                │
       │   D_KL(π'||π) small → safe update                      │
       │ Monotonic Improvement Bound                            │
       │   Lower bound on J(π') using surrogate loss Lπ(π')     │
       └────────────────────────────────────────────────────────┘
                               │
                               ▼
                   Surrogate Objective for Safe Updates
       ┌────────────────────────────────────────────────────────┐
       │ Lπ(π') = E[ (π'(a|s)/π(a|s)) * Aπ(s,a) ]               │
       │ Importance sampling + KL regularization                │
       │ Foundation of Trust-Region Policy Optimization (TRPO)  │
       └────────────────────────────────────────────────────────┘
                               │
                               ▼
              Proximal Policy Optimization (PPO) – Key Ideas
       ┌────────────────────────────┬────────────────────────────┐
       │ PPO-KL Penalty             │ PPO-Clipped Objective      │
       │ Adds β·KL to loss          │ Clips ratio r_t(θ) to      │
       │ Adjust β adaptively        │ [1−ε, 1+ε] to prevent      │
       │ Prevents large updates     │ destructive policy jumps   │
       └────────────────────────────┴────────────────────────────┘
                               │
                               ▼
                         PPO Algorithm Summary
       ┌────────────────────────────────────────────────────────┐
       │ 1. Collect trajectories from old policy                │
       │ 2. Estimate advantages Â_t (GAE, TD, etc.)            │
       │ 3. Optimize clipped surrogate for many epochs          │
       │ 4. Update parameters safely                            │
       └────────────────────────────────────────────────────────┘
                               │
                               ▼
                          Final Outcome (Chapter 6)
       ┌────────────────────────────────────────────────────────┐
       │ Stable and efficient policy optimization               │
       │ Reuse data safely across multiple updates              │
       │ Avoid catastrophic policy collapse                     │
       │ Foundation of modern deep RL algorithms                │
       │ (PPO, TRPO, A3C, IMPALA, SAC, etc.)                    │
       └────────────────────────────────────────────────────────┘
```