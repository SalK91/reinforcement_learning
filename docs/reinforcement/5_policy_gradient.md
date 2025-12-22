# Chapter 5: Policy Gradient Methods

In previous chapters, we derived policies indirectly from value functions using greedy or ε-greedy strategies. However, value-based RL has several challenges:

* Does not naturally support stochastic policies  
* Struggles in continuous action spaces  
* Optimizing through value functions is often indirect and unstable

Policy Gradient Methods directly optimize the policy itself:

$$
\pi_\theta(a|s) = P(a \mid s; \theta)
$$

Our goal becomes:

$$
\theta^* = \arg\max_\theta V(\theta)
$$

That is, learn policy parameters θ that maximize expected return.

> Value-based methods struggle in these cases because they do not directly learn the policy. Instead, they estimate action values $Q(s,a)$ and derive a policy using greedy or ε-greedy strategies. This makes the policy indirect and unstable. Small changes in $Q(s,a)$ can suddenly change the best action, making learning discontinuous and erratic — especially with function approximation like neural networks. Furthermore, value-based methods do not naturally support stochastic or continuous action spaces, since computing $\arg\max_a Q(s,a)$ is infeasible when actions are continuous or infinite. Policy-based methods solve this problem by directly modeling and learning the policy, such as using a softmax distribution for discrete actions or Gaussian distributions for continuous actions.

##  Value-Based vs Policy-Based RL

| Approach | What is Learned? | Policy Type | Works in Continuous Actions? |
|----------|------------------|-------------|-------------------------------|
| Value-Based | $V(s)$ or $Q(s,a)$ | Indirect (ε-greedy, greedy) | No |
| Policy-Based | $\pi_\theta(a/s)$ | Direct, stochastic | Yes |
| Actor-Critic | Both | Direct & learned | Yes |


## Policy Optimization Objective
In policy-based reinforcement learning, the policy itself is directly parameterized as $\pi_\theta(a|s)$, and our goal is to find the parameters $\theta$ that produce the best possible behavior. For episodic tasks starting at initial state $s_0$, the quality of a policy is measured by its expected return:

$$
V(\theta) = V_{\pi_\theta}(s_0) = \mathbb{E}_{\pi_\theta}[G_0]
$$

Therefore, policy optimization can be formulated as an optimization problem, where the goal is to find the policy parameters $\theta$ that maximize the expected return:

$$
\theta^* = \arg\max_\theta V(s_0, \theta)
$$
 

This optimization does not necessarily require gradients.  We can also use gradient-free (derivative-free) optimization methods such as:

- Hill Climbing – Iteratively adjusts parameters in small random directions and keeps changes that improve performance.
- Simplex / Amoeba / Nelder-Mead – Uses a geometric shape (simplex) to explore the parameter space and moves it towards higher-performing regions.
- Genetic Algorithms – Evolves a population of candidate policies using selection, crossover, and mutation, inspired by natural evolution.
- Cross-Entropy Method (CEM) – Samples multiple policy candidates, selects the top performers, and updates the sampling distribution towards them.
- Covariance Matrix Adaptation (CMA) – Adapts both the mean and covariance of a Gaussian distribution to efficiently search complex, high-dimensional policy spaces.

Gradient-free policy optimization methods are often excellent and simple baselines to try.  They are highly flexible, can work with any policy parameterization (including non-differentiable ones), and are easy to parallelize, as policies can be evaluated independently across multiple environments. However, these methods are typically less sample efficient because they treat each policy evaluation as a black box and ignore the temporal structure of trajectories. They do not make use of gradients, value functions, or bootstrapping.

To improve efficiency, we can use gradient-based optimization techniques, which exploit  the structure of the return function and update parameters using local information. Common gradient-based optimizers include:

- Gradient Descent
- Conjugate Gradient
- Quasi-Newton Methods (e.g., BFGS, L-BFGS)

These methods are generally more sample efficient, especially in large or continuous state-action spaces.


## Policy Gradient

The goal is to find parameters $\theta$ that maximize the expected return. Policy gradient algorithms search for a local maximum of $V(\theta)$ by performing gradient ascent:

$$
\Delta \theta = \alpha \nabla_\theta V(s_0, \theta)
$$

where $\alpha$ is the step-size (learning rate) and $\nabla_\theta V(s_0, \theta)$ is the policy gradient.

Assuming an episodic MDP with discount factor $\gamma = 1$, the value of a parameterized policy $\pi_\theta$ starting from state $s_0$ is

$$
V(s_0,\theta) 
= \mathbb{E}_{\pi_\theta} \left[ \sum_{t=0}^{T} r(s_t, a_t); \; s_0, \pi_\theta \right],
$$

where the expectation is taken over the states and actions visited when following $\pi_\theta$. This policy value can be re-expressed in multiple ways.  


* First, in terms of the action-value function:
    $$
    V(s_0,\theta) = \sum_{a} \pi_\theta(a \mid s_0) \, Q(s_0,a;\theta).
    $$

* Second, in terms of full trajectories. Let a state–action trajectory be:
    
    $\tau = (s_0,a_0,r_1,s_1,a_1,r_2,\dots,s_{T-1},a_{T-1},r_T),$
    
    and define
    
    $$
    R(\tau) = \sum_{t=0}^{T} r_{s_t,a_t}$$
    as the sum of rewards of trajectory $\tau$.  

    Let $P(\tau;\theta)$ denote the probability of trajectory $\tau$ when starting in $s_0$ and following policy $\pi_\theta$. Then:

    $$
    V(s_0,\theta) = \sum_{\tau} P(\tau;\theta) \, R(\tau).$$

In this trajectory notation, our optimization objective becomes

$$
\theta^* 
= \arg\max_{\theta} V(s_0,\theta) 
= \arg\max_{\theta} \sum_{\tau} P(\tau;\theta) \, R(\tau).
$$


Taking gradient yields:

$$
\nabla_\theta V(\theta) = 
\sum_{\tau} P(\tau|\theta)R(\tau) 
\nabla_\theta \log P(\tau|\theta)
$$


Using sampled trajectories:

$$
\nabla_\theta V(\theta) \approx
\frac{1}{m}\sum_{i=1}^{m} 
R(\tau^{(i)})
\nabla_\theta \log P(\tau^{(i)}|\theta)
$$

Trajectory probability:

$$
P(\tau|\theta) =
P(s_0)
\prod_{t=0}^{T-1}
\pi_\theta(a_t|s_t)\cdot P(s_{t+1}|s_t,a_t)
$$

Since dynamics are independent of $\theta$:

$$
\nabla_\theta \log P(\tau|\theta) =
\sum_{t=0}^{T-1}
\nabla_\theta \log \pi_\theta(a_t|s_t)
$$

Thus:

$$
\nabla_\theta V(\theta) \approx
\frac{1}{m}\sum_{i=1}^{m}
R(\tau^{(i)})
\sum_{t=0}^{T-1} 
\nabla_\theta \log \pi_\theta(a_t^{(i)}|s_t^{(i)})
$$


The term $\nabla_\theta \log \pi_\theta(a_t|s_t)$ is called the score function. 
It is the gradient of the log of a parameterized probability distribution and measures how sensitive the policy’s action probability is to changes in the parameters $\theta$.  It plays a central role in policy gradient methods because it allows us to estimate gradients without knowing the environment dynamics, using only samples from the policy.


####  Softmax Policy (Discrete Action Spaces)

In discrete action spaces, a common parameterization of the policy is the softmax policy, which assigns probabilities based on exponentiated weighted features. Each action is represented using feature vector $\phi(s,a)$, and the policy is defined as:

$$
\pi_\theta(a|s) = 
\frac{e^{\phi(s,a)^T \theta}}
     {\sum_{a'} e^{\phi(s,a')^T \theta}}
$$

The corresponding score function is:

$$
\nabla_\theta \log \pi_\theta(a|s)
= \phi(s,a) - \mathbb{E}_{a' \sim \pi_\theta}[\phi(s,a')]
$$

This means the gradient increases the probability of the selected action's features and decreases the probability of competing actions based on their expected feature values.

#### Gaussian Policy (Continuous Action Spaces)

For continuous action spaces, the Gaussian policy is a natural choice.  The policy outputs actions by sampling from a normal distribution:

$$
a \sim \mathcal{N}(\mu(s), \sigma^2)
$$

The mean is a linear function of state features:

$$
\mu(s) = \phi(s)^T \theta
$$

If we assume a fixed variance $\sigma^2$, the score function becomes:

$$
\nabla_\theta \log \pi_\theta(a|s)
= \frac{(a - \mu(s))}{\sigma^2} \, \phi(s)
$$

This tells us that the gradient increases the likelihood of actions that are close to the mean $\mu(s)$ and reduces the probability of actions that deviate from it.


Deep neural networks (and other differentiable models) can also be used  
to represent $\pi_\theta(a|s)$, allowing score functions to be computed automatically using backpropagation.
 

> Intution: 
> Think of a sample trajectory $\tau$ as something we tried — a sequence of states, actions, and rewards collected during an episode. The return $R(\tau)$ tells us how good that sample was (higher return means better behavior).
The gradient term $\nabla_\theta \log P(\tau|\theta)$ tells us how to adjust the policy parameters $\theta$ to make the trajectory more or less likely.
> So, when we multiply them:
> $$
R(\tau) \, \nabla_\theta \log P(\tau \mid \theta)$$
> we are effectively saying:
> If a trajectory was good, update the policy to make it more likely to occur again.  
> If it was bad, update the policy to make it less likely.
> This simple idea is the core of policy gradient methods.



## REINFORCE Algorithm (Monte Carlo Policy Gradient)

Update rule:

$$
\Delta \theta = \alpha \nabla_\theta \log \pi_\theta(a_t|s_t)\, R_t
$$

Algorithm:

1: Initialize policy parameters $\theta$  
2: loop (for each episode)  
3: $\quad$ Generate a trajectory $\tau = (s_0, a_0, r_1, \dots, s_T)$ using $\pi_\theta$  
4: $\quad$ for each time step $t$ in $\tau$  
5: $\quad\quad$ Compute return: $\qquad R_t = \sum_{k=t}^{T-1} \gamma^{\,k-t} r_{k+1}$  
6: $\quad\quad$ Compute policy gradient term $\nabla_\theta \log \pi_\theta(a_t|s_t)$  
7: $\quad\quad$ Update policy parameters: $\theta \leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta(a_t|s_t)\, R_t$  
8: $\quad$ end for  
9: end loop  


## Policy Gradient Methods — Mental Map
```text
                     Policy Gradient Methods
    Goal: Learn the optimal policy π* directly (no Q or V tables)
                               │
                               ▼
         Key Concept: Parameterized Policy πθ(a|s)
       ┌─────────────────────────────────────────────┐
       │ Policy is a function with parameters θ      │
       │ πθ(a|s) gives probability of taking action a│
       │ Optimization targets J(θ)=Expected Return   │
       └─────────────────────────────────────────────┘
                               │
                    Direct Policy Optimization
                               │
                               ▼
                 Optimization Objective (J(θ))
       ┌─────────────────────────────────────────────┐
       │ θ* = argmaxθ V(θ) = argmaxθ Eπθ[G₀]         │
       │ Search in parameter space for best policy   │
       └─────────────────────────────────────────────┘
                               │
                               ▼
              Two Families of Policy Optimization
       ┌────────────────────────────┬────────────────────────────┐
       │  Gradient-Free Methods     │   Gradient-Based Methods   │
       └────────────────────────────┴────────────────────────────┘
                │                                      │
                │                                      ▼
                │                          Policy Gradient Methods
                │                                      │
                ▼                                      ▼
   No gradient needed                     Uses ∇θ log πθ(a|s) * Return
   – Hill Climbing                        – REINFORCE
   – CEM, CMA                             – Actor-Critic
   – Genetic Algorithms                   – Advantage Methods
   │                                      │
   └──── Flexible & parallelizable        └──── Sample efficient
                               │
                               ▼
                   Policy Gradient Core Idea
       ┌───────────────────────────────────────────────┐
       │ Increase probability of good actions          │
       │ Decrease probability of poor actions          │
       │ Gradient term: ∇θ log πθ(a|s)(score function) │
       └───────────────────────────────────────────────┘
                               │
                               ▼
                     REINFORCE Algorithm (MC)
       ┌───────────────────────────────────────────────┐
       │ Sample full episodes (Monte Carlo)            │
       │ Compute return Gt at each time step           │
       │ Update: θ ← θ + α ∇θ log πθ(a_t|s_t) * G_t    │
       └───────────────────────────────────────────────┘
                               │
                               ▼
                      Policy Parameterization
       ┌────────────────────────────┬────────────────────────────┐
       │ Softmax Policy (Discrete)  │ Gaussian Policy(Continuous)│
       │ πθ(a|s) = exp(...)         │ a ~ N(μ(s), σ²)            │
       │ ∇θ log πθ = φ - Eφ         │ ∇θ log πθ = (a-μ)/σ² * φ   │
       └────────────────────────────┴────────────────────────────┘
                               │
                               ▼
                         Final Outcome
       ┌─────────────────────────────────────────────────┐
       │ Learn π* directly (no need for Q or V tables)   │
       │ Works naturally with stochastic & continuous    │
       │ Supports neural network policy parameterization │
       │ Foundation of modern deep RL (PPO, A3C, DDPG)   │
       └─────────────────────────────────────────────────┘
```
