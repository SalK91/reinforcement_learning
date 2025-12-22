# Chapter 11: Data-Efficient Reinforcement Learning — Bandit Foundations


In real-world applications of Reinforcement Learning (RL), data is expensive, time-consuming, or risky to collect. This necessitates data-efficient RL: designing agents that learn effectively from limited interaction. Bandits provide a foundational setting to study such principles. In this chapter, we explore multi-armed banditsas the prototypical framework for understanding the exploration-exploitation tradeoff, and examine several algorithmic approaches and regret-based evaluation criteria.


## The Multi-Armed Bandit Model

A multi-armed bandit is defined as a tuple $(\mathcal{A}, \mathcal{R})$, where:

- $\mathcal{A} = \{a_1, \dots, a_m\}$ is a known, finite set of actions (arms),
- $R_a(r) = \mathbb{P}[r \mid a]$ is an unknown probability distribution over rewards for each action.
- there is no "state".

At each timestep $t$, the agent:

1. Chooses an action $a_t \in \mathcal{A}$,
2. Receives a stochastic reward $r_t \sim R_{a_t}$.

Goal: Maximize cumulative reward:  
$$
\sum_{t=1}^{T} r_t
$$

This simple model embodies the core RL challenges—particularly exploration vs. exploitation—in an isolated setting.


### Evaluating Algorithms: Regret Framework

Regret: 

- $Q(a) = \mathbb{E}[r \mid a]$ be the expected reward for action $a$,
- $a^* = \arg\max_{a \in \mathcal{A}} Q(a)$,
- Optimal Value $V^* = Q(a^*)$

Then regret is the opportunity loss for one step:
$$
\ell_t = \mathbb{E}[V^* - Q(a_t)]
$$

Total Regret is the total opportunity loss: Total regret over $T$ timesteps

$$
L_T = \sum_{t=1}^T \ell_t = \sum_{a \in \mathcal{A}} \mathbb{E}[N_T(a)] \cdot \Delta_a
$$
Where:

- $N_T(a)$: Number of times arm $a$ is selected by time $T$,
- $\Delta_a = V^* - Q(a)$: Suboptimality gap.


> Maximize cumulative reward <=> minimize total regret





## Baseline Approaches and Their Regret

### Greedy Algorithm

$$
\hat{Q}_t(a) = \frac{1}{N_t(a)} \sum_{\tau=1}^t r_\tau \cdot \mathbb{1}(a_\tau = a)
$$

$$
a_t = \arg\max_{a \in \mathcal{A}} \hat{Q}_t(a)
$$

#### Key Insight:

- Exploits current estimates.
- May lock onto suboptimal arms due to early bad luck.
- Linear regret in expectation.

### Example:
If $Q(a_1) = 0.95, Q(a_2) = 0.90, Q(a_3) = 0.1$, and the first sample of $a_1$ yields 0, the greedy agent may ignore it indefinitely.

### $\varepsilon$-Greedy Algorithm

At each timestep:

- With probability $1 - \varepsilon$: exploit ($\arg\max \hat{Q}_t(a)$),
- With probability $\varepsilon$: explore uniformly at random.

#### Performance:
- Guarantees exploration.
- Linear regret unless $\varepsilon$ decays over time.

### Decaying $\varepsilon$-Greedy

Allows $\varepsilon_t \to 0$ as $t \to \infty$, enabling convergence.


## Optimism in the Face of Uncertainty
Prefer actions with uncertain but potentially high value:

Why? Two possible outcomes:

1. Getting a high reward:    If the arm really has a high mean reward.

2. Learning something : If the arm really has a lower mean reward, pulling it will (in expectation) reduce its average reward estimate and the uncertainty over its value.


Algorithm: 

* Estimate an upper confidence bound $U_t(a)$ for each action value, such that   $Q(a) \le U_t(a)$ with high probability.

* This depends on the number of times $N_t(a)$ action $a$ has been selected.

* Select the action maximizing the Upper Confidence Bound (UCB):

$$a_t = \arg\max_{a \in \mathcal{A}} \left[ U_t(a) \right]$$


> Hoeffding Bound Justification:  Given i.i.d. bounded rewards $X_i \in [0,1]$,
> $$
> \mathbb{P}\!\left[ \mathbb{E}[X] > \bar{X}_n + u \right]
> \;\le\; \exp(-2 n u^2).
> $$
>
> Setting the right-hand side equal to $\delta$ and solving for $u$,
> $$
> u = \sqrt{\frac{\log(1/\delta)}{2n}}.
> $$
> Here, $\delta$ is the failure probability, and the confidence interval
> holds with probability at least $1 - \delta$.
> This means that, with probability at least $1 - \delta$,
> $$
> \bar{X}_n - u \;\le\; \mathbb{E}[X] \;\le\; \bar{X}_n + u.
> $$


$$
a_t = \arg\max_{a \in \mathcal{A}} \left[ \hat{Q}_t(a) + \text{UCB}_t(a) \right]
$$


### UCB1 Algorithm

$$
\text{UCB}_t(a) = \hat{Q}_t(a) + \sqrt{\frac{2 \log \frac{1}{\delta} }{N_t(a)}}
$$

- where $\hat{Q}_t(a)$ is empirical average
- $N_t(a)$ is number of samples of $a$ after $t$ timesteps.
- Provable sublinear regret.
- Balances estimated value and exploration bonus.

Algorithm: UCB1 (Auer, Cesa-Bianchi, Fischer, 2002)

1: Initialize for each arm $a \in \mathcal{A}$:  $\quad N(a) \leftarrow 0,\;\; \hat{Q}(a) \leftarrow 0$
2: Warm start (sample each arm once):  
3: for each arm $a \in \mathcal{A}$ do  
4: $\quad$ Pull arm $a$, observe reward $r \in [0,1]$  
5: $\quad N(a) \leftarrow 1$  
6: $\quad \hat{Q}(a) \leftarrow r$  
7: end for  
8: Set $t \leftarrow |\mathcal{A}|$

9: for $t = |\mathcal{A}|+1, |\mathcal{A}|+2, \dots$ do  
10: $\quad$ Compute UCB for each arm: $\quad \mathrm{UCB}_t(a) = \hat{Q}(a) + \sqrt{\frac{2\log t}{N(a)}}$

11: $\quad$ Select action:$\quad a_t \leftarrow \arg\max_{a \in \mathcal{A}} \mathrm{UCB}_t(a)$

12: $\quad$ Pull arm $a_t$, observe reward $r_t$

13: $\quad$ Update count: $\quad N(a_t) \leftarrow N(a_t) + 1$

14: $\quad$ Update empirical mean (incremental):  
$$
\quad \hat{Q}(a_t) \leftarrow \hat{Q}(a_t) + \frac{1}{N(a_t)}\Big(r_t - \hat{Q}(a_t)\Big)
$$

15: end for


## 11.9 Optimistic Initialization in Greedy Bandit Algorithms

One of the simplest yet powerful strategies for promoting exploration in bandit algorithms is optimistic initialization. This method enhances a greedy policy with a strong initial incentive to explore, simply by setting the initial action-value estimates to unrealistically high values.

### Motivation

Greedy algorithms, by default, select actions with the highest estimated value:

$$
a_t = \arg\max_a \hat{Q}_t(a)
$$

If these $\hat{Q}_t(a)$ estimates start at zero (or some neutral value), the agent may never try better actions if initial random outcomes favor suboptimal arms. Optimistic initialization addresses this by initializing all action values with high values, thereby making unexplored actions look promising until proven otherwise.


### Algorithmic Details

We initialize:

- $\hat{Q}_0(a) = Q_{\text{init}}$ for all $a \in \mathcal{A}$, where $Q_{\text{init}}$ is set higher than any reasonable expected reward (e.g., $Q_{\text{init}} = 1$ if rewards are bounded in $[0, 1]$).
- $N(a) = 1$ to ensure initial update is well-defined.

Then we update action values using an incremental Monte Carlo estimate:

$$
\hat{Q}_{t}(a_t) = \hat{Q}_{t-1}(a_t) + \frac{1}{N_t(a_t)} \left( r_t - \hat{Q}_{t-1}(a_t) \right)
$$

This update encourages each arm to be pulled at least once, because its high initial estimate makes it look appealing.

- Encourages systematic early exploration: Untried actions appear promising and are thus selected.
- Simple to implement: No need for tuning $\varepsilon$ or computing uncertainty estimates.
- Can still lock onto suboptimal arms if the initial values are not optimistic enough.

#### Key Design Considerations

- How optimistic is optimistic enough?  
  If $Q_{\text{init}}$ is not much larger than the true values, the agent may not explore effectively.
- What if $Q_{\text{init}}$ is too high?  
  Overly optimistic values may lead to long periods of exploring clearly suboptimal actions, slowing down learning.

#### Function Approximation

Optimistic initialization is non-trivial under function approximation (e.g., with neural networks). With global function approximators, setting optimistic values for one state-action pair may affect others due to shared parameters, making it harder to ensure controlled optimism.

## 11.10 Theoretical Frameworks: Regret and PAC

### Regret-Based Evaluation

As discussed earlier, regret captures the cumulative shortfall from not always acting optimally. Total regret may arise from:

- Many small mistakes (frequent near-optimal actions),
- A few large mistakes (infrequent but very suboptimal actions).

Minimizing regret growth with $T$ is the dominant criterion in theoretical analysis of bandit and RL algorithms.


### Probably Approximately Correct (PAC) Framework

PAC-style analysis seeks stronger, step-wise performance guarantees, rather than just bounding cumulative regret.

An algorithm is $(\varepsilon, \delta)$-PAC if, on each time step $t$, it chooses an action $a_t$ such that:

$$
Q(a_t) \ge Q(a^*) - \varepsilon \quad \text{with probability at least } 1 - \delta
$$

on all but a polynomial number of time steps (in $|\mathcal{A}|$, $1/\varepsilon$, $1/\delta$, etc). This ensures:

- The agent almost always behaves nearly optimally,
- With high probability, after a reasonable amount of time.

PAC is a natural framework when you care about individual-time-step performance rather than only cumulative regret.


## Comparing Exploration Strategies

| Strategy                | Regret Behavior     | Notes |
|-------------------------|---------------------|-------|
| Greedy                  | Linear              | No exploration mechanism |
| Constant $\varepsilon$-greedy | Linear              | Fixed chance of exploring |
| Decaying $\varepsilon$-greedy | Sublinear (if tuned) | Requires prior knowledge of reward gaps |
| Optimistic Initialization | Sublinear (if optimistic enough) | Simple, effective in tabular settings |

Bottom Line: Optimistic initialization is a computationally simple strategy to induce exploration, but its effectiveness depends crucially on how optimistic the initialization is. In function approximation settings, more principled strategies like UCB or Thompson Sampling may scale better and provide stronger guarantees.

## Bayesian Bandits

So far, our treatment of bandits has made no assumptions about the underlying reward distributions, aside from basic bounds (e.g., rewards in $[0,1]$). Bayesian bandits offer a powerful alternative by leveraging prior knowledge about the reward-generating process, and updating our beliefs as data is observed.


### Key Idea: Maintain Beliefs Over Arm Reward Distributions

In the Bayesian framework, we treat the reward distribution for each arm as governed by an unknown parameter $\\phi_i$ for arm $i$. Instead of maintaining a point estimate (e.g., average reward), we maintain a distribution over possible values of $\\phi_i$, representing our uncertainty.

#### Prior and Posterior

- Prior: Our initial belief about $\\phi_i$ is encoded in a probability distribution $p(\\phi_i)$.
- Data: After pulling arm $i$ and observing reward $r_{i1}$, we update our belief.
- Posterior: The new belief is computed using Bayes' rule:

$$p(\phi_i \mid r_{i1}) =
\frac{
p(r_{i1} \mid \phi_i)\, p(\phi_i)
}{
p(r_{i1})
}
=
\frac{
p(r_{i1} \mid \phi_i)\, p(\phi_i)
}{
\int p(r_{i1} \mid \phi_i)\, p(\phi_i)\, d\phi_i
}$$

This posterior becomes the new prior for future updates as more data arrives.



### Practical Considerations
Computing the posterior $p(\phi_i \mid D)$ (where $D$ is the observed data for arm $i$) can be analytically intractable in many cases. However, tractability improves significantly if we use:

- Conjugate priors: If the prior and likelihood combine to yield a posterior in the same family as the prior.
- Many common bandit models use exponential family distributions, which have well-known conjugate priors (e.g., Beta prior for Bernoulli rewards).


### Why Use Bayesian Bandits?

- Instead of upper-confidence bounds (as in UCB), Bayesian bandits reason directly about uncertainty via posterior distributions.
- The agent chooses actions based on sampling from or optimizing over the posterior (as in Thompson Sampling).
- Captures uncertainty in a principled and statistically coherent manner.


### Summary

- Bayesian bandits treat the reward-generating parameters $\phi_i$ as random variables.
- We maintain a posterior belief $p(\phi_i \mid D)$ using Bayes' rule.
- When conjugate priors are used, analytical updates are possible.
- This leads to more informed exploration strategies based on posterior uncertainty rather than hand-designed confidence bounds.


### Thompson Sampling:
Thompson Sampling is a principled Bayesian algorithm for balancing exploration and exploitation in bandit problems. It maintains a posterior distribution over the expected reward of each arm and samples from these distributions to make decisions. By sampling, it naturally explores arms with higher uncertainty while favoring those with higher expected rewards, embodying an elegant form of probabilistic optimism.

This approach is also known as *probability matching*: at each time step, the agent selects each arm with probability equal to the chance that it is the optimal arm, according to the current posterior. Unlike greedy methods, Thompson Sampling doesn’t deterministically select the arm with the highest mean—it selects arms in proportion to their likelihood of being best, leading to efficient exploration in uncertain settings.







Algorithm: Thompson Sampling:

1: Initialize prior over each arm $a$, $p(\mathcal{R}_a)$  
2: for iteration $= 1, 2, \dots$ do  
3: $\quad$ For each arm $a$ sample a reward distribution $\mathcal{R}_a$ from posterior  
4: $\quad$ Compute action-value function $Q(a) = \mathbb{E}[\mathcal{R}_a]$  
5: $\quad a_t \equiv \arg\max_{a \in \mathcal{A}} Q(a)$  
6: $\quad$ Observe reward $r$  
7: $\quad$ Update posterior $p(\mathcal{R}_a)$ using Bayes Rule  
8: end for  

### Contextual Bandits

The contextual bandit problem extends the standard multi-armed bandit framework by incorporating side information or context. At each time step, before choosing an action, the agent observes a context $x_t$ drawn i.i.d. from some unknown distribution. The expected reward of each arm depends on this observed context.

In this setting, the goal is to learn a context-dependent policy $\pi(a \mid x)$ that maps the observed context $x_t$ to a suitable arm $a_t$, maximizing expected reward. Unlike the vanilla bandit setting, where each arm has a fixed reward distribution, here the rewards vary as a function of the context. This makes the problem more expressive and applicable to real-world decision-making scenarios, such as personalized recommendations, ad placement, or clinical treatment selection.

Formally, the interaction at each time step $t$ is:

1. Observe context $x_t \in \mathcal{X}$
2. Choose action $a_t \in \mathcal{A}$ based on policy $\pi(a \mid x_t)$
3. Receive reward $r_t(a_t, x_t)$

Over time, the algorithm must learn to choose actions that maximize expected reward conditioned on context, i.e.,

$$
\pi^*(x) = \arg\max_{a \in \mathcal{A}} \mathbb{E}[r(a, x)]
$$

This setting balances exploration across both actions and contexts, and introduces rich generalization capabilities by leveraging contextual information to predict the value of unseen actions in new situations.


## Mental Map

```` text
                Bandits: Foundations of Data-Efficient RL
     Goal: Understand exploration-exploitation in simplest setting
           Learn to act with minimal data through principled tradeoffs
                                │
                                ▼
               What Are Multi-Armed Bandits (MAB)?
 ┌─────────────────────────────────────────────────────────────┐
 │ Single-state (stateless) decision problems                  │
 │ Fixed set of actions (arms)                                 │
 │ Unknown reward distribution per arm                         │
 │ Choose an action, receive reward, repeat                    │
 │ No transition dynamics — unlike full RL                     │
 └─────────────────────────────────────────────────────────────┘
                                │
                                ▼
                Core Objective: Maximize Reward
 ┌─────────────────────────────────────────────────────────────┐
 │ Maximize total reward = minimize regret                     │
 │ Regret = missed opportunity vs optimal action               │
 │ Total regret used to evaluate algorithm efficiency          │
 └─────────────────────────────────────────────────────────────┘
                                │
                                ▼
                  Basic Bandit Algorithms
 ┌─────────────────────────────────────────────────────────────┐
 │ Greedy: exploit current best estimates (linear regret)      │
 │ ε-Greedy: random exploration with fixed ε                   │
 │ Decaying ε-Greedy: reduces ε over time                      │
 │ Optimistic Initialization: set high initial Q̂ values        │
 └─────────────────────────────────────────────────────────────┘
                                │
                                ▼
           Principle: Optimism in the Face of Uncertainty
 ┌─────────────────────────────────────────────────────────────┐
 │ Treat unvisited arms as potentially good                    │
 │ Upper Confidence Bound (UCB) algorithms                     │
 │ Tradeoff: mean reward + exploration bonus                   │
 │ Guarantees sublinear regret                                 │
 └─────────────────────────────────────────────────────────────┘
                                │
                                ▼
                 Algorithmic Realization: UCB1
 ┌─────────────────────────────────────────────────────────────┐
 │ UCB_t(a) = Q̂_t(a) + √(2 log t / N_t(a))                     │
 │ Encourages pulling uncertain arms early                     │
 │ Regret ≈ O(√(T log T))                                      │
 │ Theoretically grounded and simple to implement              │
 └─────────────────────────────────────────────────────────────┘
                                │
                                ▼
           Theoretical Frameworks: Regret vs PAC
 ┌─────────────────────────────────────────────────────────────┐
 │ Regret: cumulative gap from always acting optimally         │
 │ PAC: guarantees near-optimal behavior with high probability │
 │ Regret cares about sum of mistakes; PAC focuses on steps    │
 │ Both evaluate quality and efficiency of learning            │
 └─────────────────────────────────────────────────────────────┘
                                │
                                ▼
                Bayesian Bandits and Uncertainty
 ┌─────────────────────────────────────────────────────────────┐
 │ Treat arm rewards as random variables                       │
 │ Use prior + observed data → posterior via Bayes rule        │
 │ Conjugate priors simplify computation                       │
 │ Enable principled uncertainty reasoning                     │
 └─────────────────────────────────────────────────────────────┘
                                │
                                ▼
                   Thompson Sampling (Bayesian)
 ┌─────────────────────────────────────────────────────────────┐
 │ Sample reward distribution from posterior per arm           │
 │ Pull arm with highest sampled reward                        │
 │ Probabilistic optimism: match probability of being best     │
 │ Natural exploration and strong empirical performance        │
 └─────────────────────────────────────────────────────────────┘
                                │
                                ▼
                 Probability Matching Perspective
 ┌─────────────────────────────────────────────────────────────┐
 │ Thompson Sampling ≈ sample optimal arm w/ correct frequency │
 │ Avoids hard-coded uncertainty bonuses                       │
 │ Simpler and often better in practice                        │
 └─────────────────────────────────────────────────────────────┘
                                │
                                ▼
                       Contextual Bandits
 ┌─────────────────────────────────────────────────────────────┐
 │ Input context x_t at each timestep                          │
 │ Reward distribution depends on (action, context)            │
 │ Learn policy π(a | x): context-aware decision making        │
 │ Real-world applications: ads, medicine, personalization     │
 └─────────────────────────────────────────────────────────────┘
                                │
                                ▼
          Summary: Bandits as Foundation for Efficient RL
 ┌─────────────────────────────────────────────────────────────┐
 │ Bandits isolate the exploration-exploitation tradeoff       │
 │ Simpler than full RL, but deeply insightful                 │
 │ Concepts generalize to value estimation, uncertainty        │
 │ Key tools: regret, PAC bounds, posterior reasoning          │
 └─────────────────────────────────────────────────────────────┘
````