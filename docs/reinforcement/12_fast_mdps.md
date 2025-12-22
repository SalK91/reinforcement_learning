# Chapter 12: Fast Reinforcement Learning in MDPs and Generalization

In previous chapters, we focused on exploration strategies in bandits. This chapter builds on those foundations and explores fast learning in Markov Decision Processes (MDPs). We consider various settings (e.g., tabular MDPs, large state/action spaces), evaluation frameworks (e.g., regret, PAC), and principled exploration approaches (e.g., optimism and probability matching).

- Bandits: Single-step decision-making problems.
- MDPs: Sequential decision-making with transition dynamics.

### Evaluation Frameworks

To assess learning efficiency, we use:

- Regret: Cumulative difference between the rewards of the optimal policy and the agent's policy.
- Bayesian Regret: Expected regret under a prior distribution over MDPs.
- PAC (Probably Approximately Correct): Number of steps when the policy is not $\epsilon$-optimal is bounded with high probability.

### Exploration Approaches

- Optimism under uncertainty (e.g., UCB)
- Probability matching (e.g., Thompson Sampling)


## PAC Framework for MDPs

A reinforcement learning algorithm $A$ is PAC if with probability at least $1 - \delta$, it selects an $\epsilon$-optimal action on all but a bounded number of time steps $N$, where:

$$
N = \text{poly} \left( |S|, |A|, \frac{1}{1 - \gamma}, \frac{1}{\epsilon}, \frac{1}{\delta} \right)
$$

 
## MBIE-EB: Model-Based Interval Estimation with Exploration Bonus

The MBIE-EB algorithm (Model-Based Interval Estimation with Exploration Bonuses) is a principled model-based approach to PAC reinforcement learning. It implements the idea of optimism in the face of uncertainty by constructing an upper confidence bound (UCB) on the action-value function $Q(s, a)$.

Rather than maintaining optimistic value estimates directly, MBIE-EB achieves optimism indirectly by learning optimistic models of both the reward function and transition dynamics. That is:

* It estimates $\hat{R}(s, a)$ and $\hat{T}(s' \mid s, a)$ from data using empirical counts.

* It augments these estimates with confidence bonuses that reflect the uncertainty due to limited experience.

The Q-function is then computed using dynamic programming over these optimistically biased models, which encourages the agent to explore actions and transitions that are less well understood.

In essence, MBIE-EB balances exploitation and exploration by behaving as if the world is more favorable in parts where it has limited data, thereby systematically guiding the agent to reduce its uncertainty over time.



Algorithm:

1: Given $\epsilon$, $\delta$, $m$  
2: $\beta = \dfrac{1}{1-\gamma}\sqrt{0.5 \ln \!\left(\dfrac{2|S||A|m}{\delta}\right)}$  
3: $n_{sas}(s,a,s') = 0$, $\forall s \in S, a \in A, s' \in S$  
4: $rc(s,a) = 0$, $n_{sa}(s,a) = 0$, $\hat{Q}(s,a) = \dfrac{1}{1-\gamma}$, $\forall s \in S, a \in A$  
5: $t = 0$, $s_t = s_{\text{init}}$  
6: loop  
7: $\quad a_t = \arg\max_{a \in A} \hat{Q}(s_t, a)$  
8: $\quad$ Observe reward $r_t$ and state $s_{t+1}$  
9: $\quad n_{sa}(s_t,a_t) = n_{sa}(s_t,a_t) + 1$,  
$\quad\quad n_{sas}(s_t,a_t,s_{t+1}) = n_{sas}(s_t,a_t,s_{t+1}) + 1$  
10: $\quad rc(s_t,a_t) = \dfrac{rc(s_t,a_t)\big(n_{sa}(s_t,a_t)-1\big) + r_t}{n_{sa}(s_t,a_t)}$  
11: $\quad \hat{R}(s_t,a_t) = rc(s_t,a_t)$ and  
$\quad\quad \hat{T}(s' \mid s_t,a_t) = \dfrac{n_{sas}(s_t,a_t,s')}{n_{sa}(s_t,a_t)}$, $\forall s' \in S$  
12: $\quad$ while not converged do  
13: $\quad\quad \hat{Q}(s,a) = \hat{R}(s,a) + \gamma \sum_{s'} \hat{T}(s' \mid s,a)\max_{a'} \hat{Q}(s',a') + \dfrac{\beta}{\sqrt{n_{sa}(s,a)}}$,  
$\quad\quad\quad \forall s \in S, a \in A$  
14: $\quad$ end while  
15: end loop


## Bayesian Model-Based Reinforcement Learning

Bayesian RL methods maintain a posterior over MDP models $(P, R)$ and sample plausible environments from the posterior to plan and act.


Thompson Sampling extends naturally from bandits to MDPs by using probability matching over policies. The idea is to choose actions with a probability equal to the probability that they are optimal under the current posterior distribution over MDPs.

Formally, the Thompson sampling policy is:

$$
\pi(s, a \mid h_t) = \mathbb{P}\left(Q(s, a) \ge Q(s, a'),\; \forall a' \ne a \;\middle|\; h_t \right)
= \mathbb{E}_{\mathcal{P}, \mathcal{R} \mid h_t} \left[ \mathbb{1}\left(a = \arg\max_{a \in \mathcal{A}} Q(s, a)\right) \right]
$$

Where:
- $h_t$ is the history up to time $t$ (including all observed transitions and rewards),
- $\mathcal{P}, \mathcal{R}$ are the transition and reward functions respectively,
- The expectation is taken over the posterior belief on the MDP $(\mathcal{P}, \mathcal{R})$.


### Thompson Sampling Algorithm in MDPs

1. Maintain a posterior $p(\mathcal{P}, \mathcal{R} \mid h_t)$ over the transition and reward models based on all observed data.
2. Sample a model $(\mathcal{P}, \mathcal{R})$ from the posterior distribution.
3. Solve the sampled MDP using any planning algorithm (e.g., Value Iteration, Policy Iteration) to obtain the optimal Q-function $Q^*(s, a)$.
4. Select the action according to the optimal action in the sampled model:
   $$
   a_t = \arg\max_{a \in \mathcal{A}} Q^*(s_t, a)
   $$

### Algorithm: Thompson Sampling for MDPs

1: Initialize prior over dynamics and reward models for each $(s, a)$:  $\quad p(\mathcal{T}(s' \mid s, a)), \quad p(\mathcal{R}(s, a))$  
2: Initialize initial state $s_0$  
3: for $k = 1$ to $K$ episodes do  
4: $\quad$ Sample an MDP $\mathcal{M}$:  
5: $\quad\quad$ for each $(s, a)$ pair do  
6: $\quad\quad\quad$ Sample transition model $\mathcal{T}(s' \mid s, a)$ from posterior  
7: $\quad\quad\quad$ Sample reward model $\mathcal{R}(s, a)$ from posterior  
8: $\quad\quad$ end for  
9: $\quad$ Compute optimal value function $Q_{\mathcal{M}}^*$ for sampled MDP $\mathcal{M}$  
10: $\quad$ for $t = 1$ to $H$ do  
11: $\quad\quad a_t = \arg\max_{a \in \mathcal{A}} Q_{\mathcal{M}}^*(s_t, a)$  
12: $\quad\quad$ Take action $a_t$, observe reward $r_t$ and next state $s_{t+1}$  
13: $\quad$ end for  
14: $\quad$ Update posteriors: $\quad\quad p(\mathcal{R}_{s_t, a_t} \mid r_t), \quad p(\mathcal{T}(s' \mid s_t, a_t) \mid s_{t+1})$ using Bayes Rule  
15: end for

## Key Characteristics

- Exploration via Sampling: Exploration arises implicitly by occasionally sampling optimistic MDPs where uncertain actions appear optimal.
- Posterior-Driven Behavior: As more data is collected, the posterior concentrates, leading to increasingly greedy behavior.
- Bayesian Approach: Incorporates prior knowledge and uncertainty in a principled way.

> Thompson Sampling combines Bayesian inference with planning and offers a natural extension of bandit-style exploration to full reinforcement learning.


## Generalization in Contextual Bandits

Contextual bandits generalize standard bandits by associating a context or state $s$ with each decision:

- Reward depends on both context and action: $r \sim P[r | s,a]$
- Often model reward as linear: $r = \theta^\top \phi(s,a) + \epsilon$, with $\epsilon \sim \mathcal{N}(0, \sigma^2)$

### Benefits of Generalization

- Allows learning across states/actions
- Enables sample-efficient exploration in large state/action spaces

 
## Strategic Exploration in Deep RL

For high-dimensional domains, tabular methods fail. We must combine exploration with generalization.

### Optimistic Q-Learning with Function Approximation

Modified Q-learning update:

$$
\Delta w = \alpha \left( r + r_{\text{bonus}}(s,a) + \gamma \max_{a'} Q(s', a'; w) - Q(s,a;w) \right) \nabla_w Q(s,a;w)
$$

Bonus $r_{\text{bonus}}$ reflects novelty or epistemic uncertainty.

### Count-Based and Density-Based Exploration

- Bellemare et al. (2016) use pseudo-counts derived from density models.
- Ostrovski et al. (2017) leverage pixel-CNNs for density estimation.
- Tang et al. (2017) use hashing-based counts.

 
## Thompson Sampling for Deep RL

Applying Thompson sampling in deep RL is challenging due to the intractability of posterior distributions.

### Bootstrapped DQN

- Train multiple Q-networks on bootstrapped datasets.
- Select one head randomly at each episode for exploration.

### Bayesian Deep Q-Networks

- Bayesian linear regression on final layer
- Posterior used to sample Q-values, enabling optimism
- Outperforms naive bootstrapped DQNs in some settings
 


## Mental Map
             Fast Reinforcement Learning in MDPs & Generalization
      Goal: Learn near-optimal policies in MDPs with limited data
        Extend bandit exploration ideas to sequential decision making
                                │
                                ▼
                 Why MDPs Are Harder Than Bandits
 ┌─────────────────────────────────────────────────────────────┐
 │ MDPs involve sequential decisions with transitions           │
 │ Agent must explore over states and transitions              │
 │ Exploration affects future knowledge & rewards              │
 │ Sample inefficiency is a major practical bottleneck         │
 └─────────────────────────────────────────────────────────────┘
                                │
                                ▼
                   Evaluation Frameworks for RL
 ┌─────────────────────────────────────────────────────────────┐
 │ Regret: cumulative gap vs optimal policy over time          │
 │ PAC (Probably Approximately Correct):                       │
 │   Guarantees ε-optimality with high probability             │
 │ Bayesian Regret: expected regret under prior over MDPs      │
 └─────────────────────────────────────────────────────────────┘
                                │
                                ▼
              PAC Learning in MDPs: Formal Guarantee
 ┌─────────────────────────────────────────────────────────────┐
 │ Algorithm is PAC if all but N steps are ε-optimal           │
 │ N = poly(|S|, |A|, 1/(1-γ), 1/ε, 1/δ)                        │
 │ Ensures high-probability performance bounds                 │
 └─────────────────────────────────────────────────────────────┘
                                │
                                ▼
               Optimism: MBIE-EB Algorithm (Model-Based)
 ┌─────────────────────────────────────────────────────────────┐
 │ Estimate reward + transitions from data                     │
 │ Add bonus to Q-values: encourages actions with high uncertainty │
 │ Optimistic model induces exploration                        │
 │ Dynamic programming over Q̂ + bonus → exploration policy     │
 └─────────────────────────────────────────────────────────────┘
                                │
                                ▼
           Algorithmic Principle: Optimism Under Uncertainty
 ┌─────────────────────────────────────────────────────────────┐
 │ Add uncertainty-driven bonus to reward or Q-value           │
 │ Drives exploration to unknown regions                       │
 │ Simple but effective in tabular MDPs                        │
 └─────────────────────────────────────────────────────────────┘
                                │
                                ▼
                 Bayesian RL and Posterior Sampling
 ┌─────────────────────────────────────────────────────────────┐
 │ Maintain belief (posterior) over MDP model (P, R)           │
 │ Sample MDP from posterior → plan optimally in sampled MDP   │
 │ Leads to probability matching via Thompson Sampling         │
 │ Posterior concentrates with data → convergence to optimal   │
 └─────────────────────────────────────────────────────────────┘
                                │
                                ▼
          Algorithm: Thompson Sampling in Model-Based RL
 ┌─────────────────────────────────────────────────────────────┐
 │ Sample dynamics + rewards from posterior                    │
 │ Solve sampled MDP for optimal Q*                            │
 │ Act according to Q* in sample MDP                           │
 │ Update posterior using Bayes rule after each step           │
 └─────────────────────────────────────────────────────────────┘
                                │
                                ▼
             Exploration via Posterior Variance (Bayes)
 ┌─────────────────────────────────────────────────────────────┐
 │ Thompson Sampling ≈ Probability Matching                    │
 │ Probabilistically favors optimal but uncertain policies     │
 │ Elegant & adaptive exploration                             │
 └─────────────────────────────────────────────────────────────┘
                                │
                                ▼
              Generalization via Contextual Bandits
 ┌─────────────────────────────────────────────────────────────┐
 │ Rewards depend on both context and action                   │
 │ Learn generalizable function: Q(s,a) or π(a|s)              │
 │ Enables learning across states / actions                    │
 │ Use linear models or embeddings: φ(s,a)                     │
 └─────────────────────────────────────────────────────────────┘
                                │
                                ▼
         Exploration + Generalization in Deep RL Settings
 ┌─────────────────────────────────────────────────────────────┐
 │ Optimistic Q-learning: add r_bonus(s,a) in TD target        │
 │ r_bonus from novelty, density models, or uncertainty        │
 │ Count-based, hashing, or learned density bonuses            │
 └─────────────────────────────────────────────────────────────┘
                                │
                                ▼
                 Bayesian Deep RL: Posterior Approximation
 ┌─────────────────────────────────────────────────────────────┐
 │ Bootstrapped DQN: ensemble of Q-networks for exploration    │
 │ Bayesian DQN: sample from approximate Q-posteriors          │
 │ Enables implicit Thompson-like behavior                     │
 │ Scales to high-dimensional state/action spaces              │
 └─────────────────────────────────────────────────────────────┘
                                │
                                ▼
                         Chapter Summary
 ┌─────────────────────────────────────────────────────────────┐
 │ Strategic exploration = key to fast learning in MDPs        │
 │ Optimism (MBIE-EB) and Bayesian methods (Thompson)          │
 │ PAC and Bayesian regret are key evaluation tools            │
 │ Generalization (via features or deep nets) enables scaling  │
 │ Thompson Sampling and bootstrapped approximations bridge gap│
 │ Between tabular and high-dimensional RL                     │
 └─────────────────────────────────────────────────────────────┘
````