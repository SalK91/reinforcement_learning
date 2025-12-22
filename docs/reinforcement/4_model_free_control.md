# Chapter 4: Model-Free Control: Learning Optimal Behavior Without a Model

In Chapter 3, we learned how to estimate the value of a fixed policy using Monte Carlo and Temporal Difference methods, but we did not address how to improve that policy. The goal of Model-Free Control is to discover the optimal policy $\pi^*$ without knowing the transition probabilities or reward function. To achieve this, we must learn not only to evaluate a policy, but also to improve it through interaction with the environment.


## From State Values to Action Values

In model-based methods like Dynamic Programming, policy improvement depends on knowing the environment model. To improve a policy, we use the Bellman optimality equation:

$$
\pi_{k+1}(s) = \arg\max_a \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a)V^{\pi_k}(s') \right]
$$
This update requires two things:

 - the transition probabilities $P(s'|s,a)$
 - the expected reward R(s,a)$

If either of these is unknown, we cannot compute the right-hand side, so model-based policy improvement becomes impossible.

Instead of learning the state-value function $V^\pi(s)$ and using the model to evaluate the effect of each action, model-free RL learns the value of actions themselves.
$$
Q^\pi(s,a) = \mathbb{E}_\pi[G_t \mid s_t = s, a_t = a]
$$

The Model-Free Policy Iteration loop:

1.  Policy Evaluation: Compute $Q^{\pi}$ from experience.
2.  Policy Improvement: Update the policy $\pi$ given the estimated $Q^{\pi}$.

However, using a purely greedy policy creates a new problem: the agent will only experience actions it already believes are good, and may never discover better ones. This introduces the fundamental challenge of exploration.

## Exploration vs. Exploitation

To learn optimal behavior, the agent must balance two goals:

1. Exploitation: choose actions believed to yield high rewards.
2. Exploration: try actions whose consequences are uncertain or poorly understood.

A common solution is the $\epsilon$-greedy policy:

With probability $1 - \epsilon$, choose the action with the highest estimated value.  
With probability $\epsilon$, choose a random action.

Formally:

$$
\pi(a|s) =
\begin{cases}
1 - \epsilon + \frac{\epsilon}{|A|} & \text{if } a = \arg\max_{a'} Q(s,a') \\
\frac{\epsilon}{|A|} & \text{otherwise}
\end{cases}
$$

This approach ensures that the agent both explores and exploits, learning from a wide range of actions while gradually improving its policy.


## Monte Carlo Control

Monte Carlo Control extends the Monte Carlo methods from Chapter 3 to action-value learning. Instead of estimating $V(s)$, it estimates $Q(s,a)$ using sampled returns.
    
 Monte Carlo Policy Evaluation, Now for Q:        
1: Initialize $Q(s,a)=0$, $N(s,a)=0$  $\forall(s,a)$,  $k=1$,  Input $\epsilon=1$, $\pi$  
2: loop over epiosdes       
3: $\quad$ Sample k-th episode $(s_{k,1}, a_{k,1}, r_{k,1}, s_{k,2}, \dots, s_{k,T})$ given $\pi$  
4: $\quad$ Compute $G_{k,t} = r_{k,t} + \gamma r_{k,t+1} + \gamma^2 r_{k,t+2} + \dots + \gamma^{T-t-1} r_{k,T}$  $\forall t$  
5: $\quad$   for $t = 1, \dots, T$ do  
6: $\quad\quad$      if First visit to $(s,a)$ in episode $k$ then  
7: $\quad\quad\quad$          $N(s,a) = N(s,a) + 1$  
8: $\quad\quad\quad$           $Q(s_t,a_t) = Q(s_t,a_t) + \dfrac{1}{N(s,a)}(G_{k,t} - Q(s_t,a_t))$  
9: $\quad\quad$       end if  
10: $\quad$  end for  
11: $\quad$  $k = k + 1$  
12: end loop

The simplest approach is On-Policy MC Control (also known as MC Exploring Starts), which follows the generalized policy iteration structure using $\epsilon$-greedy policies for exploration.

* Policy Evaluation: $Q(s, a)$ is updated using the full return ($G_t$) observed after the state-action pair $(s_t, a_t)$ has occurred in an episode. The incremental update uses the formula $Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \frac{1}{N(s,a)}(G_{t} - Q(s_t, a_t))$. 
* Policy Improvement: The new policy $\pi_{k+1}$ is set to be $\epsilon$-greedy with respect to the updated $Q$ function.




### Greedy in the Limit of Infinite Exploration (GLIE)

For Monte Carlo Control to converge to the optimal action-value function $Q^*(s, a)$, the process must satisfy the Greedy in the Limit of Infinite Exploration (GLIE) conditions:

1.  Infinite Visits: All state-action pairs $(s, a)$ must be visited an infinite number of times ($\lim_{i \rightarrow \infty} N_i(s, a) \rightarrow \infty$).
2.  Converging Greed: The behavior policy (the policy used to act and generate data) must eventually converge to a greedy policy.

A simple strategy to satisfy GLIE is to use an $\epsilon$-greedy policy where $\epsilon$ is decayed over time, such as $\epsilon_i = 1/i$ (where $i$ is the episode number). Under the GLIE conditions, Monte-Carlo control converges to the optimal state-action value function $Q^*(s, a)$.

 Monte Carlo Online Control/On Policy Improvement:    

1: Initialize $Q(s,a)=0$, $N(s,a)=0$  $\forall(s,a)$,  Set $k=1$, $\epsilon=1$.     
2: $\pi_k = \epsilon - greedy (Q)$ // Create initial $\epsilon$ - greedy policy.  
3: loop over epiosdes     
4: $\quad$ Sample k-th episode $(s_{k,1}, a_{k,1}, r_{k,1}, s_{k,2}, \dots, s_{k,T})$ given $\pi$  
5: $\quad$ Compute $G_{k,t} = r_{k,t} + \gamma r_{k,t+1} + \gamma^2 r_{k,t+2} + \dots + \gamma^{T-t-1} r_{k,T}$  $\forall t$  
6: $\quad$   for $t = 1, \dots, T$ do  
7: $\quad\quad$      if First visit to $(s,a)$ in episode $k$ then  
8: $\quad\quad\quad$          $N(s,a) = N(s,a) + 1$  
9: $\quad\quad\quad$           $Q(s_t,a_t) = Q(s_t,a_t) + \dfrac{1}{N(s,a)}(G_{k,t} - Q(s_t,a_t))$  
10: $\quad\quad$       end if   
11: $\quad$  end for    
12:  $\quad$ $\pi_k = \epsilon - greedy (Q)$ //Policy improvement  
12: $\quad$  $k = k + 1$ , $\epsilon = \frac{1}{k}$     
13: end loop    

This process gradually adjusts the policy and the value estimates until they converge.

## IV. Temporal Difference (TD) Control 

TD control methods improve upon Monte Carlo control by updating action-value estimates after every step rather than at the end of an episode. They are more data-efficient and work in both episodic and continuing tasks.


### On-Policy TD Control: SARSA

SARSA is an on-policy TD control algorithm. It learns the value of the policy *currently being followed* ($\pi$). Its name is derived from the sequence of steps used in its update rule: State, Action, Reward, State, Action.

The update for the action-value $Q(s_t, a_t)$ uses the value of the *next* state-action pair, $(s_{t+1}, a_{t+1})$, selected by the current policy $\pi$.

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t) \right]
$$

The TD Target here is $r_{t+1} + \gamma Q(s_{t+1}, a_{t+1})$. SARSA learns $Q^{\pi}$ while $\pi$ is improved greedily with respect to $Q^{\pi}$, allowing it to find the optimal policy $\pi^*$.

1: Set initial $\epsilon$-greedy policy $\pi$ randomly, $t=0$, initial state $s_t=s_0$      
2: Take $a_t \sim \pi(s_t)$        
3: Observe $(r_t, s_{t+1})$         
4: loop         
5: $\quad$ Take action $a_{t+1} \sim \pi(s_{t+1})$ // Sample action from policy          
6: $\quad$ Observe $(r_{t+1}, s_{t+2})$     
7: $\quad$ Update $Q$ given $(s_t, a_t, r_t, s_{t+1}, a_{t+1})$:    $$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t) \right]$$
8: $\quad$ Perform policy improvement: The policy is updated every step, making it more greedy according to new Q-values.

$$\forall s \in S,\;\;
\pi(s) =
\begin{cases}
\arg\max\limits_a Q(s,a) & \text{with probability } 1 - \epsilon \\
\text{a random action}   & \text{with probability } \epsilon
\end{cases}$$


9: $\quad$ $t = t + 1$ , $\epsilon = \frac{1}{t}$             
10: end loop        

### B. Off-Policy TD Control: Q-Learning

Q-Learning is the most widely known off-policy TD control algorithm. Off-policy learning means we estimate and evaluate an optimal policy ($\pi^*$, the *target policy*) using experience gathered by a different behavior policy ($\pi_b$).

In Q-Learning, the agent acts using a soft, exploratory $\pi_b$ (like $\epsilon$-greedy) but the value function update is based on the *best* possible action from the next state, effectively estimating $Q^*$.

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t) \right]
$$

The key difference is the target: Q-Learning uses the value of the max action ($\max_{a'} Q(s_{t+1}, a')$), regardless of what action was actually taken in the next step. This makes it a greedy update towards $Q^*$.


Q-Learning (Off-Policy TD Control):

1: Initialize $Q(s,a)=0 \quad \forall s \in S, a \in A$, set $t = 0$, initial state $s_t = s_0$         
2: Set $\pi_b$ to be $\epsilon$-greedy w.r.t. $Q$       
3: loop     
4: $\quad$ Take $a_t \sim \pi_b(s_t)$ // Sample action from behavior policy     
5: $\quad$ Observe $(r_t, s_{t+1})$     
6: $\quad$ $Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha \left[ r_t + \gamma \max\limits_{a} Q(s_{t+1},a) - Q(s_t,a_t) \right]$        
7: $\quad$ $\pi(s_t) =
\begin{cases}
\arg\max\limits_a Q(s_t,a) & \text{with probability } 1 - \epsilon \
\text{a random action} & \text{with probability } \epsilon
\end{cases}$        
8: $\quad$ $t = t + 1$      
9: end loop     


## Value Function Approximation (VFA)

All methods discussed so far assume a tabular representation, where a separate entry for $Q(s, a)$ is stored for every state-action pair. This is only feasible for MDPs with small, discrete state and action spaces.

### Motivation for Approximation

For environments with large or continuous state/action spaces (e.g., in robotics or image-based games like Atari), we face three critical issues:

1.  Memory: Explicitly storing every $V$ or $Q$ value is impossible.
2.  Computation: Computing or updating every value is too slow.
3.  Experience: It would take vast amounts of data to visit and learn every single state-action pair.

Value Function Approximation addresses this by using a parameterized function (like a linear model or a neural network) to estimate the value function: $\hat{Q}(s, a; \mathbf{w}) \approx Q(s, a)$. The goal shifts from filling a table to finding the parameter vector $\mathbf{w}$ that minimizes the error between the true value and the estimate.

$$
J(\mathbf{w}) = \mathbb{E}_{\pi} \left[ \left( Q^{\pi}(s, a) - \hat{Q}(s, a; \mathbf{w}) \right)^2 \right]
$$

The parameter vector $\mathbf{w}$ is typically updated using Stochastic Gradient Descent (SGD), which uses a single sample to approximate the gradient of the loss function $J(\mathbf{w})$.

### Model-Free Control with VFA Policy Evaluation

When using function approximation, we substitute the old $Q(s, a)$ in the update rules (MC, SARSA, Q-Learning) with the function approximator $\hat{Q}(s, a; \mathbf{w})$.

* MC VFA for Policy Evaluation: 

    The return $G_t$ is used as the target in an SGD update: $\Delta \mathbf{w} \propto \alpha (G_t - \hat{Q}(s_t, a_t; \mathbf{w})) \nabla_{\mathbf{w}} \hat{Q}(s_t, a_t; \mathbf{w})$.
    
      
    1: Initialize $\mathbf{w}$, set $k = 1$     
    2: loop     
    3: $\quad$ Sample k-th episode $(s_{k,1}, a_{k,1}, r_{k,1}, s_{k,2}, \dots, s_{k,L_k})$ given $\pi$     
    4: $\quad$ for $t = 1, \dots, L_k$ do       
    5: $\quad\quad$ if First visit to $(s,a)$ in episode $k$ then       
    6: $\quad\quad\quad$ $G_t(s,a) = \sum_{j=t}^{L_k} r_{k,j}$      
    7: $\quad\quad\quad$ $\nabla_{\mathbf{w}} J(\mathbf{w}) = -2 \left[ G_t(s,a) - \hat{Q}(s_t,a_t;\mathbf{w}) \right] \nabla_{\mathbf{w}} \hat{Q}(s_t,a_t;\mathbf{w})$ // Compute Gradient     
    8: $\quad\quad\quad$ Update weights: $\Delta \mathbf{w}$        
    9: $\quad\quad$ end if  
    10: $\quad$ end for 
    11: $\quad$ $k = k + 1$         
    12: end loop    


* SARSA with VFA: The TD target is $r + \gamma \hat{Q}(s', a'; \mathbf{w})$, leveraging the current function approximation.

    1: Initialize $\mathbf{w}$, $s$     
    2: loop     
    3: $\quad$ Given $s$, sample $a \sim \pi(s)$, observe $r(s,a)$, and $s' \sim p(s'|s,a)$     
    4: $\quad$ $\nabla_{\mathbf{w}} J(\mathbf{w}) = -2 [r + \gamma \hat{V}(s';\mathbf{w}) - \hat{V}(s;\mathbf{w})] \nabla_{\mathbf{w}} \hat{V}(s;\mathbf{w})$       
    5: $\quad$ Update weights $\Delta \mathbf{w}$           
    6: $\quad$ if $s'$ is not a terminal state then     
    7: $\quad\quad$ Set $s = s'$        
    8: $\quad$ else         
    9: $\quad\quad$ Restart episode, sample initial state $s$       
    10: $\quad$ end if      
    11: end loop        
* Q-Learning with VFA: The TD target is $r + \gamma \max_{a'} \hat{Q}(s', a'; \mathbf{w})$.

### Control using VFA
So far, we have used function approximation mainly for policy evaluation. However, the true goal of reinforcement learning is control, which means learning policies that maximize expected return. In control, the policy itself is continually improved based on the estimated action-value function. When we replace the tabular $Q(s,a)$ with a function approximator $\hat{Q}(s,a;\mathbf{w})$, we obtain Model-Free Control with Function Approximation, where both learning and acting are driven by $\hat{Q}(s,a;\mathbf{w})$.

Value Function Approximation is especially useful for control because it enables generalization across states, allowing the agent to learn effective behavior even in large or continuous state spaces. Instead of storing separate values for each $(s,a)$, the agent learns a parameter vector $\mathbf{w}$ that works across many states and actions. The objective is to make the approximation close to the true optimal action-value function $Q^*(s,a)$.

The learning problem becomes:

$$
\min_{\mathbf{w}} \; J(\mathbf{w}) = \mathbb{E} \left[ \left( Q^*(s,a) - \hat{Q}(s,a;\mathbf{w}) \right)^2 \right]
$$

Using stochastic gradient descent, we update the weights in the direction that reduces approximation error:

$$
\Delta \mathbf{w} \propto \left( \text{target} - \hat{Q}(s_t,a_t;\mathbf{w}) \right) \nabla_{\mathbf{w}} \hat{Q}(s_t,a_t;\mathbf{w})
$$

The most important difference in control is how we choose the target, which depends on the RL method being used:

| Method | Target for updating $\mathbf{w}$ |
|--------|----------------------------------|
| Monte Carlo | $G_t$ |
| SARSA | $r + \gamma \hat{Q}(s',a';\mathbf{w})$ |
| Q-Learning | $r + \gamma \max_{a'} \hat{Q}(s',a';\mathbf{w})$ |

These methods now operate in the same way as before, except instead of updating a single $Q(s,a)$ entry, we update the weights of the approximator. The update generalizes beyond the visited state, helping the agent learn faster in high-dimensional spaces.

### Challenges: The Deadly Triad

When using function approximation for control, learning can become unstable or even diverge. Instability usually arises when these three components occur together:

$$
\text{Function Approximation} \;+\; \text{Bootstrapping} \;+\; \text{Off-policy Learning}
$$

This combination is known as the Deadly Triad .

-  Function Approximation : Generalizes across states but may introduce bias.
-  Bootstrapping : Uses existing estimates to update current estimates (as in TD methods).
-  Off-policy Learning : Learning from a different behavior policy than the target policy.

Q-Learning with neural networks (as in Deep Q-Learning) contains all three components, making it powerful but potentially unstable without stabilization techniques like  experience replay  and  target networks . Monte Carlo with function approximation is typically more stable because it does not use bootstrapping.

Function approximation enables reinforcement learning to scale to complex environments, but it introduces new challenges in stability and convergence. The next step is to address how these ideas lead to  Deep Q-Learning (DQN) , which successfully applies neural networks to approximate $Q(s,a)$.
 
### Deep Q-Networks (DQN)

The most prominent example of VFA for control is Deep Q-Learning, or Deep Q-Networks (DQN), where the action-value function $\hat{Q}(s, a; \mathbf{w})$ is approximated by a deep neural network. DQN successfully solved control problems directly from raw sensory input (e.g., pixels from Atari games).

DQN stabilizes the non-linear learning process using two critical techniques:

1.  Experience Replay (ER): Transitions $(s_t, a_t, r_t, s_{t+1})$ are stored in a replay buffer ($\mathcal{D}$). Instead of learning from sequential, correlated experiences, the algorithm samples a random mini-batch of past transitions from $\mathcal{D}$ for the update. This breaks correlations, making the data samples closer to i.i.d (independent and identically distributed).

2.  Fixed Q-Targets: The Q-Learning update requires a target value $y_i = r_i + \gamma \max_{a'} \hat{Q}(s_{i+1}, a'; \mathbf{w})$. To prevent the estimate $\hat{Q}(s, a; \mathbf{w})$ from chasing its own rapidly changing target, the parameters $\mathbf{w}^{-}$ used to compute the target are fixed for a period of time, then synchronized with the current parameters $\mathbf{w}$. This provides a stable target $y_i = r_i + \gamma \max_{a'} \hat{Q}(s_{i+1}, a'; \mathbf{w}^{-})$.


Deep Q-Network (DQN) Algorithm:

1: Input $C$, $\alpha$, $D = {}$, Initialize $\mathbf{w}$, $\mathbf{w}^- = \mathbf{w}$, $t = 0$         
2: Get initial state $s_0$              
3: loop         
4: $\quad$ Sample action $a_t$ using $\epsilon$-greedy policy w.r.t. current $\hat{Q}(s_t, a; \mathbf{w})$      
5: $\quad$ Observe reward $r_t$ and next state $s_{t+1}$        
6: $\quad$ Store transition $(s_t, a_t, r_t, s_{t+1})$ in replay buffer $D$     
7: $\quad$ Sample a random minibatch of tuples $(s_i, a_i, r_i, s'i)$ from $D$      
8: $\quad$ for $j$ in minibatch do      
9: $\quad\quad$ if episode terminates at step $i+1$ then        
10: $\quad\quad\quad$ $y_i = r_i$       
11: $\quad\quad$ else       
12: $\quad\quad\quad$ $y_i = r_i + \gamma \max\limits{a'} \hat{Q}(s'i, a'; \mathbf{w}^-)$       
13: $\quad\quad$ end if     
14: $\quad\quad$ Update $\mathbf{w}$ using gradient descent:    
$\quad\quad\quad$ $\Delta \mathbf{w} = \alpha \left( y_i - \hat{Q}(s_i, a_i; \mathbf{w}) \right) \nabla{\mathbf{w}} \hat{Q}(s_i, a_i; \mathbf{w})$      
15: $\quad$ end for         
16: $\quad$ $t = t + 1$     
17: $\quad$ if $t \mod C == 0$ then     
18: $\quad\quad$ $\mathbf{w}^- \leftarrow \mathbf{w}$       
19: $\quad$ end if      
20: end loop        



## Model Free Control Mental Map  

```text
                     Model-Free Control
    Goal: Learn the Optimal Policy π* without knowing P or R
                               │
                               ▼
           Key Concept: Action-Value Function Q(s,a)
       ┌─────────────────────────────────────────────┐
       │Qπ(s,a) = Expected return by taking action a │
       │in state s and following policy π thereafter │
       └─────────────────────────────────────────────┘
                               │
                      No model → Learn Q directly
                               │
                               ▼
                   Generalized Policy Iteration
       ┌───────────────────────────┬───────────────────────────┐
       │   Policy Evaluation       │     Policy Improvement    │
       │   Learn Qπ(s,a)           │   π ← greedy w.r.t Q      │
       └───────────────────────────┴───────────────────────────┘
                               │
                               ▼
                Challenge: Exploration vs. Exploitation
       ┌──────────────────────────────────────────────────────┐
       │Greedy policy → Exploits but stops exploring          │
       │ε-greedy policy → Balances exploration & exploitation │
       │GLIE condition: ε → 0 and ∞ exploration               │
       └──────────────────────────────────────────────────────┘
                               │
                               ▼
                Model-Free Control Families (Tabular)
       ┌────────────────────────────┬────────────────────────────┐
       │   Monte Carlo Control      │      Temporal Difference   │
       │   (Episode-based)          │      (Step-based)          │
       └────────────────────────────┴────────────────────────────┘
                               │
          ┌────────────────────┴───────────────────┐
          ▼                                        ▼
 Monte Carlo Control:                       TD Control:
 Estimates Q from full returns          Estimates Q usingbootstrapped targets
 Uses ε-greedy policy                   Works online, faster, low variance
 Episodic only                          Works for episodic & continuing
          │                                        │
    ┌─────┴─────────────┐             ┌────────────┴───────────────┐
    │ GLIE MC Control   │             │ On-Policy TD: SARSA        │
    └───────────────────┘             │ Off-Policy TD: Q-Learning  │
                                      └────────────────────────────┘
                                                    |
                                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
| On-Policy TD — SARSA                     |  Off-Policy TD — Q-Learning       |
| Learns Qπ for the policy being followed  |  Learns Q* while following π_b    |
| Update uses next action from π           |  Update uses max action (greedy)  |
| Update Target:                           |  Update Target:                   |
|  r + γ Q(s',a')                          |  r + γ maxₐ Q(s',a)               |
└────────────────────────────┴─────────────────────────────────────────────────┘

                               
                               
      Value Function Approximation (Large/Continuous spaces)
       ┌──────────────────────────────────────────────────────┐
       │ Replace Q(s,a) with Q̂(s,a;w) using function approx   │
       │ Generalization across states                         │
       │ Gradient-based updates (SGD)                         │
       └──────────────────────────────────────────────────────┘
                               │
                               ▼
            Deep Q-Learning (DQN) — Stable VFA Control
       ┌──────────────────────────────────────────────────────┐
       │ Experience Replay — decorrelate samples             │
       │ Target Networks — stabilize bootstrapped targets    │
       └──────────────────────────────────────────────────────┘
                               │
                               ▼
                      Final Outcome of Model-Free Control
       ┌───────────────────────────────────────────────────────┐
       │ Learn π* directly from experience without model       │
       │ Learn Q*(s,a) through MC, SARSA, or Q-Learning        │
       │ Scale to large spaces using function approximation    │
       │ DQN enables deep RL in complex environments           │
       └───────────────────────────────────────────────────────┘

```

