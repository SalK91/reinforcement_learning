# Chapter 2: Markov Decision Processes and Dynamic Programming
Reinforcement Learning relies on the mathematical framework of Markov Decision Processes (MDPs) to formalize sequential decision-making under uncertainty. The key idea is that an agent interacts with an environment, making decisions that influence both immediate and future rewards.

> Reinforcement Learning is about selecting actions over time to maximize long-term reward.

## The Markovian Hierarchy

The RL framework is built upon three foundational models, each adding complexity and agency.

### The Markov Process
A Markov Process, or Markov Chain, is the simplest model, concerned only with the flow of states. It is defined by the set of States ($S$) and the Transition Model ($P(s' \mid s)$). The defining characteristic is the Markov Property: the next state is independent of the past states, given only the current state.

$$
P(s_{t+1} \mid s_t, s_{t-1}, \ldots) = P(s_{t+1} \mid s_t)
$$

>  The future is conditionally independent of the past given the present. *Intuition: MPs describe what happens but do not assign any value to these events.*

### The Markov Reward Process (MRP)

A Markov Reward Process (MRP) extends an MP by adding rewards and discounting. An MRP is a tuple $(S, P, R, \gamma)$ where $R(s)$ is the expected reward for being in state $s$ and $\gamma$ is the discount factor. The return is:

$$
G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \dots
$$

The goal is to compute the value function, which is the expected return starting from a state $s$:

$$
V(s) = \mathbb{E}[G_t | s_t = s]
$$

The value function satisfies the Bellman Expectation Equation:

$$
V(s) = R(s) + \gamma \sum_{s'} P(s'|s)V(s')
$$

This recursive structure relates the value of a state to the values of its successor states.

### The Markov Decision Process (MDP)
An MDP introduces agency. Defined by the tuple $(S, A, P, R, \gamma)$, it extends the MRP by giving the agent a set of Actions ($A$) to choose from.

* Action-Dependent Transition: $P(s' \mid s, a)$
* Action-Dependent Reward: $R(s, a)$

The agent's strategy is described by a Policy ($\pi(a \mid s)$), the probability of selecting action $a$ in state $s$. A key insight is that fixing any policy $\pi$ reduces an MDP back into an MRP, allowing all tools developed for MRPs to be applied to the MDP.


$$
R_\pi(s) = \sum_a \pi(a|s) R(s,a)
$$

$$
P_\pi(s'|s) = \sum_a \pi(a|s) P(s'|s,a)
$$


> Once actions are introduced in an MDP, it becomes useful to evaluate not only how good a state is, but how good a particular action is *relative to the policy’s expected behavior*. This leads to the advantage function.

> The state-value function measures how good it is to be in a state: $V_\pi(s) = \mathbb{E}_\pi[G_t \mid s_t = s]$.

> The action-value function measures how good it is to take action $a$ in state $s$:$Q_\pi(s,a) = \mathbb{E}_\pi[G_t \mid s_t = s,\; a_t = a]$

> The **advantage function** compares these two: $A_\pi(s,a) = Q_\pi(s,a) - V_\pi(s).$

> $V_\pi(s)$ is how well the policy performs *on average* from state $s$.

> $Q_\pi(s,a)$ is how well it performs if it specifically takes action $a$.

> Therefore, the advantage tells us: How much better or worse action $a$ is compared to what the policy would normally do in state $s$.

## Value Functions and Expectation

To evaluate a fixed policy $\pi$, we define two inter-related value functions based on the Bellman Expectation Equations.

### State Value Function ($V^\pi(s)$)
$V^\pi(s)$ quantifies the long-term expected return starting from state $s$ and strictly following policy $\pi$.
$$
V^\pi(s) = \mathbb{E}[G_t \mid s_t = s, \pi]
$$
> How much total reward should I expect if I start in state s and follow policy $\pi$: forever?

### State-Action Value Function ($Q^\pi(s,a)$)
$Q^\pi(s,a)$ is a more granular measure, quantifying the expected return if the agent takes action $a$ in state $s$ first, and *then* follows policy $\pi$.
$$
Q^\pi(s,a) = R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^\pi(s')
$$
> *Intuition:* The $Q$-function is the value of doing a specific action; the $V$-function is the value of being in a state (the weighted average of the $Q$-values offered by the policy $\pi$ in that state):
$$
V^\pi(s) = \sum_a \pi(a \mid s) Q^\pi(s,a)
$$

The Bellman Expectation Equation for $V^\pi$ links the value of a state to the values of the actions chosen by $\pi$ and the resulting future states:
$$
V^\pi(s) = \sum_a \pi(a \mid s) \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^\pi(s') \right]
$$

 
## Optimal Control: Finding $\pi^*$

The ultimate goal of solving an MDP is to find the optimal policy ($\pi^*$) that maximizes the expected return from every state $s$.

$$
\pi^* = \operatorname*{arg\,max}_{\pi} V^\pi(s) \quad \text{for all } s \in S
$$

This optimal policy is characterized by the Optimal Value Functions ($V^*$ and $Q^*$).

### The Bellman Optimality Equations
These equations are fundamental, describing the unique value functions that arise when acting optimally. Unlike the expectation equations, they contain a $\max$ operator, making them non-linear.

* Optimal State Value ($V^*$): The optimal value of a state equals the maximum expected return achievable from any single action $a$ taken from that state:

    $$
    V^*(s) = \max_{a} \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^*(s') \right]
    $$

* Optimal Action-Value ($Q^*$): The optimal value of taking action $a$ is the immediate reward plus the discounted value of the optimal subsequent actions ($\max_{a'}$) in the next state $s'$:

    $$
    Q^*(s,a) = R(s,a) + \gamma \sum_{s'} P(s'|s,a) \max_{a'} Q^*(s', a')
    $$

Once $Q^*$ is known, the optimal policy $\pi^*$ is easily extracted by simply choosing the action that maximizes $Q^*(s,a)$ in every state:
$$
\pi^*(s) = \operatorname*{arg\,max}_{a} Q^*(s,a)
$$

These equations are non-linear due to the max operator and must be solved iteratively.

## Dynamic Programming Algorithms

For MDPs where the model ($P$ and $R$) is fully known, Dynamic Programming methods are used to solve the Bellman Optimality Equations iteratively.


### Policy Iteration
Policy Iteration follows an alternating cycle of Evaluation and Improvement. It takes fewer, but more expensive, iterations to converge.

1.  Policy Evaluation: For the current policy $\pi_k$, compute $V^{\pi_k}$ by iteratively applying the Bellman Expectation Equation until full convergence. This is the computationally intensive step.
    $$
    V^{\pi_k}(s) \leftarrow \text{solve } V^{\pi_k} = R_{\pi_k} + \gamma P_{\pi_k} V^{\pi_k}
    $$
2.  Policy Improvement: Update the policy $\pi_{k+1}$ by choosing an action that is greedy with respect to the fully converged $V^{\pi_k}$.
    $$
    \pi_{k+1}(s) \leftarrow \operatorname*{arg\,max}_{a} \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^{\pi_k}(s') \right]
    $$
    
The process repeats until the policy stabilizes ($\pi_{k+1} = \pi_k$), guaranteeing convergence to $\pi^*$.

### Value Iteration
Value Iteration is a single, continuous process that combines evaluation and improvement by repeatedly applying the Bellman Optimality Equation. It takes many, but computationally cheap, iterations.

1.  Iterative Update: For every state $s$, update the value function $V_k(s)$ using the $\max$ operation. This immediately incorporates a greedy improvement step into the value update.
    $$
    V_{k+1}(s) \leftarrow \max_{a} \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) V_k(s') \right]
    $$
2.  Convergence: The iterations stop when $V_{k+1}$ is sufficiently close to $V^*$.
3.  Extraction: The optimal policy $\pi^*$ is then extracted greedily from the final $V^*$.

### PI vs VI
| Feature | Policy Iteration (PI) | Value Iteration (VI) |
| :--- | :--- | :--- |
| Core Idea | Evaluate completely, then improve. | Greedily improve values in every step. |
| Equation | Uses Bellman Expectation (inner loop) | Uses Bellman Optimality (max) |
| Convergence | Few, large policy steps. Policy guaranteed to stabilize faster. | Many, small value steps. Value function converges slowly to $V^*$. |
| Cost | High cost per iteration (due to full evaluation). | Low cost per iteration (due to one-step backup). |

 

## MDPs Mental Map  


```text
                   Markov Decision Processes (MDPs)
        Formalizing Sequential Decision-Making under Uncertainty
                                  │
                                  ▼
                       Progression of Markov Models
       ┌─────────────────────────────────────────────────────────┐
       │  Markov Process (MP): States & Transition Probabilities │
       │   [S, P(s'|s)] — No rewards, no decisions               │
       └─────────────────────────────────────────────────────────┘
                                  │
                                  ▼
       ┌─────────────────────────────────────────────────────────┐
       │  Markov Reward Process (MRP): MP + Rewards + γ          │
       │  [S, P(s'|s), R(s), γ]                                  │
       │    Value Function: V(s) = E[Gt | st = s]                │
       │     Bellman Expectation Eqn:                            │
       │     V(s) = R(s) + γ ∑ P(s'|s)V(s')                      │
       └─────────────────────────────────────────────────────────┘
                                  │
                                  ▼
       ┌─────────────────────────────────────────────────────────┐
       │  Markov Decision Process (MDP): MRP + Actions           │
       │   [S, A, P(s'|s,a), R(s,a), γ]                          │
       │    Adds Agency: Agent chooses actions                   │
       └─────────────────────────────────────────────────────────┘
                                  │
                                  ▼
                             Policy π(a|s)
                        Agent’s decision strategy
                                  │
                                  ▼
                          Value Functions
     ┌────────────────────────────────────────────────────────────────┐
     │ State Value Vπ(s): Expected return following π                 │
     │ Qπ(s,a): Expected return from (s,a) then follow π              │
     │ Relationship: Vπ(s) = ∑ π(a|s) Qπ(s,a)                         │
     └────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
                    Bellman Expectation Equations
     ┌────────────────────────────────────────────────────────────────┐
     │ Vπ(s) = ∑ π(a|s)[R(s,a) + γ ∑ P(s'|s,a)Vπ(s')]                 │
     │ Qπ(s,a) = R(s,a) + γ ∑ P(s'|s,a) Vπ(s')                        │
     └────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
               Goal: Find Optimal Policy π*
     ┌─────────────────────────────────────────────────────────────┐
     │ π*(s) = argmaxₐ Q*(s,a)                                     │
     │ V*(s): Max possible value from state s under the optimal    |
     |        policy                                               │
     │ Q*(s,a): Max possible return state s by taking action a     |
     |          and thereafter following the optimal policy        │
     └─────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
                      Bellman Optimality Equations
     ┌─────────────────────────────────────────────────────────────┐
     │ V*(s) = maxₐ [R(s,a) + γ ∑ P(s'|s,a)V*(s')]                 │
     │ Q*(s,a) = R(s,a) + γ ∑ P(s'|s,a) maxₐ' Q*(s',a')            │
     └─────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
                 Solution when Model (P,R) is known:
                    Dynamic Programming (DP)
     ┌───────────────────────────────────────────────┬───────────────────┐
     │ Policy Iteration                              │ Value Iteration - │
     │ (Alternating Evaluation & Improvement)        │ Single update step│
     │                                               │ repeatedly        │
     └───────────────────────────────────────────────┴───────────────────┘
          │                                               │
          ▼                                               ▼
  ┌─────────────────┐                          ┌─────────────────────────┐
  │ Policy Eval     │                          │ Bellman Optimality      │
  │ Using Vπ until  │                          │ Update every iteration  │
  │ convergence     │                          │ V_(k+1) = max_a[....]   │
  └─────────────────┘                          └─────────────────────────┘
          │                                               │
          ▼                                               ▼
  ┌─────────────────┐                          ┌─────────────────────────┐
  │ Policy          │                          │ After convergence:      │
  │ Improvement:    │                          │ extract π* from Q*      │
  │ π_(k+1)=argmax Q│                          └─────────────────────────┘
  └─────────────────┘
                                  │
                                  ▼
             Outcome: Optimal Policy and Value Functions
       ┌─────────────────────────────────────────────────────┐
       │ π*(s) — Best action at each state                   │
       │ V*(s) — Max return achievable                       │
       │ Q*(s,a) — Max return from (s,a)                     │
       └─────────────────────────────────────────────────────┘


```