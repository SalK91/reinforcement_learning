# Chapter 13: Monte Carlo Tree Search

Monte Carlo Tree Search (MCTS) is a powerful planning algorithm that uses simulation-based search to select actions in complex decision-making problems. It is especially effective in large or unknown environments where exact planning is infeasible. MCTS balances exploration and exploitation through sampling and is the backbone of major AI breakthroughs like AlphaGo and AlphaZero.


## 13.1 Motivation

In classical reinforcement learning (RL), agents often compute policies over the *entire* state space. MCTS takes a different approach: it performs local search from the current state, using simulated episodes to estimate action values and make near-optimal decisions *on the fly*.

This method is particularly useful in:

- Large state/action spaces
- Games with high branching factor (e.g., Go, Chess)
- Black-box or simulator-only environments


## 13.2 Monte Carlo Search

A simple Monte Carlo search uses a model $\mathcal{M}$ (dynamics and resward model) and a rollout policy $\pi$ to simulate $K$ trajectories for each action $a$ from the current state $s_t$:

1. Simulate episodes $\{s_t, a, r_{t+1}^{(k)}, \ldots, s_T^{(k)}\}$ from $\mathcal{M}, \pi$.
2. Estimate $Q(s_t, a)$ via sample average:

$$
Q(s_t, a) = \frac{1}{K} \sum_{k=1}^K G_t^{(k)} \rightarrow q^\pi(s_t, a)
$$

3. Select the best action:

$$
a_t = \arg\max_a Q(s_t, a)
$$

This performs one-step policy improvement, but does not build deeper search trees.

## 13.3 Expectimax Search

To go beyond single-step rollouts, expectimax trees compute $Q^*(s, a)$ recursively using the model:

- Each node expands by looking ahead using the transition model.
- Combines maximization (over actions) and expectation (over next states).
- Forward search avoids solving the entire MDP and focuses only on the subtree starting at $s_t$.

However, the number of nodes grows exponentially with horizon $H$: $O(|S||A|)^H$.


## 13.4 Monte Carlo Tree Search (MCTS)

MCTS improves on expectimax by sampling rather than fully expanding the tree:

1. Build a tree rooted at current state $s_t$.
2. Perform $K$ simulations to expand and update parts of the tree.
3. Estimate $Q(s, a)$ using sampled returns.
4. Select the best action at the root:

$$
a_t = \arg\max_a Q(s_t, a)
$$

 
## 13.5 Upper Confidence Tree (UCT)

A key challenge in MCTS is deciding which action to simulate at each tree node. UCT addresses this by treating each decision as a multi-armed bandit problem and using an Upper Confidence Bound:

$$
Q(s, a, i) = \underbrace{\frac{1}{N(i, a)} \sum_{k=1}^{N(i,a)} G_k(i,a)}_{\text{Mean Return}} + \underbrace{c \sqrt{\frac{\log N(i)}{N(i, a)}}}_{\text{Exploration Bonus}}
$$

- $N(i, a)$: number of times action $a$ taken at node $i$
- $N(i)$: total visits to node $i$
- $c$: exploration constant
- $G_k(i, a)$: return from simulation $k$ for $(i, a)$

Action selection:

$$
a_k^i = \arg\max_a Q(s, a, i)
$$

This balances exploitation of known good actions and exploration of uncertain ones.

 
## 13.6 Advantages of MCTS

- Anytime: Can stop search at any time and use the best estimates so far.
- Model-based or black-box: Only needs sample access to the environment.
- Best-first: Focuses computation on promising actions.
- Scalable: Avoids full enumeration of action/state spaces.
- Parallelizable: Independent simulations can be run in parallel.


## 13.7 AlphaZero and Deep MCTS

AlphaZero revolutionized game-playing AI by combining deep learning with MCTS. Key ideas:

### Policy and Value Networks

A neural network $f_\theta(s)$ outputs:

- $P$: action probabilities
- $V$: value estimate

$$
(p, v) = f_\theta(s)
$$

### AlphaZero MCTS Steps

1. Select: Traverse tree using $Q + U$ to choose child nodes.
2. Expand: Add a new node, initialized with $P$ from $f_\theta$.
3. Evaluate: Use $v$ from the network as the value of the leaf.
4. Backup: Propagate value estimates up the tree.
5. Repeat: Perform many rollouts to refine the tree.

### Root Action Selection

At the root, use visit counts $N(s,a)$ to compute the improved policy:

$$
\pi(s, a) \propto N(s, a)^{1/\tau}
$$

where $\tau$ controls exploration vs exploitation.

 
## 13.8 Self-Play and Training

AlphaZero uses self-play to generate training data:

1. Play full games using MCTS.
2. Record $(s, \pi, z)$ tuples where:
   - $s$: game state
   - $\pi$: improved policy from MCTS
   - $z$: final game outcome
3. Train $f_\theta$ to minimize combined loss:

$$
\mathcal{L} = (z - v)^2 - \pi^\top \log p + \lambda \|\theta\|^2
$$

This allows continual improvement without human supervision.

 
## 13.9 Evaluation and Impact

- MCTS dramatically improves performance over raw policy/value networks.
- Essential to surpassing human performance in Go, Chess, and Shogi.
- Eliminates the need for human expert data.

Insights:

- UCT enables principled tree search with exploration.
- Neural nets guide and accelerate MCTS.
- MCTS can be used in any environment where lookahead is possible.

 
## 13.10 Summary

- MCTS uses simulation-based planning with a growing search tree.
- UCT adds upper confidence bounds to balance exploration/exploitation.
- AlphaZero combines MCTS with deep learning for superhuman performance.
- Self-play enables autonomous training without labeled data.

MCTS represents a powerful bridge between planning and learning, enabling agents to make strong decisions under uncertainty in complex domains.
