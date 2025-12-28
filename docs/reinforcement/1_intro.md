# Chapter 1: Introduction to Reinforcement Learning

Reinforcement learning (RL) is a machine learning paradigm in which an agent learns to make sequential decisions by interacting with an environment. In contrast to supervised learning (learning from labeled input–target pairs) and unsupervised learning (learning structure from unlabeled data), RL is organized around a goal: learn behavior that maximizes cumulative reward through trial and error. The agent is not given the correct action at each step; instead, it must discover good actions by exploring and observing the consequences.

A defining feature of RL is sequential decision-making under uncertainty. Actions influence not only immediate outcomes but also the future states the agent will encounter. This creates a fundamental tension:

- exploration: trying actions that may be informative about the environment, even if they seem suboptimal now
- exploitation: choosing actions that currently appear best in order to collect reward

A second defining feature is delayed feedback. Rewards may be sparse or delayed, so it can be unclear which earlier decisions caused a later success or failure. This is the temporal credit assignment problem.

## Supervised learning vs reinforcement learning

|  | supervised learning | reinforcement learning |
|---|---|---|
| learning source | labeled examples $(x \rightarrow y)$ | interaction (trial and error) |
| feedback | immediate and correct | may be delayed, sparse, or noisy |
| data distribution | typically i.i.d. | non-i.i.d., depends on agent actions |
| temporal dependence | usually none | inherently sequential |
| key challenges | generalization from labels | exploration, temporal credit assignment, stability |

## Agent–environment interaction

At each time step $t$, the agent interacts with the environment as follows:

1. the agent receives information about the current situation (a state or an observation)
2. the agent chooses an action
3. the environment transitions to a new state and produces a reward

This generates a trajectory (experience sequence), for example:
$$
s_0, a_0, r_1, s_1, a_1, r_2, \dots
$$

Key objects:

- state $s_t$: the true state of the environment at time $t$. Intuitively, it contains all information needed to predict future dynamics given an action.
- observation $o_t$: what the agent actually perceives at time $t$. In partially observable settings, $o_t$ may not uniquely determine $s_t$.
- action $a_t$: the decision made by the agent at time $t$.
- reward $r_{t+1}$ (often written as $r_t$ by convention): a scalar feedback signal. A common notation makes the reward depend on the transition:
  $$
  r_{t+1} = r(s_t, a_t, s_{t+1})
  $$
- trajectory $\tau$: a sequence of states (or observations), actions, and rewards, e.g.
  $$
  \tau = (s_0, a_0, r_1, s_1, a_1, r_2, \dots)
  $$

### The Markov property

Many RL problems are modeled as Markov decision processes (MDPs). The Markov property states that the next state depends only on the current state and action, not on the full past:
$$
P(s_{t+1} \mid s_t, a_t, s_{t-1}, a_{t-1}, \dots) \;=\; P(s_{t+1} \mid s_t, a_t).
$$

Equivalently, if $h_t$ denotes the full history up to time $t$, then a Markov state satisfies:
$$
p(s_{t+1} \mid s_t, a_t) \;=\; p(s_{t+1} \mid h_t, a_t).
$$

When the agent does not observe the true state and instead receives partial observations $o_t$, the Markov property may fail from the agent’s perspective. In that case, optimal behavior may require memory, for example a history-dependent policy:
$$
\pi_\theta(a_t \mid o_{t-m}, \dots, o_t)
\quad\text{or more generally}\quad
\pi_\theta(a_t \mid o_{0:t}).
$$

## Policies and the return

The agent’s behavior is described by a policy $\pi$. A policy is a mapping from states (or observations) to a distribution over actions:
$$
a_t \sim \pi(a_t \mid s_t).
$$

The agent’s objective is to learn a policy that maximizes long-term reward. To formalize “long-term,” we define the return. In episodic tasks, a common choice is the discounted return:
$$
G_t \;=\; r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \dots,
$$
where $0 \le \gamma \le 1$ is the discount factor. Smaller $\gamma$ emphasizes immediate rewards; larger $\gamma$ emphasizes long-term planning and makes the agent more farsighted.

Because the environment and/or the policy may be stochastic, the cumulative reward is typically a random variable. Two common sources of randomness are:

1. environment stochasticity:
   $$
   s_{t+1} \sim P(\cdot \mid s_t, a_t)
   $$
2. policy stochasticity:
   $$
   a_t \sim \pi(\cdot \mid s_t)
   $$

For this reason, RL typically maximizes expected return. For a finite horizon $T$, a standard objective is:
$$
\max_\theta \; \mathbb{E}_{\tau \sim p_\theta(\tau)}\Big[\sum_{t=0}^{T} r(s_t, a_t)\Big],
$$
where $p_\theta(\tau)$ denotes the trajectory distribution induced by the policy parameters $\theta$ and the environment dynamics.

## Fundamental challenges in reinforcement learning

Reinforcement learning combines several difficulties that do not typically appear together in supervised learning:

1. optimization: the agent must search over a space of behaviors (policies) to improve performance
2. delayed consequences: decisions can influence rewards far in the future, creating temporal credit assignment
3. exploration: the agent must collect informative experience while still trying to achieve high reward
4. generalization: the agent must use limited interaction data to perform well in new or rarely visited states

## When reinforcement learning is useful

RL is especially appropriate when:

- correct behavior is hard to specify with labels
- data must be collected by interacting with the system
- decisions have long-term consequences
- the environment may change in response to the agent’s actions

Applications include robotics, autonomous systems, game playing, resource allocation, recommendation and ranking, and other online decision-making problems.

## Problem types and common distinctions

### Episodic vs continuing tasks

RL problems can be episodic or continuing.

- episodic: interaction terminates after a finite number of steps and the environment resets. The return is naturally finite.
- continuing: interaction does not terminate. Discounting (with $\gamma < 1$) or average-reward objectives are commonly used to make the objective well-defined.

### Prediction, control, and planning

RL problems can be grouped by what the agent is asked to do:

- prediction (policy evaluation): estimate how good a given policy is
- control: find a policy that maximizes expected return
- planning: compute optimal behavior using a known model of the environment dynamics

### Model-based vs model-free methods

- model-based methods use a model of the environment dynamics and rewards. The model may be given or learned, and is used for planning or generating improved policies.
- model-free methods do not use an explicit dynamics model for decision-making and instead learn values and/or policies directly from experience.

### On-policy vs off-policy learning

- on-policy learning uses experience generated by the same policy that is being improved or evaluated.
- off-policy learning uses experience generated by a different behavior policy (for example, older versions of the policy, a different exploration policy, or a dataset collected offline).

### Tabular vs function approximation

In small environments, value functions and policies can be represented exactly with tables over discrete states and actions. In large or continuous environments, this becomes impossible, and function approximation is used. A parameterized function (such as a linear model or neural network) represents the value function and/or the policy, enabling generalization across similar states.

## Families of reinforcement learning algorithms

A useful high-level grouping of RL approaches includes:

- imitation learning: learn a policy from expert demonstrations
- value-based methods: learn value functions and derive a policy from them
- policy gradient methods: directly optimize a parameterized policy using gradients of expected return
- actor–critic methods: combine policy optimization (actor) with value estimation (critic)
- model-based methods: learn or use environment models for planning and policy improvement

There are many RL algorithms because practical settings vary widely in how costly data is to collect, what supervision is available, how stable optimization must be, and what the action and state spaces look like. Factors that often shape algorithm choice include:

- data collection cost: simulation vs real-world interaction
- supervision availability: demonstrations, dense rewards, sparse rewards
- stability and ease of tuning
- action space: discrete vs continuous, low- vs high-dimensional
- learnability of environment dynamics: whether a useful predictive model can be learned


