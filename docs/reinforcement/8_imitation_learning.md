   
# Chapter 8: Imitation Learning
In previous chapters, we focused on reinforcement learning with explicit reward signals guiding the agent's behavior. We assumed that a well-defined reward function $R(s,a)$ was provided as part of the MDP, and the agent’s goal was to learn a policy that maximizes cumulative reward. But what if specifying the reward is difficult or the agent cannot safely explore to learn from reward? Imitation Learning (IL) addresses these scenarios by leveraging expert demonstrations instead of explicit rewards.


Imitation Learning allows an agent to learn how to act by mimicking an expert’s behavior, rather than by maximizing a hand-crafted reward.

## Motivation: The Case for Learning from Demonstrations
Designing a reward function that truly captures the desired behavior can be extremely challenging. A misspecified reward can lead to unintended behaviors (reward hacking) or require exhaustive tuning. Even with a good reward, some environments present sparse rewards (e.g. only a success/failure signal at the very end of an episode) – making pure trial-and-error learning inefficient. In other cases, unsafe exploration is a concern: letting an agent freely explore (as classic RL would) could be dangerous or costly (imagine a self-driving car learning by crashing to discover that crashing is bad).

However, in many of these settings expert behavior is available: we might have logs of human drivers driving safely, or demonstrations of a robot performing the task. Imitation Learning leverages this data. Instead of specifying what to do via a reward function, we show the agent how to do it via example trajectories. The agent's objective is then to imitate the expert as closely as possible.

This paradigm contrasts with reward-based RL in key ways:

* Reward-Based RL: The agent explores and learns by trial-and-error, guided by a numeric reward signal for feedback. It requires careful reward design and often extensive exploration.

* Imitation Learning: The agent learns from demonstrations of the desired behavior, treating the expert’s actions as ground truth. No explicit reward is needed to train; learning is driven by matching the expert's behavior.

By learning from an expert, IL can produce competent policies much faster and safer in these scenarios. It essentially sidesteps the credit assignment problem of RL (because the "right" action is directly provided by the expert) and avoids dangerous exploration. In domains like autonomous driving, robotics, or any task where a human can demonstrate the skill, IL offers a powerful shortcut to get an agent up to a reasonable performance.


## Imitation Learning Problem Setup

Formally, we can describe the imitation learning scenario using the same environment structure as an MDP $(S, A, P, R, \gamma)$ except that the reward function $R$ is unknown or not used. The agent still has a state space $S$, an action space $A$, and the environment transition dynamics $P(s' \mid s, a)$. What we do have, instead of $R$, is access to expert demonstrations. An expert (which could be a human or a pre-trained optimal agent) provides example trajectories:

$$
\tau_E = (s_0, a_0, s_1, a_1, \dots , s_T)
$$

collected by following the expert’s policy $\pi_E$ in the environment. We may have a dataset $D$ of these expert trajectories (or simply a set of state-action pairs drawn from expert behavior). The key point is that in IL, the agent does not receive numeric rewards from the environment. Instead, success is measured by how well the agent’s behavior matches the expert’s behavior.

The goal of imitation learning can be stated as: find a policy $\pi$ for the agent that reproduces the expert's behavior (and ideally, achieves similar performance on the task). If the expert is optimal or highly skilled, we hope $\pi$ will achieve near-optimal results as well. This is an alternative path to finding a good policy without ever specifying a reward function explicitly or performing unguided exploration.

(If we imagine there was some true but unknown reward $R$ the expert is optimizing, then ideally $\pi$ should perform nearly as well as $\pi_E$ on that reward. IL attempts to reach that outcome via demonstrations rather than explicit reward feedback.)

## 3. Behavioral Cloning: Learning by Supervised Imitation
The most direct approach to imitation learning is Behavioral Cloning. Behavioral cloning treats imitation as a pure supervised learning problem: we train a policy to map states to the expert’s actions, using the expert demonstrations as labeled examples. In essence, the agent "clones" the expert's behavior by learning to predict the expert's action in any given state.

> BC: Learn state to action mappings using expert demonstrations.

In practice, we parameterize a policy $\pi_\theta(a\mid s)$ (e.g. a neural network with parameters $\theta$) and adjust $\theta$ so that $\pi_\theta(\cdot\mid s)$ is as close as possible to the expert’s action choice in state $s$. We define a loss function on the dataset of state-action pairs. For example:

* Discrete actions: Use cross-entropy (negative log-likelihood) of the expert’s action.

$$L(\theta) = - \mathbb{E}_{(s,a)\sim D}\left[ \log \pi_{\theta}(a \mid s) \right]$$


* Continuous actions: Use mean squared error (regression loss).

$$L(\theta) = \mathbb{E}_{(s,a)\sim D} \left[ \left( \pi_\theta(s) - a \right)^2 \right]$$

Minimizing these losses drives the policy to imitate the expert decisions on the training set.

Training a behavioral cloning agent typically involves three steps:

1. Collect demonstrations: Gather a dataset $D = {(s_i, a_i)}$ of expert state-action examples by observing the expert $\pi_E$ in the environment.

2. Supervised learning on $(s, a)$ pairs: Choose a policy representation for $\pi_\theta$ and use the collected data to adjust $\theta$. For each example $(s_i, a_i)$, we update $\pi_\theta$ to reduce the error between its prediction $\pi_\theta(s_i)$ and the expert’s action $a_i$. (For instance, if actions are discrete, we increase the probability $\pi_\theta(a_i \mid s_i)$ for the expert’s action; if continuous, we move $\pi_\theta(s_i)$ closer to $a_i$ in value.)

3. Deployment: Once the policy is trained (approximating $\pi_E$), we fix $\theta$. The agent then acts autonomously: at each state $s$, it outputs $a = \pi_\theta(s)$ as its action. Ideally, this learned policy will behave similarly to the expert in the environment.

If the expert demonstrations are representative of the situations the agent will face, behavioral cloning can yield a policy that mimics the expert’s behavior effectively. BC has some clear advantages:

* Simplicity: It reduces policy learning to standard supervised learning, for which many stable algorithms and optimizations exist.

* Offline training: The model can be trained entirely from pre-recorded expert data, without requiring interactive environment feedback. This makes it data-efficient in terms of environment interactions.

* Safety: No random exploration is needed. The agent never tries highly suboptimal actions during training, since it always learns from demonstrated good behavior (critical in safety-sensitive domains).

However, purely copying the expert also comes with important limitations.

### Covariate Shift and Compounding Errors

The main problem with behavioral cloning is that the training distribution of states can differ from the test distribution when the agent actually runs. During training, $\pi_\theta$ is only exposed to states that the expert visited. But once the agent is deployed, if it ever deviates even slightly from the expert’s trajectory, it may enter states not seen in the training data. In those unfamiliar states, the policy’s predictions may be unreliable, leading to errors that cause it to drift further from expert-like behavior.

> A small mistake can snowball: once the agent strays from what the expert would do, it encounters novel situations where its learned policy might be very poor. One error leads to another, and the agent can cascade into failure because it was never taught how to recover.

This phenomenon is known as covariate shift or distributional shift. The learner is trained on the state distribution induced by the expert policy $\pi_E$, but it is testing on the state distribution induced by its own policy $\pi_\theta$. Unless $\pi_\theta$ is perfect, these distributions will diverge over time, and the divergence can grow unchecked. In other words, the agent might handle situations similar to the expert's trajectories well, but if it finds itself in a situation the expert never encountered (often a result of a prior mistake), it has no guidance on what to do and can rapidly veer off course. This is often illustrated by the example of a self-driving car learned by BC: if it slightly misjudges a turn and drifts, it may end up in a part of the road it never saw during training, leading to more errors (compounding until possibly a crash).

Another limitation is that BC does not inherently guarantee optimality or improvement beyond the expert: the policy is only as good as the demonstration data. If the expert is suboptimal or the dataset doesn’t cover certain scenarios, the cloned policy will reflect those shortcomings and cannot improve by itself (since it has no feedback signal like reward to further refine its behavior). In reinforcement learning terms, BC has no notion of feedback for success or failure; it merely apes the expert, so it cannot discover better strategies or correct mistakes outside the expert's shadow.

Researchers have developed strategies to mitigate the covariate shift problem. One approach is Dataset Aggregation (DAgger), which is an iterative algorithm: after training an initial policy via BC, let the policy interact with the environment and observe where it makes mistakes or visits unseen states; then have the expert provide the correct actions for those states, add these state-action pairs to the training set, and retrain the policy. By repeating this process, the policy’s training distribution is gradually brought closer to the distribution it will encounter when it controls the agent. DAgger can significantly reduce compounding errors, but it requires ongoing access to an expert for feedback during training.

In summary, behavioral cloning is a powerful first step for imitation learning—it's straightforward and avoids many challenges of pure RL. But one must be mindful of its limitations: a blindly cloned policy can fail catastrophically when it encounters situations outside the expert’s experience. This motivates more sophisticated imitation learning methods that incorporate the dynamics of the environment and attempt to infer the intent behind expert actions, rather than just copying them. We turn to those next.

## Inverse Reinforcement Learning: Learning the "Why"
Behavioral cloning directly learns what to do (mapping states to actions) but does not capture why those actions are desirable. Inverse Reinforcement Learning (IRL) instead asks: Given expert behavior, what underlying reward function $R$ could explain it? In other words, IRL attempts to reverse-engineer the expert's objectives from its observed behavior.

In IRL, we assume that the expert $\pi_E$ is (approximately) optimal for some unknown reward function $R^*$. The goal is to infer a reward function $\hat{R}$ such that, if an agent were to optimize $\hat{R}$, it would reproduce the expert’s behavior. Formally, we want $\pi_E$ to be the optimal policy under the learned reward:

$$\pi_E = \arg\max_{\pi} \, V_R^{\pi}$$

where $V^{\pi}_{\hat{R}}$ is the expected return of policy $\pi$ under the reward function $\hat{R}$. In words, the expert should have higher cumulative reward (according to $\hat{R}$) than any other policy. If we can find such an $\hat{R}$, we have explained the expert’s behavior in terms of incentives.


> Intuition: IRL flips the reinforcement learning problem on its head. Rather than starting with a reward and finding a policy, we start with a policy (the expert's) and try to find a reward that this policy optimizes. It's like observing an expert driver and deducing that they must be implicitly trading off goals like "reach the destination quickly" and "avoid collisions" because their driving balances speed and safety.


One challenge is that IRL is inherently an under-defined (ill-posed) problem: many possible reward functions might make $\pi_E$ appear optimal. To resolve this ambiguity, IRL algorithms introduce additional criteria or regularization. For example, they might prefer the simplest reward function that explains the behavior, or in the case of maximum entropy IRL, prefer a reward that leads to the most random (maximally entropic) policy among those that match the expert's behavior – this avoids overly narrow explanations and spreads probability over possible behaviors unless forced by data.
 

Once a candidate reward function $\hat{R}(s,a)$ is learned through IRL, the process typically continues as follows: we plug $\hat{R}$ back into the environment and solve a forward RL problem (using any suitable algorithm from earlier chapters) to obtain a policy $\pi_{\hat{R}}$ that maximizes this recovered reward. Ideally, $\pi_{\hat{R}}$ will then behave similarly to the expert's policy $\pi_E$ (since $\hat{R}$ was chosen to explain $\pi_E$). The end result is an agent that not only imitates the expert, but also has an explicit reward model of the task it is performing.

 

IRL is usually more complex and computationally expensive than behavioral cloning, because it often involves a nested loop: for each candidate reward function, the algorithm may need to perform an inner optimization (solving an MDP) to evaluate how well that reward explains the expert. However, IRL provides several potential benefits:

* It yields a reward function, which is a portable definition of the task. This inferred reward can then be reused: for example, to train new agents from scratch, to evaluate different policies, or to modify the task (by tweaking the reward) in a principled way.

* It can generalize better to new situations. If the environment changes in dynamics or constraints, having $\hat{R}$ allows us to re-optimize and find a new optimal policy for the new conditions. A policy learned by pure BC might not adapt well beyond the situations it was shown, whereas a reward captures the goal and can be re-optimized.

* It may allow the agent to exceed the demonstrator’s performance. Since IRL ultimately produces a reward function, an agent can continue to improve with further RL optimization. If the expert was suboptimal or noisy, a sufficiently good RL algorithm might find a policy that achieves an even higher reward (i.e. fine-tunes the behavior) while still aligning with the expert’s intent encoded in $\hat{R}$.

 

In summary, IRL shifts the imitation learning problem from policy regression to reward inference. It answers a fundamentally different question: instead of directly cloning actions, infer the hidden goals that the expert is pursuing. With $\hat{R}$ in hand, we then fall back on standard RL techniques (like those from Chapters 4–8) to derive a policy. IRL is especially appealing in scenarios where we suspect the expert’s behavior is optimizing some elegant underlying objective, and we want to uncover that objective for reuse or interpretation. The cost of IRL is the added complexity of the learning process, but the payoff is a deeper understanding of the task and potentially greater robustness and optimality of the learned policy.

### Maximum Entropy Inverse Reinforcement Learning

### Principle of Maximum Entropy

The entropy of a distribution $p(s)$ is defined as:

$$H(p) = -\sum_{s} p(s)\log p(s)$$

The principle of maximum entropy states: The probability distribution that best represents our state of knowledge is the one with the largest entropy, given the constraints of precisely stated prior data. Consider all probability distributions consistent with the observed data. Select the one with maximum entropy—i.e., the least biased distribution that fits what we know while assuming nothing extra.
 

### Maximum Entropy Applied to IRL

We seek a distribution over trajectories $P(\tau)$ that:

1. Has maximum entropy, and
2. Matches expert feature expectations.

Formally, we maximize:

$$\max_{P} -\sum_{\tau} P(\tau)\log P(\tau)$$

subject to:

$$\sum_{\tau} P(\tau)\mu(\tau) = \frac{1}{|D|}\sum_{\tau_i \in D} \mu(\tau_i)$$

$$\sum_{\tau} P(\tau) = 1$$

Here:

- $\mu(\tau)$ represents feature counts for trajectory $\tau$
- $D$ is the expert demonstration set

This says: among all possible distributions consistent with observed expert feature averages, choose the one with maximum uncertainty.


### Matching Rewards

In linear reward IRL, we assume rewards take the form:

$$r_\phi(\tau) = \phi^\top \mu(\tau)$$

We want a policy $\pi$ that induces a trajectory distribution $P(\tau)$ matching the expert’s expected reward under $r_\phi$:

$$\max_{P(\tau)} -\sum_{\tau}P(\tau)\log P(\tau)$$

subject to:

$$\sum_{\tau} P(\tau)r_\phi(\tau) = \sum_{\tau} \hat{P}(\tau)r_\phi(\tau)$$

$$\sum_{\tau}P(\tau)=1$$

This aligns the learner’s expected reward with the expert’s reward estimate.

### Maximum Entropy ⇒ Exponential Family Distributions

Using constrained optimization (Lagrangians), we obtain:

$$\log P(\tau) = \lambda_1 r_\phi(\tau) - 1 - \lambda_0$$

Thus:

$$P(\tau) \propto \exp(r_\phi(\tau))$$

This reveals a key result: The maximum entropy distribution consistent with constraints belongs to the exponential family.

That is,

$$p(\tau|\phi) = \frac{1}{Z(\phi)}\exp(r_\phi(\tau))$$

where

$$Z(\phi)=\sum_{\tau}\exp(r_\phi(\tau))$$

This means we can now learn $\phi$ by maximizing likelihood of observed expert data, because the trajectory distribution becomes a normalized exponential model.


### Maximum Entropy Over $\tau$ Equals Maximum Likelihood of Observed Data Under Max Entropy (Exponential Family) Distribution

Jaynes (1957) showed:
Maximizing entropy over trajectories = maximizing likelihood of data under the maximum-entropy distribution.

So we:

1. Assume $p(\tau|\phi)$ has exponential form
2. Learn $\phi$ by maximizing:

$$\max_{\phi} \prod_{\tau \in D} p(\tau|\phi)$$

This allows IRL to treat expert demonstrations as data to be probabilistically explained.

## Maximum Entropy Inverse RL Algorithm

Assuming known dynamics and linear rewards:

1. Input: expert demonstrations $\mathcal{D}$
2. Initialize reward weights $r_\phi$
3. Compute optimal policy $\pi(a|s)$ given $r_\phi$ (via dynamic programming / value iteration)
4. Compute state visitation frequencies $\rho(s|\phi,T)$
5. Compute gradient on reward parameters:

      $\nabla J(\phi) = \frac{1}{N}\sum_{\tau_i \in \mathcal{D}} \mu(\tau_i) - \sum_{s}\rho(s|\phi,T)\mu(s)$

6. Update $\phi$ via gradient step
7. Repeat from Step 3

> Maximum Entropy IRL assumes experts act stochastically but optimally.
> Instead of selecting a single best policy, it finds a distribution over trajectories consistent with expert behavior.
> The resulting trajectory probabilities follow:
> $$P(\tau) \propto \exp(r_\phi(\tau))$$

> Learning becomes maximum likelihood estimation: find reward parameters $\phi$ that best explain expert demonstrations.


## Apprenticeship Learning

Apprenticeship Learning usually refers to the scenario where an agent learns to perform a task by iteratively improving its policy using expert demonstrations as a reference. In many contexts, this term is used when an IRL algorithm is combined with policy learning: the agent behaves as an apprentice to the expert, gradually mastering the task. The classic formulation by Abbeel and Ng (2004) introduced apprenticeship learning via IRL, which guarantees that the learner’s policy will perform nearly as well as the expert’s, given enough demonstration data.

 

One way to think of apprenticeship learning is as follows: rather than directly cloning actions, we try to match the feature expectations of the expert. Suppose we have some features $\phi(s)$ of states (or state-action pairs) that capture what we care about in the task (for example, in driving, features might include lane deviation, speed, collision count, etc.). The expert will have some expected cumulative feature values $ \mathbb{E}_{\pi_E}\left[\sum_t \phi(s_t)\right] $. Apprenticeship learning methods aim for the learner to achieve similar feature expectations.

 A prototypical apprenticeship learning algorithm proceeds like this:

1. Initialize a candidate policy (it could even start random).

2. Evaluate how this policy behaves in terms of features (run it in simulation to estimate $\mathbb{E}_{\pi}\left[\sum_t \phi(s_t)\right]$).

3. Compare the policy’s behavior to the expert’s behavior. Identify the biggest discrepancy in feature expectations.

4. Adjust the reward (implicitly defined as a weighted sum of features) to penalize the discrepancy. In other words, find reward weights $w$ such that the expert’s advantage over the apprentice in those feature dimensions is highlighted.

5. Optimize a new policy for this updated reward function (solve the MDP with the new $w$ to get $\pi_{\text{new}}$ that maximizes $w \cdot \phi$).

6. Set this $\pi_{\text{new}}$ as the apprentice’s policy and repeat the evaluation -> comparison -> reward adjustment cycle.

Each iteration pushes the apprentice to close the gap on the feature that most distinguishes it from the expert. After a few iterations, this process yields a policy that matches the expert on all key feature dimensions within some tolerance. At that point, the apprentice is essentially as good as the expert with respect to any reward expressible as a combination of those features.


The term apprenticeship learning highlights that the agent is not just mimicking blindly but is engaged in a process of improvement guided by the expert’s example. Importantly, the focus is on achieving at least the expert’s level of performance. We don’t necessarily care about identifying the exact reward the expert had; we care that our apprentice’s policy is successful. In fact, in the algorithm above, the reward weights $w$ found in each iteration are intermediate tools – at the end, one can take the final policy and deploy it, without needing to stick to a single explicit reward interpretation.


In relation to IRL, apprenticeship learning can be seen as a practical approach to use IRL for control: IRL finds a reward that explains the expert, and then the agent learns a policy for that reward; if it’s not yet good enough, adjust and repeat. Modern developments in imitation learning often follow this spirit. For example, Generative Adversarial Imitation Learning (GAIL) is a more recent technique where the agent learns a policy by trying to fool a discriminator into thinking the agent’s trajectories are from the expert – conceptually, the discriminator’s judgment provides a sort of reward signal telling the agent how "expert-like" its behavior is. This can be viewed as a form of apprenticeship learning, since the agent is iteratively tweaking its policy to become indistinguishable from the expert.

 In summary, apprenticeship learning is about learning by iteratively comparing to an expert and closing the gap. It often uses IRL under the hood, but its end goal is the policy (the apprentice’s skill), not necessarily the reward. It underscores a key point: in imitation learning, sometimes we care more about performing as well as the expert (a direct goal), and sometimes we care about understanding the expert’s intentions (the indirect goal via IRL). Apprenticeship learning emphasizes the former.

## Imitation Learning in the RL Landscape

Imitation learning fills an important niche in the overall reinforcement learning framework. It is especially useful when:

1. Rewards are difficult to specify: If it's unclear how to craft a reward that captures all aspects of the desired behavior, providing demonstrations can bypass this. IL shines in complex tasks (e.g. high-level driving maneuvers, dexterous robot manipulation) where manually writing a reward function would be cumbersome or prone to error.

2. Rewards are sparse or delayed: When reward feedback is very rare or only given at the end of an episode, a pure RL agent might struggle to get enough signal to learn. An expert trajectory provides dense guidance at every time step (state-action pairs), effectively providing a shaped signal through imitation. This can jump-start learning in tasks that are otherwise too sparse for RL to crack (Chapter 4 discussed how sparse rewards make value estimation difficult – IL sidesteps that by using expert knowledge).

3. Exploration is risky or expensive: In real-world environments like robotics, autonomous driving, or healthcare, exploring with random or untrained policies can be dangerous or costly. IL allows learning a policy without the agent ever taking unguided actions in the real environment; it learns from safe, successful behaviors demonstrated by the expert. This makes it an attractive approach when safety is a hard constraint.

It’s important to note that IL is not necessarily a replacement for reward-based RL, but rather a complement to it. A common practical approach is to bootstrap an agent with imitation learning and then fine-tune it with reinforcement learning. For example, one might first use behavioral cloning to teach a robot arm the basics of a task from human demonstrations, getting it into a reasonable regime of behavior; then, if a reward function is available (even a sparse one for success), use RL to further improve the policy, possibly surpassing the human expert's performance or adapting to slight changes in the task. The initial IL phase provides a good policy prior (saving time and avoiding dangerous exploration), and the subsequent RL phase lets the agent optimize and explore around that policy to refine skills.

 

On the flip side, imitation learning does require expert data. If obtaining demonstrations is hard (or if no expert exists for a brand-new task), IL might not be applicable. Moreover, if the expert demonstrations are of varying quality or contain noise, the agent will faithfully learn those imperfections unless additional measures (like filtering data or combining with RL optimization) are taken. In contrast, a pure RL approach, given a well-defined reward and enough exploration, can in principle discover superior strategies that no demonstrator provided. Thus, in practice, there is a trade-off: IL can dramatically speed up learning and improve safety given an expert, whereas RL remains the go-to when we only have a reward signal and the freedom to explore.

 

Imitation learning has become a critical part of the toolbox for solving real-world sequential decision problems. It enables success in domains that might be intractable for pure reinforcement learning by providing an external source of guidance. By learning directly from expert behavior – through methods like behavioral cloning (learning the policy directly) or inverse reinforcement learning (learning the underlying reward and then the policy) – an agent can shortcut the trial-and-error process. Of course, IL introduces its own challenges (distribution shift, reliance on demonstration coverage, potential suboptimality of the expert), but these can often be managed with algorithmic innovations (DAgger, combining IL with RL, etc.). In summary, imitation learning serves as a powerful paradigm for training agents in cases where designing rewards or allowing extensive exploration is impractical, and it often works hand-in-hand with traditional RL to achieve the best results in complex environments.

### Mental map

``` text
                    Imitation Learning (IL)
      Goal: Learn behavior from expert demonstrations
                     instead of explicit rewards
                                │
                                ▼
             Why Imitation Learning? (Motivation)
 ┌───────────────────────────────────────────────────────────┐
 │ Hard to design rewards → reward hacking, tuning           │
 │ Sparse rewards → inefficient trial & error                │
 │ Unsafe exploration (robots, driving, healthcare)          │
 │ Expert data available → demonstrations as guidance        │
 └───────────────────────────────────────────────────────────┘
                                │
                                ▼
                   IL vs Reward-Based RL
 ┌─────────────────────────────┬──────────────────────────────┐
 │ Reward-Based RL             │ Imitation Learning           │
 │ + Explores actively         │ + Learns from expert         │
 │ + Needs reward design       │ + No explicit reward         │
 │ – Unsafe / inefficient      │ – Depends on demo quality   │
 └─────────────────────────────┴──────────────────────────────┘
                                │
                                ▼
                         IL Problem Setup
 ┌───────────────────────────────────────────────────────────┐
 │ MDP without reward function                                │
 │ Access to expert trajectories τE (s,a pairs)               │
 │ Goal → Learn policy π that mimics πE                       │
 └───────────────────────────────────────────────────────────┘
                                │
                                ▼
                 Core Method 1: Behavioral Cloning (BC)
 ┌───────────────────────────────────────────────────────────┐
 │ Treat imitation as supervised learning                    │
 │ Train πθ(s) → aE using dataset D                          │
 │ Discrete: cross-entropy loss                              │
 │ Continuous: mean squared error                            │
 │ Advantages: simple, offline, safe                         │
 └───────────────────────────────────────────────────────────┘
                                │
                                ▼
           Key BC Problem: Covariate / Distribution Shift
 ┌───────────────────────────────────────────────────────────┐
 │ Trained only on expert states                             │
 │ When deployed, policy errors lead to unseen states        │
 │ → Poor decisions → more drift → compounding failure       │
 │ BC cannot recover or improve beyond expert                │
 └───────────────────────────────────────────────────────────┘
                                │
                                ▼
                    Fixing BC: DAgger (Idea)
 ┌───────────────────────────────────────────────────────────┐
 │ Let policy act, collect mistakes                          │
 │ Ask expert for correct action                             │
 │ Add to dataset and retrain                                │
 │ → brings training data closer to deployment distribution  │
 └───────────────────────────────────────────────────────────┘
                                │
                                ▼
       Core Method 2: Inverse Reinforcement Learning (IRL)
 ┌───────────────────────────────────────────────────────────┐
 │ Learn the “why” behind actions → infer hidden reward R*   │
 │ Expert assumed optimal                                     │
 │ Solve inverse problem: πE ≈ optimal for R*                 │
 │ After reward recovered → run normal RL to learn policy     │
 │ Benefits: generalization, interpretability, improve expert │
 └───────────────────────────────────────────────────────────┘
                                │
                                ▼
                Core Method 3: Apprenticeship Learning
 ┌───────────────────────────────────────────────────────────┐
 │ Iteratively improve policy via comparing to expert        │
 │ Match feature expectations φ(s)                           │
 │ Reweights reward → optimize → evaluate → repeat           │
 │ Goal: perform at least as well as expert                  │
 │ Often implemented via IRL (e.g., GAIL conceptually)       │
 └───────────────────────────────────────────────────────────┘
                                │
                                ▼
           Role of IL within broader RL landscape
 ┌───────────────────────────────────────────────────────────┐
 │ When IL is useful:                                        │
 │ - Reward hard to design                                   │
 │ - Unsafe or costly to explore                             │
 │ - Sparse reward tasks                                     │
 │ IL + RL hybrid: BC warm-start → RL fine-tune beyond expert│
 │ Limitations: need expert, demos may be suboptimal         │
 └───────────────────────────────────────────────────────────┘
                                │
                                ▼
                   Final Takeaway (Chapter Summary)
 ┌───────────────────────────────────────────────────────────┐
 │ IL bypasses reward engineering & risky exploration         │
 │ BC learns “what,” IRL learns “why,” apprenticeship learns  │
 │ “how to get as good as expert.”                           │
 │ IL often combined with RL for best performance.           │
 └───────────────────────────────────────────────────────────┘
```
