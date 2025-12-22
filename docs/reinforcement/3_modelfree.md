# Chapter 3: Model-Free Policy Evaluation: Learning the Value of a Fixed Policy

In Dynamic Programming, value functions are computed using a known model of the environment. In reality, however, the model is almost always unknown. This necessitates a shift to Model-Free Reinforcement Learning, where the agent must learn the values of states and actions solely from direct experience (i.e., collecting trajectories of states, actions, and rewards). The goal is to estimate the value function $V^\pi(s)$ or $Q^\pi(s,a)$ for a given policy $\pi$ using data of the form:

$$
s_0, a_0, r_1, s_1, a_1, r_2, s_2, \dots
$$

The true value of a state under policy $\pi$ is still defined by the expected return:

$$
V^\pi(s) = \mathbb{E}_\pi[G_t | s_t = s]
$$

but the agent must approximate this expectation using sampled experience.

Model-Free methods can be divided into two main categories based on how they estimate returns:

1. Monte Carlo (MC) methods: learn from complete episodes by averaging returns.
2. Temporal Difference (TD) methods: learn from incomplete episodes by bootstrapping from existing estimates.
 
## Monte Carlo Policy Evaluation

MC methods are the simplest approach to model-free evaluation. The core idea is that since the true value function $V^\pi(s)$ is the expected return, we can approximate it by simply averaging the observed returns ($G_t$) from many episodes that start at state $s$.

$$
V^\pi(s) \approx \text{Average of observed returns } G_t \text{ starting from } s
$$

### Key Properties of MC
1.  Episodic Requirement: MC can only be applied to episodic MDPs. An episode must terminate ($s_T$) to calculate the full return $G_t$.
2.  Model-Free and Markovian Assumption: MC makes no assumption that the system is Markov in the observable state features. It merely averages the outcome of executing a policy.


We can maintain the value estimates $V(s)$ using counts and sums, or through incremental updates.

#### A. First-Visit vs. Every-Visit MC
When computing the return $G_t$ for a state $s$ in a single trajectory, a state might be visited multiple times.

* First-Visit MC: The return $G_t$ is used to update $V(s)$ only the first time state $s$ is visited in an episode.
    * Properties: First-Visit MC is an unbiased estimator of $V^\pi(s)$. It is also consistent (converges to the true value as data $\rightarrow \infty$) by the Law of Large Numbers.
* Every-Visit MC: The return $G_t$ is used to update $V(s)$ every time state $s$ is visited in an episode.
    * Properties: Every-Visit MC is a biased estimator because multiple updates within the same episode are correlated. However, it is also consistent and often exhibits better Mean Squared Error (MSE) due to utilizing more data.

#### B. Incremental Monte Carlo
For computational efficiency and to avoid storing all returns, MC updates can be performed incrementally using a running average. This looks like a standard learning update:

$$
V(s) \leftarrow V(s) + \alpha \left[ G_t - V(s) \right]
$$

Where:

* $G_t$: The actual observed return (our target).
* $V(s)$: Our current estimate (our old value).
* $\alpha$: The learning rate ($\alpha \in (0, 1]$), which can be fixed or decayed.

Consistency Guarantee: For incremental MC to guarantee convergence to the True Value ($V^\pi$), the learning rate $\alpha_t$ (which may be $1/N(s)$ or a fixed constant) must satisfy the following conditions:

1.  The sum of all learning rates for state $s$ must diverge: $\sum_{t=1}^{\infty} \alpha_t(s) = \infty$
2.  The sum of the squared learning rates must converge: $\sum_{t=1}^{\infty} \alpha_t(s)^2 < \infty$

 

## Temporal Difference (TD) Learning

While MC uses the full return $G_t$, TD learning is the fundamental shift in policy evaluation. It retains the concept of the incremental update but changes the target, introducing a technique called bootstrapping.

### Bootstrapping: The Core Idea

Bootstrapping means updating a value estimate using another value estimate. In the context of Policy Evaluation, TD methods use the estimated value of the *next* state, $V(s_{t+1})$, to update the value of the *current* state, $V(s_t)$. The standard TD algorithm is TD(0) (or one-step TD).

### The TD(0) Update Rule
The TD(0) update replaces the full return $G_t$ with the TD Target ($r_t + \gamma V(s_{t+1})$):

$$
V(s_t) \leftarrow V(s_t) + \alpha \left[ \underbrace{r_{t+1} + \gamma V(s_{t+1})}_{\text{TD Target}} - V(s_t) \right]
$$

The term inside the brackets is the TD Error ($\delta_t$):
$$
\delta_t = (r_{t+1} + \gamma V(s_{t+1})) - V(s_t)
$$
This error is the difference between the estimated value of the current state and a better, bootstrapped estimate of that value.

### TD vs. Monte Carlo

The distinction between TD and MC centers on what is used as the target value:

| Feature | Monte Carlo (MC) | Temporal Difference (TD) |
| :--- | :--- | :--- |
| Target | $G_t$ (Full observed return to episode end) | $r_{t+1} + \gamma V(s_{t+1})$ (One-step return + estimated future value) |
| Bootstrapping | No (waits until episode end) | Yes (uses $V(s_{t+1})$) |
| Bias | Unbiased (First-Visit MC) | Biased (because $V(s_{t+1})$ is an estimate) |
| Variance | High Variance (Return $G_t$ is a sum of many random steps) | Low Variance (TD target depends on only one random reward/next state) |
| Convergence | Consistent (converges to true $V^\pi$) | TD(0) converges to true $V^\pi$ in the tabular case |

TD methods generally have a desirable trade-off, accepting a small bias in exchange for significantly lower variance. This often makes them more computationally and statistically efficient in practice. TD(0) is applicable to non-episodic (continuing) tasks, overcoming one of the major limitations of Monte Carlo.


> ## Example Setup
>
> ### Parameters
> * States ($S$): $s_A, s_B, s_C$
> * Discount Factor ($\gamma$): $0.9$
> * Learning Rate ($\alpha$): $0.5$ (Used for TD updates)
> * Initial Value Estimates ($V_0$): $V(s_A)=0, V(s_B)=0, V(s_C)=0$
>
> ### Episodes and Returns
>
> The full return ($G_t$) is calculated for every visit in every episode:
>
> | Episode (E) | Trajectory (State $\xrightarrow{r}$ Next State) | Visit Time ($t$) | State ($s_t$) | Full Return ($G_t$) |
> | :--- | :--- | :--- | :--- | :--- |
> | E1 | $s_A \xrightarrow{r=1} s_B \xrightarrow{r=0} s_C \xrightarrow{r=5} s_B \xrightarrow{r=2} \text{T}$ | 0 | $s_A$ | $\mathbf{6.508}$ |
> | | | 1 | $s_B$ (1st) | $\mathbf{6.12}$ |
> | | | 2 | $s_C$ | $\mathbf{6.8}$ |
> | | | 3 | $s_B$ (2nd) | $\mathbf{2.0}$ |
> | E2 | $s_A \xrightarrow{r=-2} s_C \xrightarrow{r=8} \text{T}$ | 0 | $s_A$ | $\mathbf{5.2}$ |
> | | | 1 | $s_C$ | $\mathbf{8.0}$ |
> | E3 | $s_B \xrightarrow{r=10} s_C \xrightarrow{r=-5} s_B \xrightarrow{r=1} \text{T}$ | 0 | $s_B$ (1st) | $\mathbf{6.31}$ |
> | | | 1 | $s_C$ | $\mathbf{-4.1}$ |
> | | | 2 | $s_B$ (2nd) | $\mathbf{1.0}$ |
>
>
> ## 1. First-Visit Monte Carlo (MC)
>
> Rule: Only the first return for a state in any given episode is used.
>
> ### A. Data Selection and Counts ($N(s)$)
>
> | State ($s$) | Returns Used ($G_t$) | Total Sum ($\sum G_t$) | Count ($N(s)$) |
> | :--- | :--- | :--- | :--- |
> | $s_A$ | $6.508$ (E1), $5.2$ (E2) | $11.708$ | 2 |
> | $s_B$ | $6.12$ (E1), $6.31$ (E3) | $12.43$ | 2 |
> | $s_C$ | $6.8$ (E1), $8.0$ (E2), $-4.1$ (E3) | $10.7$ | 3 |
>
> ### B. Final Estimates ($V(s) = \sum G_t / N(s)$)
>
> $
> V(s_A) = \frac{11.708}{2} = \mathbf{5.854} \\
> V(s_B) = \frac{12.43}{2} = \mathbf{6.215} \\
> V(s_C) = \frac{10.7}{3} = \mathbf{3.567}
> $
>
>
> ## 2. Every-Visit Monte Carlo (MC)
>
> Rule: The return from every time a state is encountered in any episode is used.
>
> ### A. Data Selection and Counts ($N(s)$)
>
> | State ($s$) | Returns Used ($G_t$) | Total Sum ($\sum G_t$) | Count ($N(s)$) |
> | :--- | :--- | :--- | :--- |
> | $s_A$ | $6.508, 5.2$ | $11.708$ | 2 |
> | $s_B$ | $6.12, 2.0, 6.31, 1.0$ | $15.43$ | 4 |
> | $s_C$ | $6.8, 8.0, -4.1$ | $10.7$ | 3 |
>
> ### B. Final Estimates ($V(s) = \sum G_t / N(s)$)
>
> $$
> V(s_A) = \frac{11.708}{2} = \mathbf{5.854} \\
> V(s_B) = \frac{15.43}{4} = \mathbf{3.858} \\
> V(s_C) = \frac{10.7}{3} = \mathbf{3.567}
> $$
>
>
> ## 3. Temporal Difference (TD(0))
>
> Rule: The value is updated after every step using the TD Target ($r_{t+1} + \gamma V(s_{t+1})$) and the learning rate $\alpha$. The updated $V(s)$ estimates are carried over to the next step and episode.
>
> ### A. Step-by-Step TD Calculation Summary
>
> | Step | Transition | Old $V(s_t)$ | New $V(s_t)$ | $V(s_A)$ | $V(s_B)$ | $V(s_C)$ |
> | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
> | E1-1 | $s_A \xrightarrow{r=1} s_B$ | 0 | 0.500 | 0.500 | 0.000 | 0.000 |
> | E1-2 | $s_B \xrightarrow{r=0} s_C$ | 0 | 0.000 | 0.500 | 0.000 | 0.000 |
> | E1-3 | $s_C \xrightarrow{r=5} s_B$ | 0 | 2.500 | 0.500 | 0.000 | 2.500 |
> | E1-4 | $s_B \xrightarrow{r=2} \text{T}$ | 0.0 | 1.000 | 0.500 | 1.000 | 2.500 |
> | E2-1 | $s_A \xrightarrow{r=-2} s_C$ | 0.500 | 0.375 | 0.375 | 1.000 | 2.500 |
> | E2-2 | $s_C \xrightarrow{r=8} \text{T}$ | 2.500 | 5.250 | 0.375 | 1.000 | 5.250 |
> | E3-1 | $s_B \xrightarrow{r=10} s_C$ | 1.000 | 7.863 | 0.375 | 7.863 | 5.250 |
> | E3-2 | $s_C \xrightarrow{r=-5} s_B$ | 5.250 | 3.663 | 0.375 | 7.863 | 3.663 |
> | E3-3 | $s_B \xrightarrow{r=1} \text{T}$ | 7.863 | 4.431 | 0.375 | 4.431 | 3.663 |
>
> ### B. Final Estimates ($V_{TD(0)}$)
>
> > $$
> > V(s_A) = \mathbf{0.375} \\
> > V(s_B) = \mathbf{4.431} \\
> > V(s_C) = \mathbf{3.663}
> > $$
>
>
> ## Comparison of Results
>
> | State | First-Visit MC | Every-Visit MC | TD(0) ($\alpha=0.5, \gamma=0.9$) | Note |
> | :--- | :--- | :--- | :--- | :--- |
> | $s_A$ | 5.854 | 5.854 | 0.375 | TD heavily penalized $s_A$ in E2 (Target 0.25), while MC averaged the full observed high returns. |
> | $s_B$ | 6.215 | 3.858 | 4.431 | TD's result falls between the two MC methods, demonstrating a quicker convergence due to bootstrapping. |
> | $s_C$ | 3.567 | 3.567 | 3.663 | All methods are close for $s_C$. |
>
> This comparison illustrates the bias-variance trade-off:
> * MC uses the sample return ($G_t$), which has high variance but is an unbiased target (First-Visit).
> * TD uses a bootstrapped estimate ($r + \gamma V(s')$), which has lower variance but introduces bias by relying on an estimated successor value.


## Model Free Prediction Mental Map  

``` text
            Model-Free Prediction
     (Policy Evaluation without Model P or R)
                        │
                        ▼
                Goal: Estimate
       ┌───────────────────────────────────┐
       │ State Value: Vπ(s)                │
       │ Action Value: Qπ(s,a)             │
       └───────────────────────────────────┘
                        │
              Using Sampled Experience
        (s₀,a₀,r₁,s₁,a₁,r₂,... from π)
                        │
                        ▼
            Two Families of Methods
    ┌───────────────────────────────┬───────────────────────────────┐
    │ Monte Carlo (MC)              │ Temporal Difference (TD)      │
    │ "Learn from full episodes"    │ "Learn step-by-step"          │
    └───────────────────────────────┴───────────────────────────────┘
                        │                           │
                        │                           │
                        ▼                           ▼
             Monte Carlo (MC)              Temporal Difference (TD)
      ┌─────────────────────────┐       ┌────────────────────────────┐
      │ Needs full episodes     │       │Works on incomplete episodes│
      │ No bootstrapping        │       │Uses bootstrapping          │
      │ High variance           │       │Low variance                │
      │ Unbiased (first visit)  │       │Biased                      │
      └─────────────────────────┘       └────────────────────────────┘
                        │                           │
                        │                           │
        ┌───────────────┴───────────────┐           │
        │                               │           │
        ▼                               ▼           ▼
 First-Visit MC                  Every-Visit MC     TD(0) Update Rule
 (One update per episode          (Multiple updates │ V(s) ← V(s) +
  per state)                      per episode)      │ α[ r + γV(s') − V(s) ]
                        │                           │
                        │                           │
                        └──────────┬────────────────┘
                                   │
                                   ▼
                        Comparison (Bias–Variance)
               ┌─────────────────────────────────────────┐
               │ MC: Unbiased, High variance             │
               │ TD: Biased, Lower variance              │
               │ MC: Not bootstrapping                   │
               │ TD: Bootstraps using V(s’)              │
               │ MC: Episodic only                       │
               │ TD: Works for continuing tasks          │
               └─────────────────────────────────────────┘
                                   │
                                   ▼
                      Outcome: Learned Value Function
               ┌───────────────────────────────────────┐
               │ Vπ(s) or Qπ(s,a) from real experience │
               │ (No model of environment required)    │
               └───────────────────────────────────────┘

```