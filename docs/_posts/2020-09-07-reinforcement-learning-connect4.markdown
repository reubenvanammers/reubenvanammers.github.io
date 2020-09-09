---
layout: post
title:  "Reinforcement Learning with Connect 4"
date:   2020-09-07 11:36:00 +1000
categories: blog
---
Recently, I've been doing some work with reinforcement learning with two player zero sum games. These games, which include games like Chess and Go, are some of the most commonly played competitive games. I used a couple of differrent techniques from Reinforcement learning, such as Q learning and the algorithm behind AlphaGo.

In this post, I'll be talking a bit about techniques in reinforcemnt learning, an overview of AlphaGo, how I implemented an algorithm similar to it, and what the results were. 

# Reinforcement Learning
If you're vaguely familiar with Machine Learning, you've probably heard of Supervised Learning and Unsupervised Learning. If you know a bit more, you've probably also heard of another field that is distinct from these two: Reinforcement Learning. In Reinforcement Learning, the aim is for an agent to somehow maximise their value function while within an environment. The environment may have unkown or know characteristics, but essentially the aim is to effectively get the agent to learn how it should act in order to get an optimal score, or reward,  whatever it may be. The environment might be a maze, where it has to get out within a minimum amount of time, or it might be a self-driving car, for example. One other common task is game play, whether single or multiplayer games.

Multiplayer games, like the type that is being talked about in this post, can be even trickier than most reinforcement learning tasks. The environment not only is inherently non-static, in that taking the same action from a given won't necessary give you the same result, there is an adversarial agent which attempts to maximize it's own score, and that agent might be different in different scenarios. Nevertheless, there has been quite a lot of progress in recent years, including DeepMind's AlphaGo/AlphaGoZero/AlphaZero.

# Q Learning
I'm not going to go too in depth into Q learning here: see [wikipedia](https://en.wikipedia.org/wiki/Q-learning) for more details.

Q learning is one of the more basic reinforcemnt learning algorithms, and can be used to decent success for a variety of tasks, for example playing Atari games. Nevertheless, it's generally not made for two player zero sum games, but I implemented a version anyway which I could compare to. 

In essence, Q learning attempts to attach a "quality value" to each state action pair, $Q(s,a)$, where s is the state of the environment and $a$ is the action. For discrete environments, this can be viewed as a table where each value of $s$ and $a$ has this attached value readily available. By iterating on an initial guess of what these quality values should be, increasing the score of positions and actions which lead to positive rewards, slowly reaching their actual value. We can see this in the update rule for Q learning:

$$Q^{\text{new}(s_t,a_t) = Q(s_{t},a_{t}) + \alpha \cdot (r + \gamma \cdot \text{argmax_a}Q(s_{t+1},a)-Q(s_{t},a_{t}))$$

where r is the reward given from action $a_t$ which moves the environment from state $s_t$ to state $s_{t+1}$, and $\gamma$ is a discount factor. 

In essence, Q-learning attempts to see how good a move is by seeing how good the state its moving into is, and how much a reward it gets from doing so. Initially, its guess at how good the state its going into isn't great (or why would you need this algorithm in the first place), but by successive iterations can usually get to a reasonable level of accuracy.

If you have a more complicated system whereby its impractical to have each state action pair in a table, then essentially you need to have a function to approximate the value of Q - this can be a neural network or the like, and is what was used during implementation. 

# Monte Carlo Tree Search

Before we talk about how AlphaGo works, I'll discuss one of the key components, Monte Carlo Tree search, or MCTS for short. The "Monte Carlo" component of the name refes to the random element of the algorithm, while tree search refers to (you guessed it) that it uses a game tree in order to evaluate moves.

MCTS is based on set on a relatively simple set of principles, although they can get more complicated in practise: Selection, Expansion, Scoring and Backup. Each of these is done a fixed number $(M)$ times, after which a move is selected and played. I'll go over these in more detail. 

## Selection

There is a couple of different of ways to select the moves that are used. One of the most common ways that is used in MCTS is called the Upper Confidence Bound for Trees algorithm, or UCT for short. AlphaZero uses a variant called Polynomial Upper Confidence Bound for Trees, or PUCT. The aim of these methods is to balance exploitation and exploration, a common problem in reinforcement learning algorithms. We want to choose moves that we think are good, without neglecting other possibilities that may end up being better. 

To do this, we choose the action that satisfies the search function containing both exploration and exploitation related quantities:

$$a = \argmax_a(Q(s,a) + U(s,a))$$

$Q(s,a)$ relates to the quality of the move at a specific node. It is calculated by calculating the average value of all moves below it in the tree - in other words, all moves played from this point. We'll talk about how we calculate value later. 

$U(s,a)$ is the exploration parameter. It is defined by 

$$U(s,a) = c P(s,a) \frac{\sum_bN(s,b)}{1+N(s,a)},$$

where c is a n exploration constant, $N(s,a)$ is the number of times action $a$ has been taken from state $s$, and $P(s,a)$ is the prior probability. The prior can be determined multiple ways - it could be programmed manually, similarly to how many powerful chess engines or DeepBlue were, or it could be calculated in other methods (like a neural network)! We'll again come back to this.

## Expansion and Evalution
Once a move has been chosen, we then expand, or simulate it. Once we've simulated the move, we score it. If the games has been finished, this is easy - assign a value $v$ of 1 for a win, 0 for a draw, and -1 for a loss. Otherwise, we need to determine another method for doing so. The traditional way to do this in Monte Carlo Tree Search was to randomly play games from this point, and see what proportion of the games were won - this became the value of the move. In AlphaGo, this was done via the value head of a Neural Network. 


## Backup

Once we have the value $v$ of a move, we pass this back up the tree. Every node above this get an additional data point of the value $v$ and increments the value of $N$ by one. This happens all the way back up to the root node of the tree. With this, we can keep an up to date determination of the quality of a set of nodes.



## Play

Once these steps have been done $M$ times, it is time to choose the final move to actually play from the state $s_0$. This can be done by a distribution that is proportional to the number of times that node has travelled down:

$$\pi(a)|s_0) = \frac{N(s_0,a)^{\frac{1}{\tau}}}{\sum_bN(s_0,b)^\frac{1}{\tau}}$$

$$\tau$$ is a parameter that controls the level of exploration. In practise, we usually want to vary this parameter depending on the situation - higher values of the parameter increase exploration values, with a value of $\tau=1$ ( mean that all moves get weighted proportional to how much they were explored in the search tree. On the other hand, if you want to exploit and choose the best move possible (at least from the trees perspective) you set $\tau$ to be very low such that the most explored move will almost always be chosen.

# AlphaGo

# Connect 4

# Implementation

# Results

