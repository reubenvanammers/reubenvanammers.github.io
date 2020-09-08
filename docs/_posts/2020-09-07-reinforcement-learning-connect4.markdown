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

# AlphaGo

# Connect 4

# Implementation

# Results

