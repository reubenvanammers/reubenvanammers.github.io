---
layout: post
title:  "Reinforcement Learning with Connect 4"
date:   2020-09-07 11:36:00 +1000
categories: blog
---
Recently, I've been doing some work with reinforcement learning with two player zero sum games. These games, which include games like Chess and Go, are some of the most commonly played competitive games. I used a couple of different techniques from Reinforcement learning, such as Q learning and the algorithm behind AlphaGo.

You can have a look at the repo I used to do this [here](https://github.com/reubenvanammers/self_play_reinforcement_learning).

In this post, I'll be talking a bit about techniques in reinforcement learning, an overview of AlphaGo, how I implemented an algorithm similar to it in the game of Connect 4, and what the results were. 

## Reinforcement Learning
If you're vaguely familiar with Machine Learning, you've probably heard of Supervised Learning and Unsupervised Learning. If you know a bit more, you've probably also heard of another field that is distinct from these two: Reinforcement Learning. In Reinforcement Learning, the aim is for an agent to somehow maximize their value function while within an environment. The environment may have unknown or know characteristics, but essentially the aim is to effectively get the agent to learn how it should act in order to get an optimal score, or reward,  whatever it may be. The environment might be a maze, where it has to get out within a minimum amount of time, or it might be a self-driving car, for example. One other common task is game play, whether single or multiplayer games.

Multiplayer games, like the type that is being talked about in this post, can be even trickier than most reinforcement learning tasks. The environment not only is inherently non-static, in that taking the same action from a given won't necessarily give you the same result, but there is also an adversarial agent which attempts to maximize its own score, and that agent might be different in different scenarios. Nevertheless, there has been quite a lot of progress in recent years, including DeepMind's AlphaGo/AlphaGoZero/AlphaZero.

## Q Learning
I'm not going to go too in-depth into Q learning here: see [Wikipedia](https://en.wikipedia.org/wiki/Q-learning) for more details.

Q learning is one of the more basic reinforcement learning algorithms and can be used to decent success for a variety of tasks, for example playing Atari games. Nevertheless, it's generally not made for two player zero sum games, but I implemented a version anyway which I could compare to. 

In essence, Q learning attempts to attach a "quality value" to each state-action pair, $Q(s,a)$, where s is the state of the environment and $a$ is the action. For discrete environments, this can be viewed as a table where each value of $s$ and $a$ has this attached value readily available. By iterating on an initial guess of what these quality values should be, increasing the score of positions and actions which lead to positive rewards, slowly reaching their actual value. We can see this in the update rule for Q learning:

$$Q^{\text{new}}(s_t,a_t) = Q(s_t,a_t) + \alpha \cdot \left (r + \gamma \cdot \text{argmax}_aQ(s_{t+1},a)-Q(s_t,a_t) \right )$$

where r is the reward given from action $a_t$ which moves the environment from state $s_t$ to state $s_{t+1}$, and $\gamma$ is a discount factor. 

In essence, Q-learning attempts to see how good a move is by seeing how good the state it is moving into is, and how much a reward it gets from doing so. Initially, its guess at how good the state is going into isn't great (you wouldn't need this algorithm in the first place), but by successive iterations can usually get to a reasonable level of accuracy.

If you have a more complicated system whereby it's impractical to have each state-action pair in a table, then essentially you need to have a function to approximate the value of Q - this can be a neural network or the like, and is what was used during implementation. 

## Monte Carlo Tree Search

Before we talk about how AlphaGo works, I'll discuss one of the key components, Monte Carlo Tree search, or MCTS for short. The "Monte Carlo" component of the name refers to the random element of the algorithm, while tree search refers to (you guessed it) that it uses a game tree in order to evaluate moves.

MCTS is based on set on a relatively simple set of principles, although they can get more complicated in practice: Selection, Expansion, Scoring, and Backup. Each of these is done a fixed number $(M)$ times, after which a move is selected and played. I'll go over these in more detail. 

# Selection

There is a couple of different ways to select the moves that are used. One of the most common methods that is used in MCTS is called the Upper Confidence Bound for Trees algorithm, or UCT for short. AlphaZero uses a variant called Polynomial Upper Confidence Bound for Trees, or PUCT. The aim of these methods is to balance exploitation and exploration, a common problem in reinforcement learning algorithms. We want to choose moves that we think are good, without neglecting other possibilities that may end up being better. 

To do this, we choose the action that satisfies the search function containing both exploration and exploitation related quantities:

$$a = \text{argmax}_a(Q(s,a) + U(s,a))$$

$Q(s,a)$ relates to the quality of the move at a specific node. It is calculated by averaging the value of all moves below it in the tree - in other words, all moves played from this point. We'll talk about how we calculate value later. 

$U(s,a)$ is the exploration parameter. It is defined by 

$$U(s,a) = c P(s,a) \frac{\sqrt{\sum_bN(s,b)}}{1+N(s,a)},$$

where c is an exploration constant, $N(s,a)$ is the number of times action $a$ has been taken from state $s$, and $P(s,a)$ is the prior probability. The prior can be determined multiple ways - it could be programmed manually, similarly to how many powerful chess engines or DeepBlue were, or it could be calculated in other methods (like a neural network)! We'll again come back to this.

# Expansion and Evaluation
Once a move has been chosen, we then expand, i.e. simulate, it. Once we've simulated the move, we score it. If the game has been finished, this is easy - assign a value $v$ of 1 for a win, 0 for a draw, and -1 for a loss. Otherwise, we need to determine another method for doing so. The traditional way to do this in Monte Carlo Tree Search was to randomly play games from this point, and see what proportion of the games were won - this became the value of the move. In AlphaGo, this was done via the value head of a Neural Network. 


# Backup

Once we have the value $v$ of a move, we pass this back up the tree. Every node above this gets an additional data point of the value $v$ and increments the value of $N$ by one. This happens all the way back up to the root node of the tree. With this, we can keep an up-to-date determination of the quality of a set of nodes.



# Play

Once these steps have been done $M$ times, it is time to choose the final move to actually play from the state $s_0$. This can be done by a distribution that is proportional to the number of times that the node has been travelled down:

$$\pi(a)|s_0) = \frac{N(s_0,a)^{1/ \tau}}{\sum_bN(s_0,b)^{1/ \tau}}$$

$$\tau$$ is a parameter that controls the level of exploration. In practice, we usually want to vary this parameter depending on the situation - higher values of the parameter increase exploration values, with a value of $\tau=1$ meaning that all moves get weighted proportional to how much they were explored in the search tree. On the other hand, if you want to exploit and choose the best move possible (at least from the tree's perspective) you set $\tau$ to be very low such that the most explored move will almost always be chosen.

## AlphaGo

As stated previously, AlphaGo is an evolution of MCTS.
Surprisingly, it's not a massively different algorithm - it essentially just combines neural networks with the old MCTS algorithm, and gets surprisingly good results.
One reason why this may have taken so long is the computational requirements - both tree and neural network based algorithms require significant computational power, combine these as well as the need for self-play to generate training data means that huge amounts of resources are needed for complicated games at high levels of play. 

AlphaGo/AlphaGoZero work slightly differently, so I'll talk about AlphaZero due to it being more recent.
AlphaGoZero uses a single neural network with two separate heads in the MCTS algorithm.
One of these heads is a value head, which takes as input a board state and outputs the expected value of winning from that position.
The other head is the policy head, which gives a distribution of possible moves that can be played from that board state, favoring what it perceives to be better moves. 

In order to train this network, games are played by the network against itself using a random initial configuration of weights (hence the Zero in AlphaZero), and the board configurations are stored, along with the move played and the top level node probabilities from the tree in MCTS.
These weights can then be used to train the model, via minimizing the loss function:

$$l = (z-v)^2 - \boldsymbol{\pi}^Tlog(\mathbf{p})$$

In this equation, $z$ is the actual game winner, $v$ is the expected value of the winning, $\mathbf{p}$ is the prior move probabilities from the policy head and $\boldsymbol{\pi}$ is the actual move probabilities from the MCTS network.
In other words, the loss is the sum of the Mean Squared Error of winning, and the cross-entropy of the distributions of the policies.
It therefore attempts to balance getting a good estimate of both how good the position looks, and getting an accurate measure of what moves it will play.

During training, the workers playing the game get updated with the latest versions of the trained model.
With this, the game quality improves, and the network can further strengthen the weights.
This positive feedback loop between the Neural Network training and increasing accuracy via tree-search is the basis of the success of AlphaGo.

## Connect 4

When I attempted to implement something that was similar to the AlphaGo algorithm, I decided to mainly work with Connect 4.
If you're not familiar with the game, players take turns in placing their colored token in a $6*7$ grid. The token falls to the lowest empty place in the column.
The aim of the game, as might be expected from its name, is to get 4 of your own color in a straight line, whether that is a row, column, or diagonal. 

In terms of complexity, it's significantly more complicated than a game like Tic-Tac-Toe, which has a relatively trivial solution. Connect-4 is a solved game, although the solution is very complicated - it's been found that the player that goes can force a win, but any deviation from optimal play can allow the opponent to force a win. Nevertheless, it's complicated enough to allow complex emergent behavior, without being so complicated that it's infeasible to program and train with an individual's resources.

## Implementation

I implemented this system with a combination of PyTorch and Python - torch for the neural network evaluation and training, and python for the tree algorithm and all the surrounding bits and pieces.
Because of the high computational requirements of the system, I wrote it so that it could be parallelized.
There was a set of workers which played games against themselves using the latest version of the trained model, and these workers passed the results of the games back in a queue so that they could be stored in a memory object.
There was also another worker which used the memory object to train the model.

About the neural network itself: I used a set of convolutional layers with the two heads, i.e. the policy head and value head.
Initially, I started with 6 convolutional layers with a kernel size of 3, which is enough to span the $6 * 7$ Connect 4 board.
Eventually, I ended up moving to a 15 layer network, which seemed to improve performance after a while. I did notice that dropout layers on both of the heads of the network did seem to improve performance, so I ended up using them.
This was relatively surprising, as generally dropout layers don't seem to be used too often in reinforcement learning due to the fact that stability issues are often quite a big deal in RL - in other words, with so many self-reinforcing factors, adding in extra randomness like dropout layers can often reduce performance. 
I'm guessing that the presence of the tree acted as sufficient support to allow the dropout layer to help with overfitting without causing stability issues. The board state was transformed into a $3 * 6 * 7$ representation at the start of the tree - one of the channels represented the positions of pieces of player one, another channel represented the pieces of player two, and the third and final channel represented the positions of empty board spaces.
This was chosen to help the network infer potential lines of victory. 

The training worked as follows: 

Firstly, the self-play-workers played games with themselves for a few hundred games, enough to populate the memory object.
Then, training in earnest started - epoch 0 starts.
The update-worker continually trains the model, while the self-play-workers play a set number of games (say 500 games).
Once the games have been completed, the update-worker saves the current state of the model and the self-play-workers updated themselves to use this model.
Then, the new model was evaluated by playing against a fixed model.

This fixed model was crucial. Without having a fixed model to compare against, it is basically impossible to determine how your model is training, and whether or not your training pipeline is even working at all - it's something that I only put in after a while, despite how obvious it seems, and something that I'd put in right at the start for a similar project.
Initially, this fixed model was hardcoded - for example, a random agent which played moves randomly, or an agent which looks one move ahead to go for immediate wins/stop immediate losses.
After a while, as the models got stronger as they trained, I had to use trained, fixed models to properly evaluate performance over time. 



## Results

I managed to get some decent results after training the model.
To evaluate how good the model was, I scored the models via an [Elo system](https://en.wikipedia.org/wiki/Elo_rating_system), or at least a close analogue. In this rating system, a player that is 400 points above another player is expected to win roughly 90 % of the time.
I played the different models that I'd trained against each other and recorded the wins/losses/draws. I set the random player to have an  Elo of 0 - this was purely arbitrary, but as the elo rating system is comparative rather than absolute it requires an anchor of some sort.
I then minimized the cross-entropy loss of the Elo rating of the players assuming this 90% win rate for 400 points above. Doing this, I managed to get some elo scores:

* Random: 0
* Q learning, 6 hours: ~600
* One Move Look Ahead: 640
* 1 hour training, 6 layer NN with MCTS: 870
* 1 day training, 6 layer NN with MCTS: 1100
* 5 day training, 15 layer NN with MCTS: 1490
* (Update June 2021) 12 hour training, 20 layer NN with MCTS: 1930

There were some complexities with evaluating players against each other - there are parameters for each model which determine how much they should exploit vs explore.
With all parameters set to minimum randomness, the same outcome would occur every time, leading to a 100% winrate for a particular agent.
On the other hand, increasing exploration inherently decreases the strength of the player, especially as Connect 4 has quite a small action space compared to a game like chess.
Ultimately, I chose to use a set of parameters which were heavily focused on exploitation, with a small amount of randomness, for the purposes of evaluation. 


# Observations

A couple of miscellaneous observations that I noticed while doing this project: 

During training, I noticed that it was very important to ensure sufficient exploration was occurring - without it, the increase in performance was quite slow.
I skimmed over this above, and actually missed this the first time while reading the AlphaGoZero paper, but DeepMind added an extra layer of randomness to the top level root nodes priors.
They did this by generating a random Dirichlet distribution. 
The prior weight was generated to be 75% the prior generated by the neural network and 25% from the random Dirichlet distribution.
When I added this in, I noticed a significant increase in training speed. 

While the agents play their self-play games, I noticed that there was a very heavy influence on the middle column of the Connect 4 board.
This was to the extent that in high-level play, the first few moves tend to be alternate moves by the players in the middle column.
Initially, I thought that this might be a bug or weird emergent behavior, but I do think that this is close to optimal - because of the relatively small size of a Connect 4, and the middle column is the only column that can effectively *reach* the other end of the board (due to the winning length of 4).
High importance on the middle column makes sense, due to the fact that it can "reach" all the columns on the board.


I generally noticed that the first player had a significant advantage in self-play, suggesting that there is a significant advantage to playing the first move, as might be expected.
Generally, I noticed that the first player won about 60-70 percent of the time in self-play games. 

At lower levels of play, an agent might win relatively quickly, because one of the agents managed to overlook something, and the other agent could force a win.
At higher levels of play, this was much more uncommon, as the agents would overlook moves like this less.
Therefore, games often ended with most of the board being filled up, and one of the players being forced into a losing position. 

When I finished the training, I was still noticing improvements against a reference player as time went on.
I'd like to see it trained again with a more powerful GPU, to decrease the training times to more reasonable levels. 

# Update: 30/6/2021

Recently I updated the code to more efficiently make use of the GPU.
Previously, during game creation, the states were evaluated on the worker subprocess of that games.
This is the simplest option, but doing inference on only a single state is quite inefficient, due to the constant copying back and forth between the CPU/GPU.

To deal with this, I added the option to batch up work across different CPU threads, by using queues to bring the states to be evaluated to a single process, inferred the results as a batch, and sent the results back.
This significantly sped up training - this, and the updating from an old 970 to modern 3070 increased the game creation speed around 20 times, as well as using less GPU memory.

Some further speedups were achieved by parallelizing the game creation within the worker process and within the game itself.
The MCTS algorithm can be parallelized via the use of virtual loss, where a worker thread evaluating a tree updates the node values so that it looks like the result of the evaluation is a loss while still in progress, allowing multiple different moves to be attempted.
Consequently, I could use larger models, which trained in less time than previously, and could get better results, noted above in the results section.