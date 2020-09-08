---
layout: post
title:  "Enumerations and Expectations: Calculating DnD Advantage via Combinatorics"
date:   2019-10-11 9:16:00 +1000
categories: blog
---
This post describes some methods of calculating expected values of advantage/disadvantage in Dungeons and Dragons. If you know what this is, you can skip the context and get to the mathy bits. 


# Context
In the roleplaying game Dungeon and Dragons (5E), dice are often rolled to determine the determine the results of actions, both inside and outside of combat. Often, due to the context of the situation, dice are rolled with advantage or disadvantage to reflect that an action is more or less likely to succeed respectively. In the case of advantage, two die (which are D20, a dice with 20 sides) are rolled instead of one, and the highest number is chosen. In the case of disadvantage, two dice are again rolled, but this time the lower number is chosen instead. Higher numbers are generally better, and passing thresholds will have positive effects. The question is: exactly how good is advanage?

# Goal

What we are trying to do is to calculate the exact effect of that advantage, analytically. For a normal D20 and two dice rolls, this could be relatively easily calculated via simulation or enumeration over all possibilities. This, however, only accounts for our standard version of DnD. What if we wanted to use a variant with a 100 sided dice? Therefore, in order to spice things up, we are going to use an arbitrary $k$ sided die. In addition, for an extended definition of advantage, we're going to roll $$n$$ die and only use the highest value seen. For disadvantage, we roll $$n$$ die and take the lowest. Note that we get the version used in the standard DnD rules if we use $$k=20$$ and $$n=2$$. 


# Math
To solve this we are going to use some relatively simple combinatorics in order to get an analytic value. First, we will define what we want mathematicaly. After this, we will attempt to find recurrence relations (a common trick for many combinatorics problems) in order to ultimately find a solution. 


To do this, we calculate the expected value of possible rolls $ \bar{d_e}(n,k)$ for n dice of k sides:

$$\bar{d_e}(n,k) = \frac{1}{k^n} \sum_{d_1,d_2,..d_n = 1}^k \text{max}(d_1,..,d_n)$$


To help simplify this expression and create a recurrence relation, first we define the difference between the maximum possible dice roll and the actual dice that were rolled:
 $$\text{maxdiff}_k(d_1,..,d_n) = k - \text{max}(d_1,..,d_n)$$. 

We can see that $$\text{maxdiff}_{k+1}(d_1,..,d_n)-\text{maxdiff}_{k}(d_1,..,d_n) = 1$$,

as the maximum value differs by 1 and the dice rolled were the same. 

We also can see that 
$\text{maxdiff}_k(d_1,..,k,..,d_n) = 0$,
i.e. the max difference equals zero when one of the dice is the maximum value $k$. 

Now we can substitute this into the total sum of dice rolls:

$$
\begin{aligned}


 \sum_{d_1,d_2,..d_n = 1}^k \text{max}(d_1,..,d_n) &  = \sum_{d_1,d_2,..d_n = 1}^k k - \text{maxdiff}_k(d_1,..,d_n) \\

& = \sum_{d_1,d_2,..d_n = 1}^k k -  \sum_{d_1,d_2,..d_n = 1}^k\text{maxdiff}_k(d_1,..,d_n) \\

& = k \sum_{d_1,d_2,..d_n = 1}^k -  \sum_{d_1,d_2,..d_n = 1}^k\text{maxdiff}_k(d_1,..,d_n) \\

& =k \cdot k^{n} - \sum_{d_1,d_2,..d_n = 1}^{k-1}\text{maxdiff}_k(d_1,..,d_n) - \sum_{\text{at least one dice is k}}^k\text{maxdiff}_k(d_1,..,d_n) \\

& =k^{n+1}- \sum_{d_1,d_2,..d_n = 1}^{k-1} \text{maxdiff}_k(d_1,..,d_n) - 0 \\

& =k^{n+1} -  \sum_{d_1,d_2,..d_n = 1}^{k-1} \left( \text{maxdiff}_{k-1}(d_1,..,d_n)+1 \right) \\

& =k^{n+1} - (k-1)^{n} - \sum_{d_1,d_2,..d_n = 1}^{k-1}\text{maxdiff}_{k-1}(d_1,..,d_n)  \\
& =k^{n+1} - (k-1)^{n} + (k-1)^{n+1} - (k-1)^{n+1} - \sum_{d_1,d_2,..d_n = 1}^{k-1}\text{maxdiff}_{k-1}(d_1,..,d_n)  \\
& =k^{n+1} - (k-1)^{n} - (k-1)^{n+1} + \sum_{d_1,d_2,..d_n = 1}^{k-1}\text{max}(d_1,..,d_n)  \\

\end{aligned}
$$

Now we have our recurrence relation over k! The hardest bit is over. Now we just have to telescope out the recurrence and simplify:

$$
\begin{aligned}
\sum_{d_1,d_2,..d_n = 1}^k \text{max}(d_1,..,d_n) & =k^{n+1} - (k-1)^{n} - (k-1)^{n+1} + \sum_{d_1,d_2,..d_n = 1}^{k-1}\text{max}(d_1,..,d_n) \\
 &= k^{n+1} - (k-2)^{n+1} - (k-1)^{n} - (k-1)^{n+1} + \left ((k-1)^{n+1} - (k-2)^{n}  +  \sum_{d_1,d_2,..d_n = 1}^{k-2}\text{max}(d_1,..,d_n) \right) \\
 &= k^{n+1} - (k-2)^{n+1} - (k-1)^{n}   - (k-2)^{n}  + \sum_{d_1,d_2,..d_n = 1}^{k-2}\text{max}(d_1,..,d_n) 

  \end{aligned}
 $$
 
We can see that the factors to the power of n+1 get cancelled out, meaning that only the values at k and 1 remain:

 $$
\begin{aligned}
  \sum_{d_1,d_2,..d_n = 1}^k \text{max}(d_1,..,d_n) &= k^{n+1}- 1^{n+1} - (k-1)^{n}   - (k-2)^{n} - ... (k-j)^{n} - ... -1^n  + \sum_{d_1,d_2,..d_n = 1}^{1}\text{max}(d_1,..,d_n) \\
    &= k^{n+1}   - \sum_{i=1}^{k-1}  i^n  -1^{n+1}+ \sum_{d_1,d_2,..d_n = 1}^{1}\text{max}(d_1,..,d_n)

 \end{aligned}

 $$


With this, and the fact that $\sum_{d_1,d_2,..d_n = 1}^{1}\text{max}(d_1,..,d_n) =1 =1^{n+1}$, we can see that 

$$\sum_{d_1,d_2,..d_n = 1}^k \text{max}(d_1,..,d_n) = k^{n+1} - \sum_{i=1}^{k-1} i^n$$


Phew! That simplified nicely! This gives us the relatively compact expression, by dividing both sides by $k^n$:

$$\bar{d_e}(n,k) = k- \frac{1}{k^n}\sum_{i=1}^{k-1} i^n$$

For the case of the two dice having 20 sides each in DnD, we see that the average roll is:


$$20-\frac{1}{400}\sum_{i=1}^{19}i^2 = 13.825$$

This means that on average, advantage gives $13.825 - 10.5 = 3.325$ dice pips higher than a standard dice roll. Due to symmetry, it is clear that this is the same as the penalty acquired from disadvantage, meaning that the average roll from disadvantage is $10.5 - 3.325 =  7.175$.

# Generalizing Further

Another possible generalization is to not pick the highest possible dice when rolling advantage. Rather, drop the lowest valued dice, and keep the higher valued ones, and sum them up. We can see an example of this during 5E character creation, where one method of generating character statistics is to roll 4 6 sided die, and sum the values of the highest 3 values.

I didn't have much success in generalizing to this extent, due to the more complex interplay between dice. In general, generating series of 3 variables tends to be more complicated than two as recurrence relations get more complex. An approach to do this would be the generation of an appropriate 2 variable generating function, although I haven't had much luck with doing this - although I'd be keen to hear if anyone knows a good approach to this. 

