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

 {% raw %}
  $$a^2 + b^2 = c^2$$
 {% endraw %}
$$a^2 + b^2 = c^2$$
