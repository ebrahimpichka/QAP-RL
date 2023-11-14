# Solving Quadratic Assignemt Problem using Deep Reinforcement Learning

This reposiroty contains an **unofficial** implemnetation of the model proposed in [**Bagga, P. S., & Delarue, A. (2023). Solving the quadratic assignment problem using deep reinforcement learning**.](https://arxiv.org/abs/2310.01604) for solving QAP via Deep RL.

The paper formulizes QAP as a sequential decision problem (i.e. an MDP) by selecting locations and their assigned facilitiy sequentially and proposes a **Double Pointer Network** architecrure to represent the actor to **learn to solve** the problem using the in the Advantage Actor-Critic algorithm .