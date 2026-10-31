## Graph CBFs (GCBF+)
GCBF+ learns a  neural Graph Control Barrier Function that certifies collision avoidance for a multi-agent swarm of any size, using only local neighbor and LiDAR information.
It uses a graph attention network trained on just 8 agents, with a loss that imitates a CBF-QP controller so that safety and goal-reaching don't conflict.
It scales to 1024 agents in simulation and was demonstrated on a real Crazyflie drone swarm.

## Deep Equivariant GCBF+ (EGCBF+)
EGCBF+ keeps the GCBF+ framework but makes the CBF G-invariant and the policy G-equivariant, so it exploits the geometric symmetries of the problem.
It does this with an Equivariant Graphormer that canonicalizes each node's local frame, which makes it data-efficient and works with any compatible group.
