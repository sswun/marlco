Multiwalker
This environment is part of the SISL environments. Please read that page first for general information.

Import

from pettingzoo.sisl import multiwalker_v9

Actions

Continuous

Parallel API

Yes

Manual Control

No

Agents

agents= ['walker_0', 'walker_1', 'walker_2']

Agents

3

Action Shape

(4,)

Action Values

(-1, 1)

Observation Shape

(31,)

Observation Values

[-inf,inf]

In this environment, bipedal robots attempt to carry a package placed on top of them towards the right. By default, the number of robots is set to 3.

Each walker receives a reward equal to the change in position of the package from the previous timestep, multiplied by the forward_reward scaling factor. The maximum achievable total reward depends on the terrain length; as a reference, for a terrain length of 75, the total reward under an optimal policy is around 300.

The environment is done if the package falls, or if the package goes beyond the left edge of the terrain. By default, the environment is also done if any walker falls. In all of these cases, each walker receives a reward of -100. The environment is also done if package falls off the right edge of the terrain, with reward 0.

When a walker falls, it receives an additional penalty of -10. If terminate_on_fall = False, then the environment is not done when the walker falls, but only when the package falls. If remove_on_fall = True, the fallen walker is removed from the environment. The agents also receive a small shaped reward of -5 times the change in their head angle to keep their head oriented horizontally.

If shared_reward is chosen (True by default), the agents’ individual rewards are averaged to give a single mean reward, which is returned to all agents.

Each walker exerts force on two joints in their two legs, giving a continuous action space represented as a 4 element vector. Each walker observes via a 31 element vector containing simulated noisy lidar data about the environment and information about neighboring walkers. The environment’s duration is capped at 500 frames by default (can be controlled by the max_cycles setting).

Observation Space
Each agent receives an observation composed of various physical properties of its legs and joints, as well as LIDAR readings from the space immediately in front and below the robot. The observation also includes information about neighboring walkers, and the package. The neighbour and package observations have normally distributed signal noise controlled by position_noise and angle_noise. For walkers without neighbors, observations about neighbor positions are zero.

This table enumerates the observation space:

Index: [start, end)

Description

Values

0

Hull angle

[0, 2*pi]

1

Hull angular velocity

[-inf, inf]

2

X Velocity

[-1, 1]

3

Y Velocity

[-1, 1]

4

Hip joint 1 angle

[-inf, inf]

5

Hip joint 1 speed

[-inf, inf]

6

Knee joint 1 angle

[-inf, inf]

7

Knee joint 1 speed

[-inf, inf]

8

Leg 1 ground contact flag

{0, 1}

9

Hip joint 1 angle

[-inf, inf]

10

Hip joint 2 speed

[-inf, inf]

11

Knee joint 2 angle

[-inf, inf]

12

Knee joint 2 speed

[-inf, inf]

13

Leg 2 ground contact flag

{0, 1}

14-23

LIDAR sensor readings

[-inf, inf]

24

Left neighbor relative X position (0.0 for leftmost walker) (Noisy)

[-inf, inf]

25

Left neighbor relative Y position (0.0 for leftmost walker) (Noisy)

[-inf, inf]

26

Right neighbor relative X position (0.0 for rightmost walker) (Noisy)

[-inf, inf]

27

Right neighbor relative Y position (0.0 for rightmost walker) (Noisy)

[-inf, inf]

28

Walker X position relative to package (0 for left edge, 1 for right edge) (Noisy)

[-inf, inf]

29

Walker Y position relative to package (Noisy)

[-inf, inf]

30

Package angle (Noisy)

[-inf, inf]

Arguments
multiwalker_v9.env(n_walkers=3, position_noise=1e-3, angle_noise=1e-3, forward_reward=1.0, terminate_reward=-100.0, fall_reward=-10.0, shared_reward=True,
terminate_on_fall=True, remove_on_fall=True, terrain_length=200, max_cycles=500)
n_walkers: number of bipedal walker agents in environment

position_noise: noise applied to neighbors and package positional observations

angle_noise: noise applied to neighbors and package rotational observations

forward_reward: reward received is forward_reward * change in position of the package

fall_reward: reward applied when an agent falls

shared_reward: whether reward is distributed among all agents or allocated individually

terminate_reward: reward applied to each walker if they fail to carry the package to the right edge of the terrain

terminate_on_fall: If True (default), a single walker falling causes all agents to be done, and they all receive an additional terminate_reward. If False, then only the fallen agent(s) receive fall_reward, and the rest of the agents are not done i.e. the environment continues.

remove_on_fall: Remove a walker when it falls (only works when terminate_on_fall is False)

terrain_length: length of terrain in number of steps

max_cycles: after max_cycles steps all agents will return done

Version History
v8: Replaced local_ratio, fixed rewards, terrain length as an argument and documentation (1.15.0)

v7: Fixed problem with walker collisions (1.8.2)

v6: Fixed observation space and made large improvements to code quality (1.5.0)

v5: Fixes to reward structure, added arguments (1.4.2)

v4: Misc bug fixes (1.4.0)

v3: Fixes to observation space (1.3.3)

v2: Various fixes and environment argument changes (1.3.1)

v1: Fixes to how all environments handle premature death (1.3.0)

v0: Initial versions release (1.0.0)

Usage
AEC
from pettingzoo.sisl import multiwalker_v9

env = multiwalker_v9.env(render_mode="human")
env.reset(seed=42)

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    else:
        # this is where you would insert your policy
        action = env.action_space(agent).sample()

    env.step(action)
env.close()

Parallel
from pettingzoo.sisl import multiwalker_v9

env = multiwalker_v9.parallel_env(render_mode="human")
observations, infos = env.reset()

while env.agents:
    # this is where you would insert your policy
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}

    observations, rewards, terminations, truncations, infos = env.step(actions)
env.close()

API
class pettingzoo.sisl.multiwalker.multiwalker.env(**kwargs)[source]
class pettingzoo.sisl.multiwalker.multiwalker.raw_env(*args, **kwargs)[source]
action_space(agent)[source]
Takes in agent and returns the action space for that agent.

MUST return the same value for the same agent name

Default implementation is to return the action_spaces dict

close()[source]
Closes any resources that should be released.

Closes the rendering window, subprocesses, network connections, or any other resources that should be released.

observation_space(agent)[source]
Takes in agent and returns the observation space for that agent.

MUST return the same value for the same agent name

Default implementation is to return the observation_spaces dict

observe(agent)[source]
Returns the observation an agent currently can make.

last() calls this function.

render()[source]
Renders the environment as specified by self.render_mode.

Render mode can be human to display a window. Other render modes in the default environments are ‘rgb_array’ which returns a numpy array and is supported by all environments outside of classic, and ‘ansi’ which returns the strings printed (specific to classic environments).

reset(seed=None, options=None)[source]
Resets the environment to a starting state.

state()[source]
State returns a global view of the environment.

It is appropriate for centralized training decentralized execution methods like QMIX

step(action)[source]
Accepts and executes the action of the current agent_selection in the environment.

Automatically switches control to the next agent.


Simple Spread
Warning

The environment pettingzoo.mpe.simple_spread_v3 has been moved to the new MPE2 package, and will be removed from PettingZoo in a future release. Please update your import to mpe2.simple_spread_v3.

This environment is part of the MPE environments. Please read that page first for general information.

Import

from pettingzoo.mpe import simple_spread_v3

Actions

Discrete/Continuous

Parallel API

Yes

Manual Control

No

Agents

agents= [agent_0, agent_1, agent_2]

Agents

3

Action Shape

(5)

Action Values

Discrete(5)/Box(0.0, 1.0, (5))

Observation Shape

(18)

Observation Values

(-inf,inf)

State Shape

(54,)

State Values

(-inf,inf)

This environment has N agents, N landmarks (default N=3). At a high level, agents must learn to cover all the landmarks while avoiding collisions.

More specifically, all agents are globally rewarded based on how far the closest agent is to each landmark (sum of the minimum distances). Locally, the agents are penalized if they collide with other agents (-1 for each collision). The relative weights of these rewards can be controlled with the local_ratio parameter.

Agent observations: [self_vel, self_pos, landmark_rel_positions, other_agent_rel_positions, communication]

Agent action space: [no_action, move_left, move_right, move_down, move_up]

Arguments
simple_spread_v3.env(N=3, local_ratio=0.5, max_cycles=25, continuous_actions=False, dynamic_rescaling=False)
N: number of agents and landmarks

local_ratio: Weight applied to local reward and global reward. Global reward weight will always be 1 - local reward weight.

max_cycles: number of frames (a step for each agent) until game terminates

continuous_actions: Whether agent action spaces are discrete(default) or continuous

dynamic_rescaling: Whether to rescale the size of agents and landmarks based on the screen size

Usage
AEC
from pettingzoo.mpe import simple_spread_v3

env = simple_spread_v3.env(render_mode="human")
env.reset(seed=42)

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    else:
        # this is where you would insert your policy
        action = env.action_space(agent).sample()

    env.step(action)
env.close()

Parallel
from pettingzoo.mpe import simple_spread_v3

env = simple_spread_v3.parallel_env(render_mode="human")
observations, infos = env.reset()

while env.agents:
    # this is where you would insert your policy
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}

    observations, rewards, terminations, truncations, infos = env.step(actions)
env.close()

API
class pettingzoo.mpe.simple_spread.simple_spread.raw_env(N=3, local_ratio=0.5, max_cycles=25, continuous_actions=False, render_mode=None, dynamic_rescaling=False)[source]


Simple Crypto
Warning

The environment pettingzoo.mpe.simple_crypto_v3 has been moved to the new MPE2 package, and will be removed from PettingZoo in a future release. Please update your import to mpe2.simple_crypto_v3.

This environment is part of the MPE environments. Please read that page first for general information.

Import

from pettingzoo.mpe import simple_crypto_v3

Actions

Discrete/Continuous

Parallel API

Yes

Manual Control

No

Agents

agents= [eve_0, bob_0, alice_0]

Agents

2

Action Shape

(4)

Action Values

Discrete(4)/Box(0.0, 1.0, (4))

Observation Shape

(4),(8)

Observation Values

(-inf,inf)

State Shape

(20,)

State Values

(-inf,inf)

In this environment, there are 2 good agents (Alice and Bob) and 1 adversary (Eve). Alice must sent a private 1 bit message to Bob over a public channel. Alice and Bob are rewarded +2 if Bob reconstructs the message, but are rewarded -2 if Eve reconstruct the message (that adds to 0 if both teams reconstruct the bit). Eve is rewarded -2 based if it cannot reconstruct the signal, zero if it can. Alice and Bob have a private key (randomly generated at beginning of each episode) which they must learn to use to encrypt the message.

Alice observation space: [message, private_key]

Bob observation space: [private_key, alices_comm]

Eve observation space: [alices_comm]

Alice action space: [say_0, say_1, say_2, say_3]

Bob action space: [say_0, say_1, say_2, say_3]

Eve action space: [say_0, say_1, say_2, say_3]

For Bob and Eve, their communication is checked to be the 1 bit of information that Alice is trying to convey.

Arguments
simple_crypto_v3.env(max_cycles=25, continuous_actions=False, dynamic_rescaling=False)
max_cycles: number of frames (a step for each agent) until game terminates

continuous_actions: Whether agent action spaces are discrete(default) or continuous

dynamic_rescaling: Whether to rescale the size of agents and landmarks based on the screen size

Usage
AEC
from pettingzoo.mpe import simple_crypto_v3

env = simple_crypto_v3.env(render_mode="human")
env.reset(seed=42)

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    else:
        # this is where you would insert your policy
        action = env.action_space(agent).sample()

    env.step(action)
env.close()

Parallel
from pettingzoo.mpe import simple_crypto_v3

env = simple_crypto_v3.parallel_env(render_mode="human")
observations, infos = env.reset()

while env.agents:
    # this is where you would insert your policy
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}

    observations, rewards, terminations, truncations, infos = env.step(actions)
env.close()

API
class pettingzoo.mpe.simple_crypto.simple_crypto.raw_env(max_cycles=25, continuous_actions=False, render_mode=None, dynamic_rescaling=False)[source]

