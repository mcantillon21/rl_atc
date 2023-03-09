import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


# configure states (taxing, gate, etc.)
# distance
# management of timesteps (observation)
# reset world
# get gym running for a sample

class Scenario(BaseScenario):
    # (flight #, origin, dest, flight time, depart time)
    # input = [('UA91', 'SFO', 'SEA', 2.5, 14:30), ('AA478', 'LHR', 'JFK', 7.5, 19:30)]
    def make_world(self, input):
        world = World()
        world.dim_c = 2 # variables to communicate between aircraft
        world.dim_p = 1 # dimension 1 
        num_agents = len(input)
        airports = []
        for inp in input:
            if inp[1] not in airports:
                airports.append(inp[1])
                airports.append(inp[1])
            if inp[2] not in airports:
                airports.append(inp[2])
                airports.append(inp[2])
        
        num_landmarks = 2*len(airports)
        world.collaborative = True
        # add aircraft agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'Aircraft %d' % i 
            agent.collide = False
            agent.silent = False
            agent.size = 0.15 # flag
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        cur_airport = 0
        for i in range(len(world.landmarks)):
            if i % 2 == 0:
                world.landmarks[i].name = airports[cur_airport] + '-1'
                if i != 0:
                    cur_airport += 1
            elif i % 2 == 1:
                world.landmarks[i].name = airports[cur_airport] + '-2'
            world.landmarks[i].collide = False
            world.landmarks[i].movable = False

        # make initial conditions
        self.reset_world(world)
        return world


    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states for airplanes
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)



    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            rew -= min(dists)
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        '''
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        '''
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)
