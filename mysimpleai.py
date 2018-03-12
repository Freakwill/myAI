# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as LA
import scipy.spatial.distance

from simpleai.search import SearchProblem, astar
from simpleai.search.models import SearchNodeHeuristicOrdered

roles = ('peasant', 'goat', 'fox', 'vegetables')

#  peasant-goat-fox-vegetables problem
class PGFVProblem(SearchProblem):
    GOAL = (1,1,1,1)

    @staticmethod
    def is_dead(state):
        return state[1] == state[2] != state[0] or state[1] == state[3] != state[0]

    def actions(self, state):
        orig = [0] + [k for k, s in enumerate(state[1:], 1) if state[0] == s]
        return [a for a in orig if not PGFVProblem.is_dead(self.result(state, a))]

    def result(self, state, action):
        state = list(state)
        state[0] = 1 - state[0]
        if action != 0:
            state[action] = 1 - state[action]
        return tuple(state)

    def is_goal(self, state):
        return state == PGFVProblem.GOAL

    def heuristic(self, state):
        return sum(k for k in state if k != 1)

    def action_representation(self, action):
        """
        Returns a string representation of an action.
        By default it returns str(action).
        """
        if action is None:
            return 'Start'
        if action == 0:
            return 'only peasant'
        else:
            return 'peasant & ' + roles[action]

    def state_representation(self, state):
        return ','.join('%s:%s'%(role, ('left', 'right')[pos]) for role, pos in zip(roles, state))



# problem = PGFVProblem(initial_state=(0,0,0,0))

# result = astar(problem)

# for a, s in result.path():
#     print(problem.action_representation(a), '#', problem.state_representation(s))



#  peasant-goat-fox-vegetables problem
all_actions = ((2,0), (0,2), (1,1), (0,1),(1,0))

class M3C3Problem(SearchProblem):
    GOAL = (0,0,1)

    @staticmethod
    def is_dead(state):
        return state[1] > state[0] > 0 or state[1] < state[0] < 3

    def actions(self, state):
        if state[2] == 0:
            return [(m, c) for m, c in all_actions if m <= state[0] and c <= state[1]]
        else:
            return [(m, c) for m, c in all_actions if m <= 3 - state[0] and c <= 3 - state[1]]


    def result(self, state, action):
        state = list(state)
        if state[2] == 0:
            state[0] -= action[0]
            state[1] -= action[1]
            state[2] = 1
        else:
            state[0] += action[0]
            state[1] += action[1]
            state[2] = 0
        return tuple(state)

    def is_goal(self, state):
        return state == M3C3Problem.GOAL

    def heuristic(self, state):
        return scipy.spatial.distance.cityblock(state, M3C3Problem.GOAL)

    def action_representation(self, action):
        """
        Returns a string representation of an action.
        By default it returns str(action).
        """
        if action is None:
            return 'Start'
        return 'take %d M & %d C'%(action[0], action[1])

    def state_representation(self, state):
        if state[2] == 0:
            return 'There are %d M & %d C on the left bank'%(state[0], state[1])
        else:
            return 'There are %d M & %d C on the right bank'%(3-state[0], 3-state[1])



problem = M3C3Problem(initial_state=(3,3,0))

result = astar(problem)

for a, s in result.path():
    print(problem.action_representation(a), '#', problem.state_representation(s))
