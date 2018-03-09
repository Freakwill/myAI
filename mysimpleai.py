# -*- coding: utf-8 -*-

import numpy as np

from simpleai.search import SearchProblem, astar
from simpleai.search.models import SearchNodeHeuristicOrdered

roles = ('peasant', 'goat', 'fox', 'vegetables')

def is_dead(state):
    return state[1] == state[2] != state[0] or state[1] == state[3] != state[0]

#  peasant-goat-fox-vegetables problem
class PGFVProblem(SearchProblem):
    def actions(self, state):
        orig = [0] + [k for k, s in enumerate(state[1:], 1) if state[0] == s]
        return [a for a in orig if not is_dead(self.result(state, a))]

    def result(self, state, action):
        state = list(state)
        state[0] = 1 - state[0]
        if action != 0:
            state[action] = 1 - state[action]
        return tuple(state)

    def is_goal(self, state):
        return state == (1,1,1,1)

    def heuristic(self, state):
        return sum(k for k in state if k != 1)

    def action_representation(self, action):
        """
        Returns a string representation of an action.
        By default it returns str(action).
        """
        if action is None:
            return 'State'
        if action == 0:
            return 'only peasant'
        else:
            return 'peasant & ' + roles[action]

    def state_representation(self, state):
        return ','.join('%s:%s'%(role, ('left', 'right')[pos]) for role, pos in zip(roles, state))



problem = PGFVProblem(initial_state=(0,0,0,0))

result = astar(problem)

for a, s in result.path():
    print(problem.action_representation(a), '#', problem.state_representation(s))