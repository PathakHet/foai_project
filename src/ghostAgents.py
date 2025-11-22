from game import Agent
from game import Actions
from game import Directions
import random
from util import manhattanDistance
import util


class GhostAgent(Agent):
    def __init__(self, index):
        self.index = index

    def getAction(self, state):
        dist = self.getDistribution(state)
        if len(dist) == 0:
            return Directions.STOP
        else:
            return util.chooseFromDistribution(dist)

    def getDistribution(self, state):
        util.raiseNotDefined()


class RandomGhost(GhostAgent):
    def getDistribution(self, state):
        dist = util.Counter()
        for a in state.getLegalActions(self.index):
            dist[a] = 1.0
        dist.normalize()
        return dist


class DeterministicGhost(GhostAgent):
    def __init__(self, index, clockwise: bool = True):
        super().__init__(index)
        base_order = [
            Directions.NORTH,
            Directions.EAST if clockwise else Directions.WEST,
            Directions.SOUTH,
            Directions.WEST if clockwise else Directions.EAST,
        ]
        shift = index % len(base_order)
        self.priority_order = base_order[shift:] + base_order[:shift]

    def getAction(self, state):
        legalActions = state.getLegalActions(self.index)
        if not legalActions:
            return Directions.STOP
        for action in self.priority_order:
            if action in legalActions:
                return action
        return sorted(legalActions)[0]

    def getDistribution(self, state):
        legalActions = state.getLegalActions(self.index)
        dist = util.Counter()
        if not legalActions:
            return dist
        for action in self.priority_order:
            if action in legalActions:
                dist[action] = 1.0
                dist.normalize()
                return dist
        first_action = sorted(legalActions)[0]
        dist[first_action] = 1.0
        dist.normalize()
        return dist


class DirectionalGhost(GhostAgent):
    def __init__(self, index, prob_attack=0.8, prob_scaredFlee=0.8):
        self.index = index
        self.prob_attack = prob_attack
        self.prob_scaredFlee = prob_scaredFlee

    def getDistribution(self, state):
        ghostState = state.getGhostState(self.index)
        legalActions = state.getLegalActions(self.index)
        pos = state.getGhostPosition(self.index)
        isScared = ghostState.scaredTimer > 0

        speed = 1
        if isScared:
            speed = 0.5

        actionVectors = [Actions.directionToVector(
            a, speed) for a in legalActions]
        newPositions = [(pos[0]+a[0], pos[1]+a[1]) for a in actionVectors]
        pacmanPosition = state.getPacmanPosition()

        distancesToPacman = [manhattanDistance(
            pos, pacmanPosition) for pos in newPositions]
        if isScared:
            bestScore = max(distancesToPacman)
            bestProb = self.prob_scaredFlee
        else:
            bestScore = min(distancesToPacman)
            bestProb = self.prob_attack
        bestActions = [action for action, distance in zip(
            legalActions, distancesToPacman) if distance == bestScore]

        dist = util.Counter()
        for a in bestActions:
            dist[a] = bestProb / len(bestActions)
        for a in legalActions:
            dist[a] += (1-bestProb) / len(legalActions)
        dist.normalize()
        return dist
