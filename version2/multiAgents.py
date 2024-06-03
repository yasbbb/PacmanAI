# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # start with game state score
        score = successorGameState.getScore()
        
        # distance to the closest food
        foodList = newFood.asList()
        if foodList:
            closestFoodDist = min([manhattanDistance(newPos, food) for food in foodList])
            score += 1.0 / closestFoodDist

        # distance to the ghosts and scared times
        for ghostState in newGhostStates:
            ghostPos = ghostState.getPosition()
            ghostDist = manhattanDistance(newPos, ghostPos)
            if ghostState.scaredTimer == 0 and ghostDist <= 1:
                # Dangerously close to an active ghost
                return -float('inf')  # Highly undesirable state
            elif ghostState.scaredTimer > 0 and ghostDist <= 1:
                # Close to a scared ghost
                score += 200  # Encourage eating scared ghosts
        
        score -= 3 * len(foodList)  
        if action == Directions.STOP:
            score -= 10  
        
        return score
        # return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def minimax(agentIndex, depth, gameState):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            if agentIndex == 0:  # Pacman's turn
                return max(minimax(1, depth, gameState.generateSuccessor(agentIndex, action))
                           for action in gameState.getLegalActions(agentIndex))
            else:  # Ghosts' turn
                nextAgent = agentIndex + 1
                if nextAgent == gameState.getNumAgents():
                    nextAgent = 0
                    depth += 1
                return min(minimax(nextAgent, depth, gameState.generateSuccessor(agentIndex, action))
                           for action in gameState.getLegalActions(agentIndex))

        # Pacman is agentIndex 0
        maximum = float("-inf")
        bestAction = None
        for action in gameState.getLegalActions(0):
            value = minimax(1, 0, gameState.generateSuccessor(0, action))
            if value > maximum or bestAction is None:
                maximum = value
                bestAction = action
        return bestAction
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def max_value(state, depth, alpha, beta):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state), None
            value = float("-inf")
            best_action = None
            for action in state.getLegalActions(0):
                successor = state.generateSuccessor(0, action)
                successor_value, _ = min_value(successor, depth, 1, alpha, beta)
                if successor_value > value:
                    value, best_action = successor_value, action
                if value > beta:
                    return value, best_action
                alpha = max(alpha, value)
            return value, best_action

        def min_value(state, depth, agentIndex, alpha, beta):
            if state.isWin() or state.isLose():
                return self.evaluationFunction(state), None
            value = float("inf")
            best_action = None
            for action in state.getLegalActions(agentIndex):
                if agentIndex == gameState.getNumAgents() - 1:
                    successor = state.generateSuccessor(agentIndex, action)
                    successor_value, _ = max_value(successor, depth + 1, alpha, beta)
                else:
                    successor = state.generateSuccessor(agentIndex, action)
                    successor_value, _ = min_value(successor, depth, agentIndex + 1, alpha, beta)
                if successor_value < value:
                    value, best_action = successor_value, action
                if value < alpha:
                    return value, best_action
                beta = min(beta, value)
            return value, best_action

        # Body of getAction starts here
        _, action = max_value(gameState, 0, float("-inf"), float("inf"))
        return action
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectimax(agentIndex, depth, gameState):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            
            numAgents = gameState.getNumAgents()
            nextAgent = (agentIndex + 1) % numAgents
            nextDepth = depth if agentIndex < numAgents - 1 else depth + 1
            
            legalActions = gameState.getLegalActions(agentIndex)
            
            if agentIndex == 0:  # Pacman's turn, maximize score
                scores = [expectimax(nextAgent, nextDepth, gameState.generateSuccessor(agentIndex, action)) for action in legalActions]
                return max(scores) if depth > 0 else legalActions[scores.index(max(scores))]
            else:  # Ghosts' turn, calculate expected value
                scores = [expectimax(nextAgent, nextDepth, gameState.generateSuccessor(agentIndex, action)) for action in legalActions]
                return sum(scores) / len(scores) if scores else self.evaluationFunction(gameState)
        
        return expectimax(0, 0, gameState)
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    if currentGameState.isWin():
        return float("inf")  # Very high score for winning
    if currentGameState.isLose():
        return -float("inf")  # Very low score for losing

    pacmanPos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    capsuleList = currentGameState.getCapsules()
    score = scoreEvaluationFunction(currentGameState)

    # Higher reward for closer food
    foodDistances = [manhattanDistance(pacmanPos, food) for food in foodList]
    if foodList:
        nearestFoodDist = min(foodDistances)
        score += 15.0 / nearestFoodDist

    # Increase rewards/penalties for ghost states
    for ghost in ghostStates:
        distance = manhattanDistance(pacmanPos, ghost.getPosition())
        if ghost.scaredTimer == 0:
            score -= 200 / (distance + 1)
        else:
            if distance < 5:
                score += 100 / (distance + 1)

    # penalty for uneaten capsule
    score -= 120 * len(capsuleList)

    # Penalty for remaining food pellets 
    score -= 8 * len(foodList)

    scaredGhosts = sum(ghostState.scaredTimer > 0 for ghostState in ghostStates)
    score += 200 * scaredGhosts

    return score
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
