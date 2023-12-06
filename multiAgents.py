from Agents import Agent
import util
import random
from Game import GameState


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def __init__(self, *args, **kwargs) -> None:
        self.index = 0  # your agent always has index 0

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        It takes a GameState and returns a tuple representing a position on the game board.
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions(self.index)

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        The evaluation function takes in the current and proposed successor
        GameStates (Game.py) and returns a number, where higher numbers are better.
        You can try and change this evaluation function if you want but it is not necessary.
        """
        nextGameState = currentGameState.generateSuccessor(self.index, action)
        return nextGameState.getScore(self.index) - currentGameState.getScore(self.index)


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    Every player's score is the number of pieces they have placed on the board.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore(0)


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxAgent, AlphaBetaAgent & ExpectimaxAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (Agents.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2', **kwargs):
        self.index = 0  # your agent always has index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent which extends MultiAgentSearchAgent and is supposed to be implementing a minimax tree with a certain depth.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(**kwargs)

    def getAction(self, state: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction

        But before getting your hands dirty, look at these functions:

        gameState.isGameFinished() -> bool
        gameState.getNumAgents() -> int
        gameState.generateSuccessor(agentIndex, action) -> GameState
        gameState.getLegalActions(agentIndex) -> list
        self.evaluationFunction(gameState) -> float
        """

        def best_action(index, depth, current_state: GameState):
            if current_state.isGameFinished():
                return current_state.getScore()
            if depth == 0:
                return self.evaluationFunction(current_state), None
            players = current_state.getNumAgents()
            legal_actions = current_state.getLegalActions(index)
            best_value = None
            best_act = None
            max_value = -1000000
            min_value = 1000000
            for action in legal_actions:
                next_state = current_state.generateSuccessor(index, action)
                val, _ = best_action((index + 1) % players, depth - 1, next_state)
                if index == 0:
                    if val > max_value:
                        max_value = val
                        best_act = action
                    best_value = max_value
                else:
                    if val < min_value:
                        min_value = val
                        best_act = action
                    best_value = min_value
            return best_value, best_act

        _, act = best_action(self.index, self.depth, state)
        return act


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning. It is very similar to the MinimaxAgent but you need to implement the alpha-beta pruning algorithm too.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(**kwargs)

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction

        You should keep track of alpha and beta in each node to be able to implement alpha-beta pruning.
        """

        def best_action(index, depth, current_state: GameState, alpha, beta):
            if current_state.isGameFinished():
                return current_state.getScore(0), None
            if depth == 0:
                return self.evaluationFunction(current_state), None
            players = current_state.getNumAgents()
            legal_actions = current_state.getLegalActions(index)
            best_act = None
            best_value = None
            max_value = -1000000
            min_value = 1000000
            for action in legal_actions:
                next_state = current_state.generateSuccessor(index, action)
                val, _ = best_action((index + 1) % players, depth - 1, next_state, alpha, beta)
                if index == 0:
                    if val > beta:
                        return val, action
                    alpha = max(alpha, val)

                    if val > max_value:
                        max_value = val
                        best_act = action
                    best_value = max_value
                else:
                    if val < alpha:
                        return val, action
                    beta = min(beta, val)

                    if val < min_value:
                        min_value = val
                        best_act = action
                    best_value = min_value

            return best_value, best_act

        _, act = best_action(self.index, self.depth, gameState, -100, 100)
        return act



class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent which has a max node for your agent but every other node is a chance node.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(**kwargs)

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All opponents should be modeled as choosing uniformly at random from their
        legal moves.
        """

        def best_action(index, depth, current_state: GameState):
            if current_state.isGameFinished():
                return current_state.getScore()
            if depth == 0:
                return self.evaluationFunction(current_state), None
            players = current_state.getNumAgents()
            legal_actions = current_state.getLegalActions(index)
            best_act = None
            max_value = -1000000
            best_value = 0
            for action in legal_actions:
                next_state = current_state.generateSuccessor(index, action)
                val, _ = best_action((index + 1) % players, depth - 1, next_state)
                if index == 0:
                    if val > max_value:
                        max_value = val
                        best_act = action
                    best_value = max_value
                else:
                    val, _ = best_action((index + 1) % players, depth - 1, next_state)
                    best_value += val

            if index != 0:
                best_value /= legal_actions.__len__()
            return best_value, best_act

        _, act = best_action(self.index, self.depth, gameState)
        return act


def betterEvaluationFunction(currentGameState):
    """
    Your extreme evaluation function.

    You are asked to read the following paper on othello heuristics and extend it for two to four player rollit game.
    Implementing a good stability heuristic has extra points.
    Any other brilliant ideas are also accepted. Just try and be original.

    The paper: Sannidhanam, Vaishnavi, and Muthukaruppan Annamalai. "An analysis of heuristics in othello." (2015).

    Here are also some functions you will need to use:

    gameState.getPieces(index) -> list
    gameState.getCorners() -> 4-tuple
    gameState.getScore() -> list
    gameState.getScore(index) -> int

    """
    players_count = currentGameState.getNumAgents()

    # parity
    max_player_coins = currentGameState.getScore(0)
    min_players_coins = 0
    for i in currentGameState.getScore()[1:players_count]:
        min_players_coins += i
    parity = 100 * (max_player_coins - min_players_coins) / (max_player_coins + min_players_coins)

    # corners
    max_player_corners = 0
    min_players_corners = 0
    for corner in currentGameState.getCorners():
        if corner == 0:
            max_player_corners += 1
        elif corner != -1:
            min_players_corners += 1
    if max_player_corners + min_players_corners != 0:
        corners = 100 * (max_player_corners - min_players_corners) / (max_player_corners + min_players_corners)
    else:
        corners = 0

    # mobility
    max_player_mobility = currentGameState.getLegalActions(0).__len__()
    min_player_mobility = 0
    for i in range(1, players_count):
        min_player_mobility += currentGameState.getLegalActions(i).__len__()
    if max_player_mobility + min_player_mobility != 0:
        mobility = 100 * (max_player_mobility - min_player_mobility) / (max_player_mobility + min_player_mobility)
    else:
        mobility = 0

    # stability
    stability_static = [
        [4, -3, 2, 2, 2, 2, -3, 4],
        [-3, -4, -1, -1, -1, -1, -4, -3],
        [2, -1, 1, 0, 0, 1, -1, 2],
        [2, -1, 0, 1, 1, 0, -1, 2],
        [2, -1, 0, 1, 1, 0, -1, 2],
        [2, -1, 1, 0, 0, 1, -1, 2],
        [-3, -4, -1, -1, -1, -1, -4, -3],
        [4, -3, 2, 2, 2, 2, -3, 4],
    ]
    max_player_stability = 0
    min_players_stability = 0
    for piece in currentGameState.getPieces(0):
        max_player_stability += stability_static[piece[0]][piece[1]]
    for i in range(1, players_count):
        for piece in currentGameState.getPieces(i):
            min_players_stability += stability_static[piece[0]][piece[1]]

    if max_player_mobility + min_players_stability != 0:
        stability = 100 * (max_player_mobility - min_players_stability) / (max_player_mobility + min_players_stability)
    else:
        stability = 0

    return 5 * parity + 5 * corners + mobility + stability


# Abbreviation
better = betterEvaluationFunction
