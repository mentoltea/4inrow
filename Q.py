import game
import mcts

def Q(position: game.Game, myturn: int, depth_search=5) -> float:
    x = mcts.loss(position, myturn, depth_search=depth_search)
    y = 1 - 2* ( (x*x) / (x*x + (1-x)*(1-x)))
    return y