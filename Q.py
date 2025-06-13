import game
import mcts

def Q(position: game.Game, myturn: int, depth_search=5) -> float:
    x = mcts.loss(position, myturn, depth_search=depth_search)
    y = 1 - 2* ( (x*x) / (x*x + (1-x)*(1-x)))
    return y

# classifies Q value
# classes:
# -2: only from -1      -> absolute loss
# -1: from (-1, -1/3]   -> loss
#  0: from (-1/3, 1/3)  -> draw
#  1: from [1/3, 1)     -> win
#  2: only from 1       -> absolute win
def classify(prob: float) -> float:
    prob *= 1.5
    res = round(prob)
    return res


class Q_based_opponent:
    def __init__(self, depth=6) -> None:
        self.depth = depth
    
    def get_move(self, gm: game.Game) -> int:
        moves: list[tuple[int,float]] = []
        for m in range(gm.columns):
            copy = gm.copy()
            if copy.move(m):
                moves.append( (m, Q(copy, gm.turn, self.depth)) )
        moves.sort(key=lambda t: t[1], reverse=True)
        return moves[0][0]
            