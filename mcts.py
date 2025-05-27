import game
from functools import lru_cache, cache


def get_moves_n_steps_from_win(start_position: game.Game, myturn: int, n: int) -> list[ list[ int ] ]:
    if (start_position.ended):
        # print(start_position.winner)
        if (start_position.winner == myturn):
            return [ [  ] ]
        else:
            return []
    
    if (n==0): return []
    
    moves: list[list[int]] = []
    
    if (start_position.turn == myturn):
        for m in range(start_position.columns):
            pos = start_position.copy()
            if (pos.move(m)):
                next_moves = get_moves_n_steps_from_win(pos, myturn, n-1)
                for nm in next_moves:
                    nm.insert(0, m)
                    moves.append(nm)
    
    else:
        m_possible_count = 0
        m_count = 0
        for m in range(start_position.columns):
            pos = start_position.copy()
            if (pos.move(m)):
                m_possible_count += 1
                next_moves = get_moves_n_steps_from_win(pos, myturn, n-1)
                if (len(next_moves)>0):
                    m_count += 1
                    for nm in next_moves:
                        nm.insert(0, m)
                        moves.append(nm)
        # print(f"{n}: {m_count} / {m_possible_count}")
        if (m_count != m_possible_count):
            moves = []

    return moves

def get_chance_n_steps_from_win(start_position: game.Game, myturn: int, n: int) -> float:
    if (start_position.ended):
        # print(start_position.winner)
        if (start_position.winner == myturn):
            return 1
        else:
            return 0
    
    if (n==0): return 1/2
    
    chances: list[float] = []
    k = 1
    
    if (start_position.turn == myturn):
        for m in range(start_position.columns):
            pos = start_position.copy()
            if (pos.move(m)):
                next_chance = get_chance_n_steps_from_win(pos, myturn, n-1)
                chances.append(next_chance)
    else:
        m_possible_count = 0
        m_count = 0
        for m in range(start_position.columns):
            pos = start_position.copy()
            if (pos.move(m)):
                m_possible_count += 1
                next_chance = get_chance_n_steps_from_win(pos, myturn, n-1)
                if (next_chance>0):
                    m_count += 1
                    chances.append(next_chance)
        k = m_count/m_possible_count
        
    if len(chances)==0: return 0
    return k*sum(chances)/len(chances)

def get_moves_and_chance_n_steps_from_win(start_position: game.Game, myturn: int, n: int) -> tuple[list[ list[ int ] ], float]:
    if (start_position.ended):
        # print(start_position.winner)
        if (start_position.winner == myturn):
            return ([ [] ], 1)
        else:
            return ([], 0)
    
    if (n==0): return ([], 1/2)
    
    moves: list[list[int]] = []
    chances: list[float] = []
    k = 1
    
    if (start_position.turn == myturn):
        for m in range(start_position.columns):
            pos = start_position.copy()
            if (pos.move(m)):
                (next_moves, next_chance) = get_moves_and_chance_n_steps_from_win(pos, myturn, n-1)
                chances.append(next_chance)
                for nm in next_moves:
                    nm.insert(0, m)
                    moves.append(nm)
    else:
        m_possible_count = 0
        m_c_count = 0
        m_m_count = 0
        for m in range(start_position.columns):
            pos = start_position.copy()
            if (pos.move(m)):
                m_possible_count += 1
                (next_moves, next_chance) = get_moves_and_chance_n_steps_from_win(pos, myturn, n-1)
                if (next_chance>0):
                    m_c_count += 1
                    chances.append(next_chance)
                if (len(next_moves)>0):
                    m_m_count += 1
                    for nm in next_moves:
                        nm.insert(0, m)
                        moves.append(nm)
        # print(f"{n}: {m_count} / {m_possible_count}")
        if (m_m_count != m_possible_count):
            moves = []
        k = m_c_count/m_possible_count
        
    if len(chances)==0: return (moves, 0)
    return (moves, k*sum(chances)/len(chances))

def loss(position: game.Game, myturn:int, depth_search:int=5) -> float:
    if position.ended:
        if position.winner == myturn:
            return 0
        if position.winner == -myturn:
            return 10
        return 1/position.number_of_turns

    (certain_wins, chance) = get_moves_and_chance_n_steps_from_win(position, myturn, depth_search)
    if len(certain_wins)>0:
        return 0
    
    add = 0
    if (position.turn != myturn):
        add += 1/position.number_of_turns
    
    return 1-chance+add
