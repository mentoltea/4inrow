import game
from functools import lru_cache, cache


# def get_moves_n_steps_from_win(start_position: game.Game, myturn: int, n: int) -> list[ list[ int ] ]:
#     if (start_position.ended):
#         # print(start_position.winner)
#         if (start_position.winner == myturn):
#             return [ [  ] ]
#         else:
#             return []
    
#     if (n==0): return []
    
#     moves: list[list[int]] = []
    
#     if (start_position.turn == myturn):
#         for m in range(start_position.columns):
#             pos = start_position.copy()
#             if (pos.move(m)):
#                 next_moves = get_moves_n_steps_from_win(pos, myturn, n-1)
#                 for nm in next_moves:
#                     nm.insert(0, m)
#                     moves.append(nm)
    
#     else:
#         m_possible_count = 0
#         m_count = 0
#         for m in range(start_position.columns):
#             pos = start_position.copy()
#             if (pos.move(m)):
#                 m_possible_count += 1
#                 next_moves = get_moves_n_steps_from_win(pos, myturn, n-1)
#                 if (len(next_moves)>0):
#                     m_count += 1
#                     for nm in next_moves:
#                         nm.insert(0, m)
#                         moves.append(nm)
#         # print(f"{n}: {m_count} / {m_possible_count}")
#         if (m_count != m_possible_count):
#             moves = []

#     return moves

# def get_chance_n_steps_from_win(start_position: game.Game, myturn: int, n: int) -> float:
#     if (start_position.ended):
#         # print(start_position.winner)
#         if (start_position.winner == myturn):
#             return 1
#         else:
#             return 0
    
#     if (n==0): return 1/2
    
#     chances: list[float] = []
#     k = 1
    
#     if (start_position.turn == myturn):
#         for m in range(start_position.columns):
#             pos = start_position.copy()
#             if (pos.move(m)):
#                 next_chance = get_chance_n_steps_from_win(pos, myturn, n-1)
#                 chances.append(next_chance)
#     else:
#         m_possible_count = 0
#         m_count = 0
#         for m in range(start_position.columns):
#             pos = start_position.copy()
#             if (pos.move(m)):
#                 m_possible_count += 1
#                 next_chance = get_chance_n_steps_from_win(pos, myturn, n-1)
#                 if (next_chance>0):
#                     m_count += 1
#                     chances.append(next_chance)
#         k = m_count/m_possible_count
        
#     if len(chances)==0: return 0
#     return k*sum(chances)/len(chances)

def get_moves_and_chance_n_steps_from_win(start_position: game.Game, myturn: int, n: int) -> tuple[list[ list[ int ] ], float, float]:
    if (start_position.ended):
        # print(start_position.winner)
        if (start_position.winner == myturn):
            return ([ [] ], 1, 1)
        elif (start_position.winner == -myturn):
            return ([], 0, 0)
        else:
            return ([], 0, 0)
    
    if (n==0): return ([], 1/2, 1/2)
    
    moves: list[list[int]] = []
    chances: list[float] = []
    certain_chances: list[float] = []
    
    if (start_position.turn == myturn):
        for m in range(start_position.columns):
            pos = start_position.copy()
            if (pos.move(m)):
                (next_moves, next_chance, next_certain_chance) = get_moves_and_chance_n_steps_from_win(pos, myturn, n-1)
                chances.append(next_chance)
                certain_chances.append(next_certain_chance)
                for nm in next_moves:
                    nm.insert(0, m)
                    moves.append(nm)
        certain_chance = max(certain_chances)
    else:
        m_possible_count = 0
        m_m_count = 0
        for m in range(start_position.columns):
            pos = start_position.copy()
            if (pos.move(m)):
                m_possible_count += 1
                (next_moves, next_chance, next_certain_chance) = get_moves_and_chance_n_steps_from_win(pos, myturn, n-1)
                chances.append(next_chance)
                certain_chances.append(next_certain_chance)
                if (len(next_moves)>0):
                    m_m_count += 1
                    for nm in next_moves:
                        nm.insert(0, m)
                        moves.append(nm)
        if (m_m_count != m_possible_count):
            moves = []
        certain_chance = min(certain_chances)
    
    
    if len(chances)==0: avg_chance=0
    else: avg_chance = sum(chances)/len(chances)
    
    return (moves, avg_chance, certain_chance)

def get_certain_chance_n_steps(start_position: game.Game, myturn: int, n: int) -> float:
    if (start_position.ended):
        # print(start_position.winner)
        if (start_position.winner == myturn):
            return 1
        elif (start_position.winner == -myturn):
            return 0
        else:
            return 0
    if (n==0): return 1/2

    chances: list[float] = []
    
    for m in range(start_position.columns):
        pos = start_position.copy()
        if (pos.move(m)):
            next_chance = get_certain_chance_n_steps(pos, myturn, n-1)
            chances.append(next_chance)
    
    if (start_position.turn == myturn):
        return max(chances)
    else:
        return min(chances)
    

def loss(position: game.Game, myturn:int, depth_search:int=5) -> float:
    if position.ended:
        if position.winner == myturn:
            return 0
        if position.winner == -myturn:
            return 1
        return 4/position.number_of_turns

    add = 0
    if (position.turn != myturn):
        add += 1/position.number_of_turns

    (certain_wins, chance, certain_chance) = get_moves_and_chance_n_steps_from_win(position, myturn, depth_search)
    if (certain_chance != 1/2):
        return 1 - certain_chance
    
    return 1 - chance + add