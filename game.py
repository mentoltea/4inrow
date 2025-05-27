from __future__ import annotations
import enum
import numpy

class CellEnum(enum.Enum):
    EMPTY = 0
    FILLED_P1 = 1
    FILLED_P2 = -1

class Game:
    def __init__(self, rows:int = 6, columns: int = 7, startturn: int = 0):
        # matrix[y][x]
        self.rows = rows
        self.columns = columns
        
        self.lastmove_xy: tuple[int,int] = (0,0)

        # self.gamemap: list[ list[int] ] = [ [ CellEnum.EMPTY.value for _ in range(columns)] for _ in range(rows)]
        self.gamemap = numpy.zeros((rows, columns), dtype=int)
        
        # 0 for P1
        # 1 for P2
        self.turn: int = startturn
        self.number_of_turns: int = 0
        
        self.ended: bool = False
        
        # -1 - nobody
        # 0 - P1
        # 1 - P2
        self.winner = 0
    
    def copy(self) -> Game:
        gm = Game(self.rows, self.columns, self.turn)
        gm.gamemap = self.gamemap.copy()
        gm.number_of_turns = self.number_of_turns
        gm.lastmove_xy = self.lastmove_xy
        gm.ended = self.ended
        gm.winner = self.winner
        return gm
    
    def are_empty_cells(self) -> bool:
        for x in range(self.columns):
            if self.gamemap[0][x] == CellEnum.EMPTY.value:
                return True
        self.ended = True
        self.winner = -1
        return False
    
    def check_win_cell(self, x: int, y: int) -> bool:
        # print(x, y)
        ref = self.gamemap[y][x]
        # print(f"ref = {ref}")
        inrow = 0
        for dx in range(-3, 3+1):
            if x+dx<0: continue
            if x+dx>=self.columns: break
            
            if (self.gamemap[y][x+dx] == ref):
                inrow+=1
            else:
                inrow=0
            
            if inrow == 4:
                # print("on x")
                return True
        if inrow == 4:
            # print("on x o")
            return True
        
        inrow = 0
        for dy in range(-3, 3+1):
            if y+dy<0: continue
            if y+dy>=self.rows: break
            
            if (self.gamemap[y+dy][x] == ref):
                inrow+=1
            else:
                inrow=0
            
            if inrow == 4:
                # print("on y")
                return True
        if inrow == 4:
            # print("on y o")
            return True
        
        inrow = 0
        for di in range(-3, 3+1):
            if x+di<0 or y+di<0: continue
            if x+di>=self.columns or y+di>=self.rows: break
            
            if (self.gamemap[y+di][x+di] == ref):
                inrow+=1
            else:
                inrow=0
            
            if inrow == 4:
                # print("on x+y")
                return True
        if inrow == 4:
            # print("on x+y o")
            return True
        
        inrow = 0
        for dj in range(-3, 3+1):
            if x+dj<0 or y-dj>=self.rows: continue
            if x+dj>=self.columns or y-dj<0: break
            
            if (self.gamemap[y-dj][x+dj] == ref):
                inrow+=1
            else:
                inrow=0
            
            if inrow == 4:
                # print("on x-y")
                return True
        if inrow == 4:
            # print("on x-y o")
            return True
        
        return False
    
    # 1 - P1 won
    # -1 - P2 won
    # 0 - no win
    # is based on the last move
    def check_win(self) -> int:
        # print(self.lastmove_xy)
        win = self.check_win_cell(self.lastmove_xy[0], self.lastmove_xy[1])
        
        if not win: return -1
        
        self.ended = True
        if (self.gamemap[self.lastmove_xy[1]][self.lastmove_xy[0]] == CellEnum.FILLED_P1.value):
            self.winner = CellEnum.FILLED_P1.value
            return CellEnum.FILLED_P1.value
        else:
            self.winner = CellEnum.FILLED_P2.value
            return CellEnum.FILLED_P2.value
            
    
    # True -> ok
    # False -> move is illegal
    def move(self, column:int) -> bool:
        if (self.ended): return False
        
        fill = CellEnum.FILLED_P1.value
        if (self.turn == 1):
            fill = CellEnum.FILLED_P2.value
        
        found_empty = False
        for y in range(self.rows - 1, -1, -1):
            if (self.gamemap[y][column] == CellEnum.EMPTY.value):
                self.gamemap[y][column] = fill
                found_empty = True
                self.lastmove_xy = (column, y)
                break
                
        if (not found_empty):
            return False
        
        self.turn = (self.turn+1)%2
        self.number_of_turns += 1
        
        return True