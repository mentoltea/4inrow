import pygame
import game as GM
from game import Game
import nn

pygame.init()

WIN_X = 1080
WIN_Y = 720
FPS = 30
clock = pygame.time.Clock()
wn = pygame.display.set_mode((WIN_X, WIN_Y), pygame.SRCALPHA, vsync=1)
window = pygame.Surface((WIN_X, WIN_Y), pygame.SRCALPHA)
RUN = True
GAME_ENDED = False

game = Game(6, 7)
game_surf = pygame.Surface((WIN_X, WIN_Y), pygame.SRCALPHA)
game_surf_pos = (0,0)

NN = nn.game_NN.load_from("epoch100/NN_6x7_i_4_R_109.keras")
NN_turn = 1

DefaultFont = pygame.font.Font(pygame.font.match_font('timesnewroman'), 44)
def draw_text(text: str, x: float, y: float, surf: pygame.Surface, font: pygame.font.Font = DefaultFont, color:tuple[int,int,int]=(0,0,0)):
    surf.blit(font.render(text,True,color), (x,y))

def in_constrain(value:float, minimum:float, maximum:float) -> bool:
    if (value < minimum): return False
    if (value > maximum): return False
    return True

def constrain(value:float, minimum:float, maximum:float) -> float:
    if (value < minimum): return minimum
    if (value > maximum): return maximum
    return value

def uv2xy(surf: pygame.Surface, uv: tuple[float, float]) -> tuple[int, int]:
    (xsize, ysize) = surf.get_size()
    (u,v) = uv
    return (int(constrain(u*xsize, 0, xsize-1)), 
            int(constrain(v*ysize, 0, ysize-1)))

def draw_game(game: Game, surf: pygame.Surface):
    BACK = (0, 37, 225, 255)
    GRAY = (125,125,125, 255)
    P2 = (255, 126, 0, 255)
    P1 = (255, 0, 0, 255)
    surf.fill(BACK)
    (xsize, ysize) = surf.get_size()
    rad_uv = (
        1/(game.columns+1),
        1/(game.rows+1)
    )
    radius = min(uv2xy(surf, rad_uv))/2 - 5
    for y in range(game.rows):
        for x in range(game.columns):
            clr = GRAY
            ref = game.gamemap[y][x]
            if (ref == GM.CellEnum.FILLED_P1.value):
                clr = P1
            elif (ref == GM.CellEnum.FILLED_P2.value):
                clr = P2
            
            coord_uv = (
                (x+1)/(game.columns+1),
                (y+1)/(game.rows+1)
            )
            coord_xy = uv2xy(surf, coord_uv)
            
            pygame.draw.circle(
                surf, clr,
                coord_xy,
                radius
            )
    
    if game.ended:
        text = "Game ended"
        if game.winner == 0:
            text = "Player 1 won"
        elif game.winner == 1:
            text = "Player 2 won"
        
        (text_size_x, text_size_y) = DefaultFont.size(text)
        draw_text(text, (xsize-text_size_x)/2, (ysize-text_size_y)/2, surf)

# -> (clicked_circle, xy_matrix_coords)
def click_inside_game(game: Game, surf: pygame.Surface, mouse_posrel: tuple[int,int]) -> tuple[bool, tuple[int,int]]:
    BAD_ANS = (False, (-1,-1))
    (mx, my) = mouse_posrel
    if (mx<0 or my<0): return BAD_ANS
    
    (xsize, ysize) = surf.get_size()
    if (mx>=xsize or my>ysize): return BAD_ANS
    
    rad_uv = (
        1/(game.columns+1),
        1/(game.rows+1)
    )
    radius = min(uv2xy(surf, rad_uv))
    for y in range(game.rows):
        for x in range(game.columns):
            coord_uv = (
                (x+1)/(game.columns+1),
                (y+1)/(game.rows+1)
            )
            coord_xy = uv2xy(surf, coord_uv)
            (cx, cy) = coord_xy
            if ( (cx-mx)*(cx-mx) + (cy-my)*(cy-my) <= radius*radius ):
                return (True, (x,y))
    
    return BAD_ANS
            

while RUN:
    clock.tick(FPS)
    window.fill((255,255,255))
    EVENTS = pygame.event.get()
    
    for event in EVENTS:
        if (event.type == pygame.QUIT):
            RUN = False # type: ignore
        if (event.type == pygame.MOUSEBUTTONUP):
            mouse_pos = pygame.mouse.get_pos()
            
            mouse_posrel = (
                mouse_pos[0] - game_surf_pos[0],
                mouse_pos[1] - game_surf_pos[1]
            )
            
            if (not in_constrain(mouse_posrel[0], 0, game_surf.get_size()[0]-1) 
                or  not in_constrain(mouse_posrel[1], 0, game_surf.get_size()[1]-1)):
                continue
            
            if (game.ended):
                game = Game()
                GAME_ENDED = False # type: ignore
                continue
            
            if (game.turn == NN_turn): continue
            
            (status, coord) = click_inside_game(game, game_surf, mouse_posrel)
            
            if (status):
                if (game.move(coord[0])):
                    if (game.check_win() > -1):
                        GAME_ENDED = True # type: ignore
                    if (not game.are_empty_cells()):
                        GAME_ENDED = True # type: ignore
                        
    keys = pygame.key.get_pressed()
    
    if (keys[pygame.K_r]):
        game = Game()
        GAME_ENDED = False # type: ignore

    if (game.turn == NN_turn):
        players = 0
        nns = 0
        if (NN_turn == 0):
            nns = GM.CellEnum.FILLED_P1.value
            players = GM.CellEnum.FILLED_P2.value
        else:
            nns = GM.CellEnum.FILLED_P2.value
            players = GM.CellEnum.FILLED_P1.value
        columns = NN.get_columns(game.gamemap, game.rows, game.columns, nns, players) # type: ignore
        for c in columns:
            if game.move(c):
                break
        if (game.check_win() > -1):
            GAME_ENDED = True # type: ignore
        if (not game.are_empty_cells()):
            GAME_ENDED = True # type: ignore
        
    
    draw_game(game, game_surf)
    window.blit(game_surf, game_surf_pos)
    wn.blit(window, (0,0))
    pygame.display.update()