from __future__ import annotations
import game
import nn
import random
import os

class GamePair:
    def __init__(self, rows, columns, NN1: nn.game_NN, NN2: nn.game_NN):
        self.game = game.Game(rows, columns)
        self.NN1 = NN1.copy()
        self.NN2 = NN2.copy()
    
    def act(self):
        if (self.game.ended): 
            return
        
        if (not self.game.are_empty_cells()): return
        
        match self.game.turn:
            case 0:
                mine = game.CellEnum.FILLED_P1.value
                opponents = game.CellEnum.FILLED_P2.value
            case 1:
                mine = game.CellEnum.FILLED_P2.value
                opponents = game.CellEnum.FILLED_P1.value
                
        columns = self.NN1.get_columns(
                    self.game.gamemap, 
                    self.game.rows, self.game.columns,
                    mine, opponents)
        
        for c in columns:
            if self.game.move(c):
                break
        
        self.game.check_win()
            

class Batch:
    def __init__(self, rows, columns, NNs: list[nn.game_NN]):
        self.rows = rows
        self.columns = columns
        self.nets = NNs.copy()
        self.results = [0.0 for _ in range(len(NNs))]
        
        # for realtime outputing
        self.pairs = []
        self.num_pair = []
    
    def play(self, iterations=5):
        for ii in range(iterations):
            print(f"\tIteration {ii+1}/{iterations}")
            self.pairs: list[GamePair] = []
            self.num_pairs: list[tuple[int,int]] = []
            
            rng = list(range(len(self.nets)))
            
            while (len(rng) > 1):
                p1 = random.choice(rng)
                rng.remove(p1)
                p2 = random.choice(rng)
                rng.remove(p2)
                self.num_pairs.append((p1, p2))
                self.pairs.append(GamePair(
                    self.rows, self.columns, self.nets[p1], self.nets[p2]
                ))
            
            stopflag = False
            finished_games = 0
            reported_finished_games = None
            while (not stopflag):
                finished_games = 0
                for gp in self.pairs:
                    for _ in range(4):
                        gp.act()
                    if (gp.game.ended): finished_games += 1
                    
                if (finished_games != reported_finished_games):
                    print(f"\t\t{finished_games}/{len(self.pairs)} games finished")
                    reported_finished_games = finished_games
                    if (finished_games == len(self.pairs)): 
                        stopflag = True
                        print("\t\tbreak")
                        break
            
            for i in range(len(self.pairs)):
                (p1, p2) = self.num_pairs[i]
                gp = self.pairs[i]

                if (gp.game.winner == -1): # nobody won
                    self.results[p1] += 0.5 - 1/gp.game.number_of_turns
                    self.results[p2] += 0.5 - 1/gp.game.number_of_turns
                elif (gp.game.winner == 0): # P1 won
                    self.results[p1] += 1 + 1/gp.game.number_of_turns
                    self.results[p2] += - 1/gp.game.number_of_turns
                elif (gp.game.winner == 1): # P2 won
                    self.results[p1] += - 1/gp.game.number_of_turns
                    self.results[p2] += 1 + 1/gp.game.number_of_turns
        
        for i in range(len(self.results)):
            self.results[i] = self.results[i]/iterations
    
    def save(self, directory):
        if (os.path.exists(directory)):
            os.rmdir(directory)
        os.mkdir(directory)
        
        if directory[-1] != '\\' and directory[-1] != '/':
            directory += '/'
        
        for i in range(len(self.nets)):
            self.nets[i].save(directory + f"NN_{self.rows}x{self.columns}_i_{i}_R_{int(round(self.results[i], 2)*100)}.keras")

    @staticmethod
    def load(rows, columns, directory) -> Batch:
        if directory[-1] != '\\' and directory[-1] != '/':
            directory += '/'
        
        files = [f for f in os.listdir(directory) if os.path.isfile(directory + f) and f.find(".keras")!=-1]
        
        NNs = []
        for f in files:
            NN = nn.game_NN.load_from(directory + f)
            NNs.append(NN)
        
        B = Batch(rows, columns, NNs)
        return B


def next_step(b: Batch, best_k:float=0.3, random_k:float=0.3, inner_mix_k:float=0.5, **kwargs) -> Batch:
    N = len(b.nets)
    N_best = int(N*best_k)
    N_rand = int(N*random_k)
    
    N_mixed = N - N_best - N_rand
    N_inner_mixed = int(N_mixed*inner_mix_k)
    N_outer_mixed = N_mixed - N_inner_mixed
    
    if (N - N_best - N_rand - N_inner_mixed - N_outer_mixed != 0):
        N_rand += (N - N_best - N_rand - N_inner_mixed - N_outer_mixed)
    
    next_nets: list[nn.game_NN] = []
    
    all_results = sorted(list(enumerate(b.results)), key=lambda e: e[1], reverse=True)
    bests = all_results[0:N_best]
    print(f"Best result: {bests[0]}")
    
    for (i, _) in bests:
        next_nets.append(
            b.nets[i].copy()
        )
    
    for _ in range(N_rand):
        next_nets.append(
            nn.game_NN(
                nn.keras.models.clone_model(b.nets[0].model)
            )
        )
    
    for _ in range(N_inner_mixed):
        n1 = random.choice(bests)
        n2 = random.choice(bests)
        while n1==n2:
            n2 = random.choice(bests)
        
        next_nets.append(
            nn.game_NN.merge(
                b.nets[n1[0]],
                b.nets[n2[0]],
                **kwargs
            )
        )
    
    for _ in range(N_outer_mixed):
        n1 = random.choice(bests)
        n2 = random.choice(all_results)
        while (n2 in bests):
            n2 = random.choice(all_results)
        
        next_nets.append(
            nn.game_NN.merge(
                b.nets[n1[0]],
                b.nets[n2[0]],
                **kwargs
            )
        )
    
    new_b = Batch(b.rows, b.columns, next_nets)
    
    return new_b



def evolve(start_batch: Batch, epochs: int) -> Batch:
    batch = start_batch
    
    for i in range(epochs):
        print(f"Epoch {i+1}/{epochs}")
        batch.play()
        if (i != epochs-1):
            batch = next_step(batch)
        
    return batch


def random_batch(rows: int, columns: int, size:int, *args, **kwargs) -> Batch:
    NNs: list[nn.game_NN] = []
    for _ in range(size):
        NNs.append(
            nn.game_NN.new(rows, columns, *args, **kwargs)
        )
    b = Batch(rows, columns, NNs)
    return b