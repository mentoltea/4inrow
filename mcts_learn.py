import game as Game
import mcts
import qnn
import random
import numpy as np
import math
import os
import tensorflow
import keras

# NN = qnn.game_qNN.new(6, 7)
# epochs = 10


def generate_data(N: int, min_moves: int, max_moves: int, depth:int=6, verbose=False) -> list[tuple[np.ndarray, np.ndarray]]:
    data: list[tuple[np.ndarray, np.ndarray]] = []
    if verbose: print("Generating...")
    
    for n in range(N):
        if verbose: print(f"\tPair {n+1}/{N}")
        
        gm = Game.Game(6,7)
        flag = True
        while flag:
            m = random.randint(min_moves, max_moves)
            gm.random_moves(m)
            if (gm.ended):
                gm = Game.Game(6,7)
            else:
                flag = False
        # gm.print()
        if verbose: print(f"\t\tGame generated")
        
        # (nn_moves, stat) = NN.get_columns(gm, gm.turn)
        # stat.sort(key=lambda t: t[0])
        # stat = list(map(lambda t: t[1], stat))
        # print(f"Output:  {stat}")
        
        expected: list[float] = []
        for m in range(gm.columns):
            pos = gm.copy()
            v = 0
            if (pos.move(m)):
                game_loss = mcts.loss(pos, gm.turn, depth)
                v = 2*math.exp(3/2*(-game_loss)) - 1
            else:
                v = 0
            expected.append(v)
        # print(f"Expected:  {expected}")
        if verbose: print(f"\t\tExpected output generated")
        
        nn_input = np.array(gm.gamemap, dtype=np.float64)
        expected_output = np.array(expected, dtype=np.float64)
        
        data.append( (nn_input, expected_output) )
    
    return data


def save_data(dirname:str , data: list[tuple[np.ndarray, np.ndarray]]):
    if dirname[-1]!='/' and dirname[-1]!='\\':
        dirname += '/'
    if (not os.path.exists(dirname)):
        os.mkdir(dirname)
    
    for (i, d) in enumerate(data):
        subdirname = f"{i}/"
        os.mkdir(dirname+subdirname)
        
        np.save(dirname + subdirname + "input.npy", d[0])
        np.save(dirname + subdirname + "expected.npy", d[1])
    
    
def load_data(dirname:str) -> list[tuple[np.ndarray, np.ndarray]]:
    if dirname[-1]!='/' and dirname[-1]!='\\':
        dirname += '/'
        
    data: list[tuple[np.ndarray, np.ndarray]] = []
    for root, dirs, files in os.walk(dirname):
        if len(files)==0: continue
        input_data = np.load(root + "/input.npy")
        expected_data = np.load(root + "/expected.npy")
        
        data.append( (input_data, expected_data) )
    
    return data

def train_model(model_orig: keras.Model, data: list[tuple[np.ndarray, np.ndarray]],
                training_koef:float=0.7,
                optimizer:keras.optimizers.Optimizer = keras.optimizers.Adam(learning_rate=0.01),
                loss:keras.losses.Loss = keras.losses.MeanSquaredError(),
                **kwargs) -> keras.Model:
    model = keras.models.clone_model(model_orig)
    model.set_weights(model_orig.get_weights())
    model.compile(optimizer, loss) # type: ignore
    
    
    N = len(data)
    N_train = int(N*training_koef)
    N_valid = N - N_train
    
    Xs = list(map(lambda t: t[0], data))
    Ys = list(map(lambda t: t[1], data))
    
    X_train = np.array(Xs[:N_train])
    Y_train = np.array(Ys[:N_train])
    
    X_valid = np.array(Xs[N_train:])
    Y_valid = np.array(Ys[N_train:])
    
    print("Training...")
    
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_valid, Y_valid),
        **kwargs
        # batch size
        # epochs
    )
    
    # print('\nhistory dict:', history.history)
    
    return model


NN = qnn.game_qNN.new(6, 7, inner_layers=[63, 32, 16])
data = generate_data(100, 13, 17, depth=3, verbose=True)
# save_data('data', data)
# data = load_data('data')
g = Game.Game()
g.gamemap = data[0][0]
print(NN.get_columns(g, 1))

mod = train_model(NN.model, data, verbose=0, batch_size=2, epochs=200)
NN.model = mod

print(NN.get_columns(g, 1))
print(sorted(enumerate(data[0][1]), key=lambda t:t[0], reverse=True))
