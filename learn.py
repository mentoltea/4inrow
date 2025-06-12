import game
import numpy as np
import random
import Q
# import Q_nn
import conv_nn
import tensorflow
import keras
import os


NN_CLASS = conv_nn.game_conv_NN
TARGET_FUNC = Q.Q
CLASSIFY_FUNC = Q.classify

def generate_data(
    target_function,
    N: int, min_moves: int, max_moves: int, depth:int=6,
    verbose=False, ended=None
) -> list[tuple[np.ndarray, np.ndarray]]:
    data: list[tuple[np.ndarray, np.ndarray]] = []
    if verbose: print("Generating...")
    
    for n in range(N):
        if verbose: print(f"\tPair {n+1}/{N}")
        
        gm = game.Game(6,7)
        flag = True
        m = random.randint(min_moves, max_moves)
        while flag:
            gm.random_moves(m)
            if (gm.ended):
                if (ended==False):
                    gm = game.Game(6,7)
                else:
                    flag = False
            else:
                if (ended==True):
                    pass
                else:
                    flag = False
                
        if verbose:
            gm.print()
        if verbose: print(f"\t\tGame generated")
        
        turn = gm.turn
        
        Q_value: float = target_function(gm, turn, depth_search=depth)
        if verbose: print(f"\t\tQ_value: {Q_value}")
        
        nn_input = np.array(gm.gamemap*turn, dtype=np.float64)
        expected_output = np.array(Q_value, dtype=np.float64)
        
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

if 1:
    data = []
    savedir = "data3/" + "Q_2_"
    n = 500
    d = 5
    i = 0
    flag = True
    
    # for regression
    func = TARGET_FUNC 
    
    # for classification
    # func = lambda position, turn, depth: CLASSIFY_FUNC(TARGET_FUNC(position, turn, depth)) 
    
    ended = None
    # True -> only ended games
    # False -> only not ended games
    # None -> random games (ended & not ended)
    
    while flag:
        print(f"\n\nGenerated {i*n} samples\n")
        try:
            newdata = generate_data(func, n, 1, 28, depth=d, ended=ended, verbose=True)
            data += newdata
        except:
            flag = False
            break
        i += 1

    if (len(data)>0):
        savedir += str(len(data))
        print(f"Saving {len(data)} samples to {savedir}")
        save_data(savedir, data)
else:
    # NN = Q_nn.game_Q_NN.new(6, 7, inner_layers=[48, 32, 16, 7])
    NN = NN_CLASS.new(6, 7, 3, (4,4), inner_layers=[42, 21, 7])
    mod = NN.model
    mod.summary()

    modMAE = mod
    modMSE = mod
     
    data = load_data('data3/Q_117000/')

    batch_size = 64
    epochs = 100

    N = 25
    for I in range(N):
        print(f"Iteration {I+1}/{N}")
        try:
            random.shuffle(data)
            print("MAE:")
            modMAE = train_model(
                modMAE, data,
                training_koef=0.85,
                optimizer='Adam', # type: ignore
                loss=keras.losses.MeanAbsoluteError(),
                batch_size=batch_size,
                epochs=epochs
            )
            print("MSE:")
            modMSE = train_model(
                modMSE, data,
                training_koef=0.85,
                optimizer='Adam', # type: ignore
                loss=keras.losses.MeanSquaredError(),
                batch_size=batch_size,
                epochs=epochs
            )
        except:
            break
    print("Saving...")
    NN.model = modMAE
    NN.save("NN3/MAE.keras")

    NN.model = modMSE
    NN.save("NN3/MSE.keras")
