from __future__ import annotations
import game
import numpy as np
import Q
import tensorflow
import keras


class game_Q_NN:
    # map -> flatten -> dense -> single float (Q-value)
    
    def __init__(self, model: keras.Model) -> None:
        self.model = keras.models.clone_model(model)    
        self.model.set_weights(model.get_weights())
        self.model.compile(optimizer="Adam")
    
    @staticmethod
    def new(rows, columns, inner_layers=None, inner_activation=None, output_activation=None) -> game_Q_NN:
        if inner_layers==None:
            inner_layers = [rows*columns//2, rows*columns//4]
        if inner_activation==None:
            inner_activation = keras.activations.tanh
        if output_activation==None:
            output_activation = keras.activations.tanh
        
        input_layer = keras.layers.Input(shape=tuple((rows, columns)))
        flatten_layer = keras.layers.Flatten()(input_layer)
        hidden_layer = flatten_layer
        for n in inner_layers:
            hidden_layer = keras.layers.Dense(n, activation = inner_activation)(hidden_layer)
        output_layer = keras.layers.Dense(1, activation = output_activation)(hidden_layer)
        
        model = keras.Model(inputs=input_layer, outputs=output_layer)
        NN = game_Q_NN(model)
        
        return NN

    def copy(self) -> game_Q_NN:
        model = keras.models.clone_model(self.model)    
        model.set_weights(self.model.get_weights())   
        NN = game_Q_NN(model)
        return NN
    
    def save(self, filename, *args, **kwargs):
        self.model.save(filename, *args, **kwargs)
        
    @staticmethod
    def load_from(filename) -> game_Q_NN:
        model = keras.models.load_model(filename)
        NN = game_qNN(model) # type: ignore
        return NN
    
    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)
    
    def get_columns(self, gm: game.Game, my_turn: int) -> tuple[list[int], list[tuple[int,float]]]:
        res: list[tuple[int,float]] = []
        for c in range(gm.columns):
            pos = gm.copy()
            if (pos.move(c)):
                nn_input = np.array([pos.gamemap*my_turn], dtype=np.float64)
                raw_result = self.model.predict(nn_input, verbose=0) # type: ignore
                value = raw_result[0][0]
                res.append( (c, value) )
        res.sort(key = lambda e: e[1], reverse=True)
        
        return ([e[0] for e in res], res)