from __future__ import annotations

import tensorflow as tf
import keras
import numpy as np

import game

class game_qNN:
    def __init__(self, model: keras.Model) -> None:
        self.model = keras.models.clone_model(model)    
        self.model.set_weights(model.get_weights())
        self.model.compile(optimizer="Adam")
    
    @staticmethod
    def new(rows, columns, inner_layers=None, inner_activation=None, output_activation=None) -> game_qNN:
        if inner_layers==None:
            inner_layers = [rows*columns//2, rows*columns//4]
        if inner_activation==None:
            inner_activation = keras.activations.tanh
        if output_activation==None:
            output_activation = keras.activations.tanh
        
        input_layer = keras.layers.Input(shape=tuple([columns*rows]))
        flatten_layer = keras.layers.Flatten()(input_layer)
        hidden_layer = flatten_layer
        for n in inner_layers:
            hidden_layer = keras.layers.Dense(n, activation = inner_activation)(hidden_layer)
        output_layer = keras.layers.Dense(columns, activation = output_activation)(hidden_layer)
        
        model = keras.Model(inputs=input_layer, outputs=output_layer)
        NN = game_qNN(model)
        
        return NN

    def copy(self) -> game_qNN:
        model = keras.models.clone_model(self.model)    
        model.set_weights(self.model.get_weights())   
        NN = game_qNN(model)
        return NN
    
    def save(self, filename, *args, **kwargs):
        self.model.save(filename, *args, **kwargs)
        
    @staticmethod
    def load_from(filename) -> game_qNN:
        model = keras.models.load_model(filename)
        NN = game_qNN(model) # type: ignore
        return NN
    
    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)
    
    def get_columns(self, gamemap: list[ list[int] ], rows, columns, mine: int, opponents: int) -> tuple[list[int], list[tuple[int,float]]]:
        # config = self.model.get_config()
        # shape = config["layers"][0]["config"]["batch_shape"]
        # shape = tuple(filter(None, list(shape))) # (columns, rows)
        
        # nn_input = np.zeros((columns*rows))
        nn_input = [0.0 for _ in range(columns*rows)]
        
        for y in range(rows):
            for x in range(columns):
                v = gamemap[y][x]
                set_v = 0
                if v==mine:
                    set_v = 1
                    # print("mine")
                elif v==opponents:
                    set_v = -1
                    # print("opponents")
                nn_input[x + y*columns] = float(set_v)
        
        nn_input = np.array([nn_input])
        # nn_input =  [nn_input]
        
        raw_result = self.model.predict(nn_input, verbose=0) # type: ignore
        # print(raw_result)
        clean_result = []
        
        # for r in raw_result[0]:
        #     clean_result.append( r[0] )
        clean_result = raw_result[0]
        clean_result = np.array(clean_result).tolist()
        # print(clean_result)
        
        zipped_result = list(enumerate(clean_result))
        zipped_result.sort(key = lambda e: e[1], reverse=True)
        # print(zipped_result)
        
        return ([e[0] for e in zipped_result], zipped_result)
        