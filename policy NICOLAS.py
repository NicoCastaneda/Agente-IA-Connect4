import math
import os
import numpy as np
from connect4.policy import Policy
#from typing import override
import json
from connect4.connect_state import ConnectState


class NICOLAS(Policy):

    def serializar(self, tablero): #es para convertir el tablero en una cadena unica
        return str(tuple(map(tuple, tablero)))
    
    def acciones_legales(self, tablero): #si una oclumna esta llena 
        col_validas= []

        for col in range(7):
            if tablero[0, col]== 0: #si la fila de arriba esta vacia
                col_validas.append(col)
        return col_validas
    

    def estado_raiz(self):
        tablero_vacio = np.zeros((6,7), dtype=int)
        s0= self.serializar(tablero_vacio)

        self.Q_global[s0]= {} 
        self.N_global[s0]= {}

        #elejir el json segun el color
        if self.mi_color==-1:
            json=self.mejores_rojo["0"]
        else:
            json=self.mejores_amarillo["0"]
        
        #llenar las acciones guardadas
        for item in json:
            col = item["columna"]
            q  = item["q_value"]

            self.Q_global[s0][col] = q
            self.N_global[s0][col] = 1


    def ucb(self, state_key, tablero): 
        acciones=self.Q_global[state_key]
        total= sum(self.N_global[state_key].values())
        mejor=None
        mejor_valor=-999

        for a in acciones.keys():
            if self.N_global[state_key][a]==0:
                ucb=999
            else:
                q=self.Q_global[state_key][a]
                n=self.N_global[state_key][a]
                ucb= q + math.sqrt(math.log(total+1)/(n)) #formula del ucb

            if ucb > mejor_valor:
                mejor_valor=ucb
                mejor=a
        return mejor

    def sim_random(self, tablero, color): #terminar la partida con jugadas random
        s=ConnectState(tablero.copy(), player=color)

        while not s.is_final():
            acciones=s.get_free_cols()
            c=np.random.choice(acciones)

            try: #para que por si acaso no se muera la simulacion y lo de como empate
                s=s.transition(c)
            except:
                reward=0
                break 
            
        #sabwr quien gano
        ganador=s.get_winner()
        if ganador== color:
            reward = 1 #ganar
        elif ganador==0:
            reward = 0 #empatar
        else:
            reward = -1 #perder
            
        self.rollout_rewards.append(reward)
        return reward



    def mcts(self, tablero, color):

        tablero_copia= tablero.copy()
        state=ConnectState(tablero_copia, player=color)

        camino=[]
        state_key= self.serializar(state.board)

        #seleccion
        while state_key in self.Q_global and len(self.Q_global[state_key])>0:
            a=self.ucb(state_key, state.board)
            camino.append((state_key, a))

            state= state.transition(a)

            #cambiar de turno
            #color=color*-1
            state_key= self.serializar(state.board)

            if state.is_final(): #acabar si ya es el estado final
                break
        
        #expansion
        if state_key not in self.Q_global:
            self.Q_global[state_key]={}
            self.N_global[state_key]={}

            for a in self.acciones_legales(state.board):
                self.Q_global[state_key][a]=0
                self.N_global[state_key][a]=0

        #simulacion
        reward= self.sim_random(state.board, state.player)

        #backpropagation
        for (s,a) in camino:
            self.N_global[s][a] +=1
            n= self.N_global[s][a]
            qanterior= self.Q_global[s][a]
            self.Q_global[s][a] = qanterior + (reward - qanterior)/n


    def expandir_estado(self, tablero, state_key):
        self.Q_global[state_key]={}
        self.N_global[state_key]={}

        for a in self.acciones_legales(tablero):
            self.Q_global[state_key][a]=0
            self.N_global[state_key][a]=0
        
        for i in range(60): #cuantas simulaciones se van a hacer para cada estado
            self.mcts(tablero, self.mi_color)


    #@override
    def mount(self, time_out: int) -> None:
        self.time_out=time_out
        self.Q_global={}
        self.N_global={}
        
        self.rollout_rewards=[] #para ver como van las simulaciones en las graficas

        self.mi_color= None #mientras se sabe si es rojo o amarillo
        self.estado_raiz_iniciado= False

        #cargar los jsons
        dir_path = os.path.dirname(__file__)  # carpeta donde está policy.py
        rojo_path = os.path.join(dir_path, "mejores_rojo.json")
        amarillo_path = os.path.join(dir_path, "mejores_amarillo.json")
        
        #json del rojo
        with open(rojo_path, 'r') as f:
            self.mejores_rojo = json.load(f)

        #json del amarillo
        with open(amarillo_path, 'r') as f:
            self.mejores_amarillo = json.load(f)

        

    #@override
    def act(self, s: np.ndarray) -> int:

        if self.mi_color is None:
            if np.count_nonzero(s == 1) == np.count_nonzero(s == -1):
                self.mi_color = -1 #rojo
            else:
                self.mi_color = 1 #amarillo
        
        #iniciar el estado incial con el color correcto
        if not self.estado_raiz_iniciado:
            self.estado_raiz()
            self.estado_raiz_iniciado=True

        #serializar el estado actual
        state_key= self.serializar(s)

        #si este no existe en el q global entonces se empieza a expandir a partir de ahi
        if state_key not in self.Q_global:
            self.expandir_estado(s, state_key)

        #elegir la mejor accion segun los q values
        acciones=self.Q_global[state_key]

# si no hay q values para ese estado jugar random valido
        if not acciones:
            cols_libres = self.acciones_legales(s) 
            return int(np.random.choice(cols_libres))
        
        mejor_qvalue=-999
        mejor_accion=None

        for accion, q_value in acciones.items():
            if q_value > mejor_qvalue:
                mejor_qvalue=q_value
                mejor_accion=accion
        return mejor_accion