import math
import os
import numpy as np
from connect4.policy import Policy
#from typing import override
import json
from connect4.connect_state import ConnectState
import time


class MAN(Policy):

    def serializar(self, tablero): #es para convertir el tablero en cadena unica de tuplas para usarla como key del diccionario
        return str(tuple(map(tuple, tablero)))
    

    def acciones_legales(self, tablero): #columnas validas
        col_validas=[]

        for col in range(7):
            if tablero[0, col]==0: #si al parte de arriba esta libre (fila 0, columna x)
                col_validas.append(col)

        return col_validas
    

    def sim_random(self, tablero, color): #ya no es random sino que usa a mateo
        s=ConnectState(tablero.copy(), player=color)

        while not s.is_final():
           
            if np.count_nonzero(s.board == 1) == np.count_nonzero(s.board == -1): #color del sigueionte
                myColor = -1
            else:
                myColor = 1

            state = ConnectState(s.board,player=myColor)

            #1) Gano yo?
            mov=None #para guardar la jugada que se haria en ese estado en especifico
            for c in state.get_free_cols():
                try:
                    newState = state.transition(c)
                    if newState.get_winner() == myColor:
                        mov=c
                        break
                except:
                    continue

            #2) Gana el otro?
            if mov is None: #si no gane yo no entro al if entonces mov no se asigana a c
                stateOponent = ConnectState(s.board,player=-myColor)
                for c in state.get_free_cols():
                    try:
                        newState = stateOponent.transition(c)
                        if newState.get_winner() == -myColor:
                            mov=c
                            break
                    except:
                        continue

            #3)  Si no juego centro,lasiguiente del centro y la anterior PERO que al jugar ahi no le permita ganar al otro  
            if mov is None:
                center_options = [3, 4, 2]
                noPlay=[]
                for c in state.get_free_cols():
                    try:
                        newState = state.transition(c)
                        stateOponent = ConnectState(newState.board,player=-myColor)
                        for c2 in stateOponent.get_free_cols():
                            newStateOponent = stateOponent.transition(c2)
                            if newStateOponent.get_winner() == -myColor:
                                noPlay.append(c)
                    except:
                        continue

                available_center_options = [c for c in center_options if state.is_applicable(c) and c not in noPlay]
                if available_center_options:
                    mov=int(available_center_options[0])
                else:
                    
                    possiblePlays=[c for c in state.get_free_cols() if c not in noPlay]
                    if not possiblePlays:
                        #para evitar el cannot be empty si no hay posible plays
                        posibles=state.get_free_cols()
                        if not posibles:
                            break
                        mov=int(np.random.choice(posibles))
                    #5) si no aleatorio, PERO que al jugar ahi no le permita ganar al otro
                    else:
                        rng=np.random.default_rng()
                        mov=int(rng.choice(possiblePlays))

            try:
                s=s.transition(mov)
            except:
                reward=0
                break

        ganador=s.get_winner()
        if ganador==color:
            reward=1
        elif ganador==0:
            reward=0
        else:
            reward=-1

        self.rollout_rewards.append(reward)
        return reward
    

    def mcts(self, tablero, color, t): #busqueda mcts
        state_root=ConnectState(tablero.copy(), player=color)
        root_key=self.serializar(state_root.board)

        Q={}
        N={}

        def estado_inicial(state, key, es_raiz): #inicializar estado del arbol
            Q[key]={}
            N[key]={}

            libres=state.get_free_cols()

            if es_raiz:
                if color==-1:
                    json_val=self.mejores_rojo[str(t)] if str(t) in self.mejores_rojo else []
                else:
                    json_val=self.mejores_amarillo[str(t)] if str(t) in self.mejores_amarillo else []

                usados=set()

                for item in json_val:
                    col=item["columna"]
                    q=item["q_value"]

                    if col in libres:
                        Q[key][col]= q
                        N[key][col]=1
                        usados.add(col)

                for col in libres:
                    if col not in usados:
                        Q[key][col]=0.0
                        N[key][col]=0

            else:
                for col in libres:
                    Q[key][col]=0.0
                    N[key][col]=0

        estado_inicial(state_root, root_key, True)

        start_time=time.time()
        maxt=self.time_out - 1 #un colchonsito pa no pasase del time out

        #iteraciones
        for i in range(self.budget): #inner trial

            if time.time()-start_time>maxt:
                break

            state=state_root
            key=root_key
            camino=[]

            #seleccion
            while True:
                if state.is_final():
                    break

                acciones=list(Q[key].keys())
                total=sum(N[key].values())+1

                mejor=None
                mejor_valor=-999999

                for a in acciones:
                    n=N[key][a]
                    q=Q[key][a]

                    if n==0:
                        ucb=999999
                    else:
                        ucb=q+math.sqrt(math.log(total)/n)

                    if ucb>mejor_valor:
                        mejor_valor=ucb
                        mejor=a

                camino.append((key, mejor, state.player))
                state=state.transition(mejor)

                key=self.serializar(state.board)

                if key not in Q:
                    estado_inicial(state, key, False)
                    break

            #simulacion
            reward=self.sim_random(state.board, color)

            #backpropagation
            for (k, a, jugador) in reversed(camino):
                valor=reward

                if jugador!=color:
                    valor=-reward

                N[k][a]+=1
                n=N[k][a]
                Q[k][a]+= (valor-Q[k][a])/n

        #elegir la mejor accion
        acciones=Q[root_key]
        mejor_q=-999999
        mejor_a=None

        for a, q in acciones.items():
            if q>mejor_q:
                mejor_q=q
                mejor_a=a

        return int(mejor_a)
    
    #@override
    def mount(self, time_out: int)->None: #cargar jsons de qvalues precargados y parametros
        self.time_out=time_out
        self.rollout_rewards=[] #para ver como van las simulaciones en las graficas
        self.mi_color=None

        dir_path=os.path.dirname(__file__)
        rojo_path=os.path.join(dir_path, "mejores_rojo.json")
        amarillo_path=os.path.join(dir_path, "mejores_amarillo.json")

        with open(rojo_path, 'r') as f:
            self.mejores_rojo=json.load(f)

        with open(amarillo_path, 'r') as f:
            self.mejores_amarillo=json.load(f)

        self.budget=5000
    
    #@override
    def act(self, s: np.ndarray)->int: #elige la jugada final
        if self.mi_color is None:
            if np.count_nonzero(s==1)==np.count_nonzero(s==-1):
                self.mi_color=-1
            else:
                self.mi_color=1

        t=int(np.count_nonzero(s))

        return self.mcts(s, self.mi_color, t)