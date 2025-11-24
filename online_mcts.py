import math
from typing import Any, Callable, Dict, Iterable, Tuple
import numpy as np
from connect_state import ConnectState
import numpy as np
import json

"""""
Se establecen unas reward de 0.024 a los estados centrales y a aquellos limitan con ellos una
rewar de 0.012, todos los demás tienen unos reward de 0. Esto sigue el principio que los estados
centrales son superiores a los demás.
"""""
def reward()-> Dict[tuple [int,int], float]:
    s = ConnectState()
    s.board[:, 3] = 24
    s.board[:, 2] = 12
    s.board[:, 4] = 12
    R={}
    for row in range (6):
        for col in range (7):
            R[(row,col)]=float(s.board[row,col]/1000)
    return R

"""""
Se creo un diccionario de q_values que va por turno y estado, cada estado tiene un q_values (al inicio son las recompensas),
el contador de acción inicializado en 1 y un contador de trial creado para actualizar solo los estados que se jugaron en el
trial.
"""""

def q_values()-> Dict[int, Dict[tuple[int, int], Dict[str, float | int]]]:
    q={}
    r=reward()
    for t in range (21):
        q[t]={}
        for row in range (6):
            for col in range (7):
                q[t][row,col]= {"count_a": 1, "q_value": r[(row,col)], "trial":0}
    return q
            
"""""
Se realizo un mcts, el jugador rojo y amarillo juegan partidas hasta que uno de los dos gane generando trials y actualizando
los q_values de los estados por los que pase el trial, se establecio que la recompensa final es 1 si gana, 0 si empata y -1 si 
pierde. La acción que realiza cada jugador en su turno se basa en estos principios:
    - Si puede ganar gane
    - Si va perder bloquee el movimiento
    - Escoger la mejor acción teniendo en cuenta el valor del UCBA
    - Si dos tienen el mejor valor escoger aleatorio
"""""
def mcts_uct(qR:dict,qA:dict) -> Tuple[
    Dict[int, Dict[tuple[int, int], Dict[str, float | int]]],
    Dict[int, Dict[tuple[int, int], Dict[str, float | int]]],
    Dict[int, int]
]:
    n_t={}  #Contador de todas las acciones que se realizan en un turno
    for t in range (21):
        n_t[t]=1    #Se pone el valor inicial al contador
    q_r=qR  #Se carga los q_values iniciales
    q_a=qA
    for iteration in range (100000): #Se define la cantidad de trials que se va a hacer
        t=0   #Se inicia el contador de turnos y el juego
        trial = ConnectState()
        
        while trial.is_final() is False: #El trial sigue hasta que acabe
            for i in [-1,1]: #Jugador rojo y amarillo intercambian turnos
                if trial.get_winner() !=-1: #Si gana el rojo no dejar jugar al amarillo
                    a=None  #Validación de que se realizo la acción
                #1) Gano yo?
                    for c in trial.get_free_cols():
                        if a is None:
                            try:
                                newState = trial.transition(c)
                                if newState.get_winner() == i and a is None:
                                    if i == -1:
                                        q_r[t][trial.get_heights()[c], c]["count_a"] +=1
                                        q_r[t][trial.get_heights()[c], c]["trial"] =iteration + 1
                                    else:
                                        q_a[t][trial.get_heights()[c], c]["count_a"] +=1
                                        q_a[t][trial.get_heights()[c], c]["trial"] =iteration + 1
                                    trial = trial.transition(c)
                                    a=True
                                    n_t[t]+=1
                            except:
                                continue

                    #2) Gana el otro?
                    stateOponent = ConnectState(trial.board, player=-i)
                    if a is None:
                        for c in trial.get_free_cols():
                                try:
                                    newState = stateOponent.transition(c)
                                    if newState.get_winner() == -i and a is None:
                                        if i == -1:
                                            q_r[t][trial.get_heights()[c], c]["count_a"] +=1
                                            q_r[t][trial.get_heights()[c], c]["trial"] =iteration + 1
                                        else:
                                            q_a[t][trial.get_heights()[c], c]["count_a"] +=1
                                            q_a[t][trial.get_heights()[c], c]["trial"] =iteration + 1
                                        trial = trial.transition(c)
                                        a=True
                                        n_t[t]+=1
                                except:
                                    continue
                    
                    #2) Escoger mejor acción con UCBA
                    if a is None:
                        d=[]
                        ucba={}
                        maxUcba=0
                        for c in trial.get_free_cols():
                        
                                if i ==-1: 
                                    ucba[c]= q_r[t][trial.get_heights()[c],c]["q_value"] + (math.sqrt((math.log(n_t[t]))/(q_r[t][trial.get_heights()[c],c]["count_a"])))
                                else:
                                    ucba[c]= q_a[t][trial.get_heights()[c],c]["q_value"] + (math.sqrt((math.log(n_t[t]))/(q_a[t][trial.get_heights()[c],c]["count_a"])))
                    
                        
                        maxUcba=max(ucba.values())
                        for c in trial.get_free_cols():
                            if ucba[c] == maxUcba:
                                d.append(c)
                        e = np.random.choice(d) #Se hace aleatorio si da valores iguales
                        e = int(e) 
                        if i ==-1 : #Se actualiza el contador de acción
                            q_r[t][trial.get_heights()[e],e]["count_a"] +=1
                            q_r[t][trial.get_heights()[e],e]["trial"] =iteration + 1
                        else:
                            q_a[t][trial.get_heights()[e],e]["count_a"] +=1
                            q_a[t][trial.get_heights()[e],e]["trial"] = iteration + 1
                        trial=trial.transition(e)
                        a=True                   
                        n_t[t]+=1
        
            t += 1
                    
        if trial.is_final():    #Se actualiza los q_values con la recompensa final según el ganador
            for m in range(t):
                        for state, data in q_r[m].items():
                            if data.get("trial") == iteration+1:
                                q_r[m][state]["q_value"] = q_r[m][state]["q_value"] + (((-1*trial.get_winner()) - q_r[m][state]["q_value"])/(q_r[m][state]["count_a"]))
                        for state, data in q_a[m].items():
                            if data.get("trial") == iteration+1:
                                q_a[m][state]["q_value"] = q_a[m][state]["q_value"] + ((trial.get_winner() - q_a[m][state]["q_value"])/(q_a[m][state]["count_a"]))
                                
    return q_a,q_r

def mostrar_mcts(q_r: dict, q_a: dict,
                 file_rojo="qvalues_rojo.json",
                 file_amarillo="qvalues_amarillo.json"):
    # -------- GUARDAR EN JSON ----------
    def convertir(diccionario):
        salida = {}
        for t, states in diccionario.items():
            salida[str(t)] = {}
            for state, data in states.items():
                salida[str(t)][str(state)] = data
        return salida

    json_rojo_Mateo = convertir(q_r)
    json_amarillo_Mateo = convertir(q_a)

    with open(file_rojo, "w", encoding="utf-8") as f:
        json.dump(json_rojo_Mateo, f, indent=4)

    with open(file_amarillo, "w", encoding="utf-8") as f:
        json.dump(json_amarillo_Mateo, f, indent=4)

    print(f"\nArchivos generados:")
    print(f" → {file_rojo}")
    print(f" → {file_amarillo}")



qR=q_values() 
qA=q_values()                                        

q_a, q_r = mcts_uct(qR, qA)

mostrar_mcts(q_r,q_a)








            
                           
                           

            

