import math
from typing import Any, Callable, Dict, Iterable, Tuple
import numpy as np
from connect4.connect_state import ConnectState
import numpy as np
import json


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

def q_values()-> Dict[int, Dict[tuple[int, int], Dict[str, float | int]]]:
    q={}
    r=reward()
    for t in range (21):
        q[t]={}
        for row in range (6):
            for col in range (7):
                q[t][row,col]= {"count_a": 1, "q_value": r[(row,col)], "trial":0}
    return q
            
def mcts_uct(qR:dict,qA:dict) -> Tuple[
    Dict[int, Dict[tuple[int, int], Dict[str, float | int]]],
    Dict[int, Dict[tuple[int, int], Dict[str, float | int]]],
    Dict[int, int]
]:
    n_t={}
    for t in range (21):
        n_t[t]=1
    q_r=qR
    q_a=qA
    for iteration in range (10000):
        t=0
        trial = ConnectState()
        
        while trial.is_final() is False:
            for i in [-1,1]:
                if trial.get_winner() !=-1:
                    a=None
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
                    
                    #2) Escoger mejor acción
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
                        e = np.random.choice(d)
                        e = int(e) 
                        if i ==-1 :
                            q_r[t][trial.get_heights()[e],e]["count_a"] +=1
                            q_r[t][trial.get_heights()[e],e]["trial"] =iteration + 1
                        else:
                            q_a[t][trial.get_heights()[e],e]["count_a"] +=1
                            q_a[t][trial.get_heights()[e],e]["trial"] = iteration + 1
                        trial=trial.transition(e)
                        a=True                   
                        n_t[t]+=1
        
            t += 1
                    
        if trial.is_final():
            for m in range(t):
                        for state, data in q_r[m].items():
                            if data.get("trial") == iteration+1:
                                q_r[m][state]["q_value"] = q_r[m][state]["q_value"] + (((-1*trial.get_winner()) - q_r[m][state]["q_value"])/(q_r[m][state]["count_a"]))
                        for state, data in q_a[m].items():
                            if data.get("trial") == iteration+1:
                                q_a[m][state]["q_value"] = q_a[m][state]["q_value"] + ((trial.get_winner() - q_a[m][state]["q_value"])/(q_a[m][state]["count_a"]))
                                
    return q_a,q_r



qR=q_values() 
qA=q_values()                                        
q_a, q_r = mcts_uct(qR, qA)







            
                           
                           

            

