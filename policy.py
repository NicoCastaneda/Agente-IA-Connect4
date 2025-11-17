import numpy as np
from connect4.policy import Policy
from connect4.connect_state import ConnectState

#from typing import override


class MATEO(Policy):

    #@override
    def mount(self) -> None:
        pass

    #@override
    def act(self, s: np.ndarray) -> int:
        

        if np.count_nonzero(s == 1) == np.count_nonzero(s == -1):
            myColor = -1
        else:
            myColor = 1

        state = ConnectState(s, player=myColor)
        #1) Gano yo?
        for c in state.get_free_cols():
            try:
                newState = state.transition(c)
                if newState.get_winner() == myColor:
                    return c
            except:
                continue

        #2) Gana el otro?
        stateOponent = ConnectState(s, player=-myColor)
        for c in state.get_free_cols():
            try:
                newState = stateOponent.transition(c)
                if newState.get_winner() == -myColor:
                    return c
            except:
                continue

        
        #3) Si no juego centro,lasiguiente del centro y la anterior PERO que al jugar ahi no le permita ganar al otro
        center_options = [3, 4, 2]
        noPlay=[]
        for c in state.get_free_cols():
            try:
                newState = state.transition(c)
                stateOponent = ConnectState(newState.board, player=-myColor)
                for c in stateOponent.get_free_cols():
                    newStateOponent = stateOponent.transition(c)
                    if newStateOponent.get_winner() == -myColor:
                        noPlay.append(c)
            except:
                continue

        available_center_options = [c for c in center_options if state.is_applicable(c) and c not in noPlay]
        if available_center_options:
            return int(available_center_options[0])


        #5) si no aleatorio, PERO que al jugar ahi no le permita ganar al otro
        possiblePlays = [c for c in state.get_free_cols() if c not in noPlay]
        rng = np.random.default_rng()
        return int(rng.choice(possiblePlays))
    
    
