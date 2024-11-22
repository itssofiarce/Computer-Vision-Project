import numpy as np

def all_actions():
    return np.array(['pointR', 'pointL', 'Substi', 'DbHit', 'OutofB', 'Form', 'Tiemp', 'TyAf']) # same order as all_actions_training

def all_actions_training():
    return np.array(['pointR', 'pointL', 'Substi', 'DbHit', 'OutofB', 'Form', 'Tiemp', 'TyAf', 'Nada']) # We add the a 'Nothing' action for training

def action_fullname(action):
    if action == 'pointL':
        return 'Punto Izquierda'
    elif action == 'pointR':
        return 'Point Derecha'
    elif action == 'Substi':
        return 'Cambio'
    elif action == 'DbHit':
        return 'Doble Golpe'
    elif action == 'OutofB':
        return 'Fuera'
    elif action == 'Form':
        return 'Formación'
    elif action == 'Tiemp':
        return 'Tiempo Muerto'
    elif action == 'TyAf':
        return 'Toque y Afuera'