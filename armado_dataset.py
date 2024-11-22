import os
import shutil
import time
import mediapipe as mp
import utils.dataCollection as dataCollection
from utils.actions import all_actions_training

def get_paremeters():
    '''
    Seteo de parametros para el armado del dataset
    '''
    parameters = {
        'DATA_PATH': os.path.join('data/'),
        'no_sequences': 50,
        'start_folder': 1,
        'sequence_length': 30,
        'num_epochs': 70,
        'batch_size': 8
    }
    return parameters


# Armado de dataset:
#   - Grabaciones con la webcam. 
#   - Cada acción se guardará en una carpeta específica

DATA_PATH = get_paremeters()['DATA_PATH']
actions = all_actions_training() # acciones a detectar
no_sequences = get_paremeters()['no_sequences'] # 30 videos
sequence_length = get_paremeters()['sequence_length'] # Videos de 30 frames de largo
start_folder = get_paremeters()['start_folder'] # Directorio inicial


if os.path.isdir(DATA_PATH):
    print(f"La carpeta {DATA_PATH} ya existe")
    user_input = input("Borrar el directorio actual? (si/no): ").strip().lower()

    if user_input in ["si", "y", "s"]:
        # Limpio el directorio        
        shutil.rmtree(DATA_PATH)
        print("TSe borró el directorio y su contenido.")
        os.makedirs(DATA_PATH)
        for i in range(len(actions)):
            os.makedirs(os.path.join(DATA_PATH, actions[i]))
        print(f"Nuevo directorio {DATA_PATH} creado.")
        time.sleep(1)
        
        dataCollection.data_collection(actions, DATA_PATH, no_sequences, sequence_length, start_folder)
    
    elif user_input in ["no", "n"]:
        print("Carpeta no eliminada")
    else:
        print("Respuesta incorrecta.")
else:
    # Creo el directorio si no existe
    os.makedirs(DATA_PATH)
    for i in range(len(actions)):
            os.makedirs(os.path.join(DATA_PATH, actions[i]))
    print(f"The folder has been created.")
    time.sleep(1)
    
    dataCollection.data_collection(actions, DATA_PATH, no_sequences, sequence_length, start_folder)