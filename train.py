import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

import utils.preProcess as preProcess
#from models.model import DetectionModel
from utils.actions import all_actions_training, all_actions
from utils.params_dataset import get_paremeters

DATA_PATH = get_paremeters()['DATA_PATH']
actions_labels = all_actions_training() # Actions that we try to detect
actions = all_actions()
sequence_length = get_paremeters()['sequence_length'] # Videos are going to be 30 frames in length
num_epochs = get_paremeters()['num_epochs']
batch_size = get_paremeters()['batch_size']

print('Loading data...')
sequences, labels = preProcess.pre_processing(actions_labels, DATA_PATH, sequence_length)

X = np.array(sequences)
y = np.array(labels)


model = tf.keras.Sequential([
    # First Conv1D layer with ReLU activation
    layers.Conv1D(64, kernel_size=3, padding='same', activation='relu', input_shape=(30, 258)),
    
    # MaxPooling1D layer with smaller pool size
    layers.MaxPooling1D(pool_size=2, strides=2),  # Reduces sequence length from 30 -> 15
    
    # Second Conv1D layer with ReLU activation
    layers.Conv1D(128, kernel_size=3, padding='same', activation='relu'),
    
    # MaxPooling1D layer with smaller pool size
    layers.MaxPooling1D(pool_size=1, strides=2),  # Reduces sequence length from 15 -> 7
    
    # Dropout layer
    layers.Dropout(0.3),
    
    # LSTM layer (return_sequences=True to preserve sequence)
    #layers.LSTM(64, return_sequences=False),
    layers.Bidirectional(layers.LSTM(64, return_sequences=False)),
    
    # Dense output layer with sigmoid activation for binary classification
    layers.Dense(8, activation='sigmoid')  # num_actions is 10
])


# Compilar el modelo con el optimizador Adam y función de pérdida categórica para clasificación.
model.compile(optimizer='adam', 
              loss='categorical_crossentropy',  # Función de pérdida para problemas de clasificación multiclase
              metrics=['accuracy'])  # Métrica de precisión

# Entrenar el modelo con el dataset aumentado.
# Se entrena por 500 épocas y se usa un 20% del conjunto de datos para validación.
### Pendiente aumentar el dataset



print("Entrenando el modelo con dataset aumentado y Dropout...")
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
history = model.fit(X, y, epochs=15, batch_size=16, validation_split=0.2,callbacks=[early_stopping])

# Guardar el modelo entrenado en un archivo .h5 para usarlo posteriormente.
model.save('mi_modelo.keras')
print("Modelo guardado en 'mi_modelo.keras'.")


# Extraer los datos de la historia de entrenamiento y validación (precisión y pérdida).
acc = history.history['accuracy']  # Precisión en el conjunto de entrenamiento
val_acc = history.history['val_accuracy']  # Precisión en el conjunto de validación
loss = history.history['loss']  # Pérdida en el conjunto de entrenamiento
val_loss = history.history['val_loss']  # Pérdida en el conjunto de validación

# Definir el rango de épocas para graficar
epochs_range = range(len(history.history['accuracy'])) 

# Graficar la precisión y la pérdida tanto para el entrenamiento como para la validación.
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Precisión de Entrenamiento')
plt.plot(epochs_range, val_acc, label='Precisión de Validación')
plt.legend(loc='lower right')
plt.title('Precisión de Entrenamiento y Validación')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Pérdida de Entrenamiento')
plt.plot(epochs_range, val_loss, label='Pérdida de Validación')
plt.legend(loc='upper right')
plt.title('Pérdida de Entrenamiento y Validación')

# Guardar el gráfico en un archivo PNG
plt.savefig('graf_metricas_entrenamiento.png')
