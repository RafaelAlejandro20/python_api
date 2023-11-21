from flask import Flask, render_template
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import zipfile
import cv2
from skimage import io
import os
import random
import plotly.graph_objects as go

app = Flask(__name__)

@app.route('/')
def index():
    # Cambia la ruta al directorio que contiene el archivo CSV
    csv_file_path = 'C:/Users/Usuario/Documents/API-TUMORES/entorno-virtual-python/data_mask.csv'

    # Cargar el archivo CSV
    brain_df = pd.read_csv(csv_file_path)

    
    # Bloque de código 1
    code1 = """
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import zipfile
    import cv2
    from skimage import io
    import tensorflow as tf
    from tensorflow.python.keras import Sequential
    from tensorflow.keras import layers, optimizers
    from tensorflow.keras.applications import DenseNet121
    from tensorflow.keras.applications.resnet50 import ResNet50
    from tensorflow.keras.layers import *
    from tensorflow.keras.models import Model, load_model
    from tensorflow.keras.initializers import glorot_uniform
    from tensorflow.keras.utils import plot_model
    from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
    from IPython.display import display
    from tensorflow.keras import backend as K
    from sklearn.preprocessing import StandardScaler, normalize
    import os
    import glob
    import random
    %matplotlib inline
    
    
    brain_df = pd.read_csv('data_mask.csv')
    """
    results['block1'] = execute_code(code1)

    # Bloque de código 2
    code2 = """
    brain_df.info()
    
    brain_df.head(50)
    
    brain_df.mask_path[1]  #Ruta a la imagen de la MRI
    
    brain_df.image_path[1] #Ruta a la máscara de segmentación
    """
    results['block2'] = execute_code(code2)

    # Bloque de código 3
    code3 = """
    brain_df
    """
    results['block3'] = execute_code(code3)

    # Bloque de código 4
    code4 = """
    brain_df['mask'].value_counts().index
    
    brain_df['mask'].value_counts()
    """
    results['block4'] = execute_code(code4)

    # Bloque de código 5
    code5 = """
    brain_df
    brain_df['mask'].value_counts().index
    """
    results['block5'] = execute_code(code5)
    
    
    code6 = """
    # Usaremos plotly para hacer un diagrama de barras iterativo
    import plotly.graph_objects as go

    fig = go.Figure([go.Bar(x = brain_df['mask'].value_counts().index, y = brain_df['mask'].value_counts())])
    fig.update_traces(marker_color = 'rgb(0,200,0)', marker_line_color = 'rgb(0,255,0)',
                  marker_line_width = 7, opacity = 0.6)
    fig.show()
    """
    results['block6'] = execute_code(code6)
    
    code7 = """
    brain_df.mask_path
    
    brain_df.image_path
    
    plt.imshow(cv2.imread(brain_df.mask_path[623]))
    
    plt.imshow(cv2.imread(brain_df.image_path[623]))
    
    cv2.imread(brain_df.mask_path[623]).max()
    
    cv2.imread(brain_df.mask_path[623]).min()
    """
    results['block7'] = execute_code(code7)

    code8 = """
    # MRI Imagenes por resonancia magnetica
    # Visualización básica visualizaremos imágenes (MRI y Máscaras)
    import random
    fig, axs = plt.subplots(6,2, figsize=(16,32))
    count = 0
    for x in range(6):
        i = random.randint(0, len(brain_df)) # Seleccionamos un indice aleatorio
        axs[count][0].title.set_text("MRI del Cerebro") # Configuramos el titulo para cada imagen
        axs[count][0].imshow(cv2.imread(brain_df.image_path[i])) # Mostramos el MRI
        axs[count][1].title.set_text("Máscara - " + str(brain_df['mask'][i])) # Titulo de la mascará en 0 y 1
        axs[count][1].imshow(cv2.imread(brain_df.mask_path[i])) # Mostrar máscara
        count += 1

    fig.tight_layout()
    """
    results['block8'] = execute_code(code8)

    return render_template('index.html', results=results)


def execute_code(code):
    # Ejecuta el código y devuelve el resultado
    # Aquí debes adaptar la ejecución del código según tus necesidades
    try:
        exec_globals = {}
        exec(code, globals(), exec_globals)
        result = {'output': exec_globals, 'error': None}
    except Exception as e:
        result = {'output': None, 'error': str(e)}
    return result

if __name__ == '__main__':
    app.run(debug=True)
