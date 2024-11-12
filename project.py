# Importa as bibliotecas necessárias
#import numpy as np  # Biblioteca para manipulação de arrays e operações matemáticas
#import cv2  # Biblioteca para captura e manipulação de imagens e vídeo
#from tensorflow.keras.models import load_model  # Função para carregar o modelo de IA treinado
# import tensorflow as tf  # Biblioteca para construção e execução de modelos de aprendizado profundo
#from keras.models import load_model

# Exibe a versão do TensorFlow em uso
#print(tf.__version__)
#from keras.models import load_model
from tensorflow.keras.models import load_model

import numpy as np
import cv2

# Carrega o modelo de aprendizado profundo previamente treinado e salvo em um arquivo .h5
model = load_model('Model/Keras_model.h5')

# Cria um array NumPy para armazenar a imagem de entrada no formato esperado pelo modelo (1 imagem, 224x224, 3 canais de cor)
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Inicializa a captura de vídeo da webcam (usando a câmera padrão, índice 0)
cap = cv2.VideoCapture(0)

# Define as classes que o modelo foi treinado para reconhecer
classes = ['CALCULADORA', 'CONTROLE', 'LAMPADA', 'FUNDO']

# Loop infinito para capturar e processar o vídeo em tempo real
while True:

    success, img = cap.read() # Lê um quadro de vídeo da webcam
    imgS = cv2.resize(img, (224, 224))   # Redimensiona o quadro para 224x224 pixels, que é o tamanho de entrada esperado pelo modelo
    image_array = np.asarray(imgS) # Converte a imagem redimensionada em um array NumPy
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1  # Normaliza os valores dos pixels para o intervalo [-1, 1] para ser compatível com o modelo
    data[0] = normalized_image_array   # Atribui a imagem normalizada ao array 'data', que será usado como entrada do modelo
    prediction = model.predict(data)  # Realiza a predição com o modelo usando o array de dados preparado
    indexVal = np.argmax(prediction)     # Encontra o índice da classe com a maior probabilidade na previsão

    # Exibe o nome da classe reconhecida na imagem capturada pela webcam
    cv2.putText(img, str(classes[indexVal]), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)

    # Imprime o nome da classe reconhecida no console
    print(classes[indexVal])

    # Mostra o quadro de vídeo com o nome da classe sobreposto
    cv2.imshow('img', img)
    # Aguarda brevemente antes de processar o próximo quadro (1 milissegundo)
    cv2.waitKey(1)
