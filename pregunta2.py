import face_recognition
import numpy as np
import os
import random
import matplotlib.pyplot as plt


dataset_path = "./lfw"

# Lista para almacenar las incrustaciones de los rostros
embeddings = []

# Recorremos el dataset
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        # Cargamos
        image = face_recognition.load_image_file(os.path.join(root, file))
        
        # Obtenemos las incrustaciones de los rostros
        face_bboxes = face_recognition.face_locations(image)
        face_emb = face_recognition.face_encodings(image, face_bboxes)

        # Agregamos las incrustaciones de los rostros a la lista
        embeddings.extend(face_emb)

embeddings = np.stack(embeddings)

N = 100

# Lista para almacenar las distancias euclidianas
distances = []

for i in range(N):
    # Seleccionar dos rostros al azar
    face1, face2 = random.sample(embeddings, 2)

    # distancia Euclidiana
    dist = np.linalg.norm(face1 - face2)
    distances.append(dist)

# Grafico
plt.hist(distances, bins=50)
plt.show()
