import os
import time
import numpy as np
from rtree import index
import random

def knn(points, q, K):
    distances = np.linalg.norm(points - q, axis=1)
    return np.argpartition(distances, K)[:K]

p = index.Property()
p.dimension = 2 #D
p.buffering_capacity = 3 #M
p.dat_extension = 'data'
p.idx_extension = 'index'

# Números de puntos a insertar en el RTree
num_points = [10**2, 10**3, 10**4, 10**5, 10**6, 10**7, 10**8]

# Valores de K para la búsqueda KNN
k_values = [3, 6, 9]

for N in num_points:

    # Verifica si los archivos de índice y datos existen, y si es así, los elimina
    if os.path.exists('2d_index2.data'):
        os.remove('2d_index2.data')
    if os.path.exists('2d_index2.index'):
        os.remove('2d_index2.index')

    # Crea un nuevo índice RTree
    idx = index.Index('2d_index3', properties=p)

    points = []
    # Inserta N puntos aleatorios en el RTree
    for i in range(N):
        pt = (random.random(), random.random())
        idx.insert(i, pt + pt)
        points.append(pt)

    points = np.array(points)
    
    for K in k_values:
        # Elige un punto aleatorio para la búsqueda KNN
        q = (random.random(), random.random())

        # Comienza a cronometrar el tiempo
        start_time = time.time()

        # Realiza la búsqueda KNN con RTree
        lres = list(idx.nearest(coordinates=q, num_results=K))

        # Calcula el tiempo total de la búsqueda
        total_time_rtree = time.time() - start_time

        # Comienza a cronometrar el tiempo
        start_time = time.time()

        # Realiza la búsqueda KNN con Lineal Scan
        lres = knn(points, q, K)

        # Calcula el tiempo total de la búsqueda
        total_time_scan = time.time() - start_time

        # Imprime el tiempo total
        print(f"N={N}, K={K}, Tiempo RTree={total_time_rtree} ms, Tiempo Lineal Scan={total_time_scan} ms")

    # Elimina el índice RTree para liberar memoria
    idx.close()
