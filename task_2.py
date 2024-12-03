import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import random


# Функция для вычисления евклидова расстояния между двумя точками
def euclidean_distance(a, b):
    return np.linalg.norm(a - b)


# Реализация алгоритма K-means
def k_means(X, n_clusters, max_iters=100):
    # Включение интерактивного режима
    plt.ion()

    # Выбор случайных точек из данных для инициализации центроидов кластеров
    centroids_idx = random.sample(range(X.shape[0]), n_clusters)
    centroids = X[centroids_idx]

    for i in range(max_iters):
        # Создаем пустой список для хранения точек каждого кластера
        clusters = [[] for _ in range(n_clusters)]

        # Распределяем каждую точку данных в ближайший кластер
        for point in X:
            distances = [euclidean_distance(point, centroid) for centroid in centroids]
            cluster_idx = np.argmin(distances)
            clusters[cluster_idx].append(point)

        # Сохраняем предыдущие центроиды для проверки сходимости позже
        previous_centroids = centroids.copy()

        # Пересчитываем центроиды как среднее значение всех точек в кластере
        for idx, cluster in enumerate(clusters):
            if cluster:  # Проверка на то, что кластер не пустой
                centroids[idx] = np.mean(cluster, axis=0)

        # Визуализация текущей итерации
        plt.figure(figsize=(8, 5))
        colors = ['r', 'g', 'b', 'y', 'c'][:n_clusters]  # Ограничение цветов для количества кластеров
        for idx, cluster in enumerate(clusters):
            cluster = np.array(cluster)
            plt.scatter(cluster[:, 0], cluster[:, 1], color=colors[idx], label=f'Cluster {idx + 1}')
            plt.scatter(previous_centroids[idx][0], previous_centroids[idx][1], color='k',
                        marker='x')  # Старый центроид
            plt.scatter(centroids[idx][0], centroids[idx][1], color=colors[idx], marker='o', edgecolor='k', s=150)
        plt.title(f'Iteration {i + 1}')
        plt.legend()
        plt.draw()  # Рисуем текущий график
        plt.pause(3)  # Пауза на 3 секунды
        plt.close()  # Закрываем текущий график

        # Проверка на сходимость (если центроиды не изменились — завершить цикл)
        if np.all(previous_centroids == centroids):
            break

    # Выключение интерактивного режима
    plt.ioff()


# Загрузка набора данных Iris
iris = load_iris()
data = iris.data

# Запуск K-means с установленным оптимальным количеством кластеров
optimal_clusters = 3
k_means(data, optimal_clusters)
