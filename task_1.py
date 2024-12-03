from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Загружаем набор данных Iris
iris_dataset = load_iris()
features = iris_dataset.data

# Инициализируем список для хранения значений SSE (сумма квадратов ошибок)
sse_scores = []

# Перебираем возможное количество кластеров от 1 до 8
for num_clusters in range(1, 9):
    # Создаем и обучаем модель KMeans
    kmeans_model = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans_model.fit(features)
    # Сохраняем значение инерции (SSE)
    sse_scores.append(kmeans_model.inertia_)

# Параметры для построения графика
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(1, 9), sse_scores, marker='o')

# Устанавливаем заголовки и подписи осей
ax.set_title('Метод локтя')
ax.set_xlabel('Количество кластеров')
ax.set_ylabel('SSE')

# Отображаем график
plt.show()