import numpy as np  # Для роботи з масивами та випадковими числами
from sklearn.datasets import make_moons, make_circles  # Вбудовані датасети

# ----------------------------
# XOR Dataset
# ----------------------------
def make_xor(n=600, noise=0.2, rng=None):
    """Генерація XOR-подібного датасету"""
    if rng is None:
        rng = np.random.default_rng()  # Ініціалізація генератора випадкових чисел
    X = rng.uniform(-1, 1, size=(n, 2))  # Рівномірний розподіл точок у квадраті [-1,1]x[-1,1]
    y = (X[:, 0] * X[:, 1] > 0).astype(int)  # Мітки: 1 якщо x*y>0, інакше 0
    X += rng.normal(0, noise, size=X.shape)  # Додаємо гаусівський шум
    return X, y  # Повертаємо ознаки та мітки

# ----------------------------
# Спіральний датасет
# ----------------------------
def make_spirals(n=600, noise=0.2, rng=None, turns=3):
    """Генерація спірального датасету"""
    if rng is None:
        rng = np.random.default_rng()
    n2 = n // 2  # Кількість точок на одну спіраль
    theta = np.linspace(0, turns * 2*np.pi, n2)  # Кутові координати
    r = np.linspace(0.0, 1.0, n2)  # Радіальні координати
    x1 = np.c_[r*np.cos(theta), r*np.sin(theta)]  # Перша спіраль
    x2 = np.c_[r*np.cos(theta + np.pi), r*np.sin(theta + np.pi)]  # Друга спіраль (180° зсув)
    X = np.vstack([x1, x2]) + rng.normal(0, noise, size=(2*n2, 2))  # Додаємо шум
    y = np.hstack([np.zeros(n2, dtype=int), np.ones(n2, dtype=int)])  # Мітки класів
    return X, y

# ----------------------------
# Кільцевий датасет (Rings)
# ----------------------------
def make_rings(n=600, rng=None, noise=0.1):
    """Генерація концентричних кіл"""
    if rng is None:
        rng = np.random.default_rng()
    n2 = n // 2
    angles1 = rng.uniform(0, 2*np.pi, n2)  # Кути для внутрішнього кільця
    angles2 = rng.uniform(0, 2*np.pi, n2)  # Кути для зовнішнього кільця
    r1 = rng.normal(0.5, noise, n2)  # Радіус внутрішнього кільця
    r2 = rng.normal(1.0, noise, n2)  # Радіус зовнішнього кільця
    x1 = np.c_[r1*np.cos(angles1), r1*np.sin(angles1)]  # Координати внутрішнього кільця
    x2 = np.c_[r2*np.cos(angles2), r2*np.sin(angles2)]  # Координати зовнішнього кільця
    X = np.vstack([x1, x2])
    y = np.hstack([np.zeros(n2, dtype=int), np.ones(n2, dtype=int)])
    return X, y

# ----------------------------
# Шаховий датасет (Checker)
# ----------------------------
def make_checker(n=600, rng=None, noise=0.1, tiles=4):
    """Генерація шахового (checkerboard) датасету"""
    if rng is None:
        rng = np.random.default_rng()
    s = int(np.sqrt(n))  # Розмір сітки
    x1 = np.linspace(-1, 1, s)
    x2 = np.linspace(-1, 1, s)
    XX, YY = np.meshgrid(x1, x2)  # Створення сітки координат
    X = np.c_[XX.ravel(), YY.ravel()]  # Перетворення в масив точок
    # Обчислення міток: шахова структура
    y = (((np.floor((XX + 1)*tiles/2) + np.floor((YY + 1)*tiles/2)) % 2) > 0).astype(int).ravel()
    X += rng.normal(0, noise, size=X.shape)  # Додаємо шум
    idx = rng.choice(len(X), size=n, replace=True)  # Вибір n точок випадково
    return X[idx], y[idx]

# ----------------------------
# Завантаження датасету за назвою
# ----------------------------
def load_dataset(kind, n, rng):
    """Функція для генерації або завантаження різних датасетів"""
    if kind == "moons":
        X, y = make_moons(n_samples=n, noise=0.25, random_state=0)
    elif kind == "circles":
        X, y = make_circles(n_samples=n, noise=0.1, factor=0.5, random_state=0)
    elif kind == "xor":
        X, y = make_xor(n=n, noise=0.15, rng=rng)
    elif kind == "spirals":
        X, y = make_spirals(n=n, noise=0.08, rng=rng, turns=3)
    elif kind == "rings":
        X, y = make_rings(n=n, rng=rng, noise=0.05)
    elif kind == "checker":
        X, y = make_checker(n=n, rng=rng, noise=0.03, tiles=6)
    else:
        raise ValueError("Unknown dataset name.")
    return X, y
