import math
import time
import numpy as np
import matplotlib.pyplot as plt


# ---------- разбор строки функции ----------

def make_function(expr: str):
    """
    По строке expr вида 'x + math.sin(math.pi*x)'
    строит функцию f(x), безопасно передавая только math и x.
    """
    allowed = {name: getattr(math, name) for name in dir(math) if not name.startswith("_")}
    # добавим ещё numpy-константы/функции, если нужно
    allowed.update({"np": np, "math": math})

    def f(x: float) -> float:
        return float(eval(expr, {"__builtins__": {}}, {**allowed, "x": x}))

    return f


# ---------- алгоритм Пиявского (поиск глобального минимума) ----------

def global_min_piyavskii(f, a: float, b: float, eps: float, max_iter: int = 1000):
    """
    Глобальный минимум одномерной липшицевой функции на [a, b].
    Алгоритм Пиявского с оценкой константы Липшица по текущим точкам.

    Возвращает:
        x_best, f_best, точки [(x_i, f_i)], число итераций
    """
    # старт: две точки на концах отрезка
    x_points = [a, b]
    f_points = [f(a), f(b)]

    # начальная оценка Липшица
    L = abs(f_points[1] - f_points[0]) / (b - a)
    if L <= 0:
        L = 1.0

    iters = 2  # уже 2 вычисления функции

    while iters < max_iter:
        # сортируем точки по x (на всякий случай)
        pts = sorted(zip(x_points, f_points), key=lambda p: p[0])
        x_points, f_points = [p[0] for p in pts], [p[1] for p in pts]

        # пересчёт L по всем соседним парам (чуть расширяем для надёжности)
        for i in range(len(x_points) - 1):
            dx = x_points[i + 1] - x_points[i]
            if dx == 0:
                continue
            df = abs(f_points[i + 1] - f_points[i]) / dx
            if df > L:
                L = df
        L = L * 1.1  # небольшой запас

        # характеристики интервалов и нижняя оценка
        R = []
        for i in range(len(x_points) - 1):
            xi, fi = x_points[i], f_points[i]
            xj, fj = x_points[i + 1], f_points[i + 1]
            R_i = 0.5 * (fi + fj - L * (xj - xi))
            R.append(R_i)

        R_min = min(R)
        idx = R.index(R_min)  # интервал с наименьшей характеристикой

        # текущий лучший верхний предел минимума
        f_best = min(f_points)

        # критерий остановки: разность верхней и нижней оценок
        if f_best - R_min <= eps:
            break

        # новая точка по формуле Пиявского
        xi, fi = x_points[idx], f_points[idx]
        xj, fj = x_points[idx + 1], f_points[idx + 1]
        x_new = 0.5 * (xi + xj) - (fj - fi) / (2 * L)

        f_new = f(x_new)
        iters += 1

        x_points.append(x_new)
        f_points.append(f_new)

    # финальное упорядочивание
    pts = sorted(zip(x_points, f_points), key=lambda p: p[0])
    x_points, f_points = [p[0] for p in pts], [p[1] for p in pts]
    f_best = min(f_points)
    x_best = x_points[f_points.index(f_best)]

    return x_best, f_best, list(zip(x_points, f_points)), iters


# ---------- визуализация ----------

def visualize(f, a: float, b: float, x_best: float, f_best: float, sample_points):
    """
    Рисует:
      - исходную функцию на [a, b];
      - ломаную по точкам выборки;
      - найденный минимум.
    """
    xs = np.linspace(a, b, 1000)
    ys = [f(x) for x in xs]

    plt.figure(figsize=(10, 6))
    plt.plot(xs, ys, label="f(x)")

    # точки выборки и ломаная
    sp_x = [p[0] for p in sample_points]
    sp_y = [p[1] for p in sample_points]
    plt.plot(sp_x, sp_y, "o-", label="точки алгоритма")

    # минимум
    plt.scatter([x_best], [f_best], s=80, zorder=5, label=f"минимум ≈ ({x_best:.3f}, {f_best:.3f})")

    plt.title("Глобальный поиск минимума (метод Пиявского)")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ---------- пример запуска: функция Растригина в 1D ----------

if __name__ == "__main__":
    # Пример функции: одномерная Растригина, много локальных минимумов
    # f(x) = x^2 - 10*cos(2*pi*x) + 10  на [-4, 4]
    expr = "x**2 - 10*math.cos(2*math.pi*x) + 10"

    a, b = -4.0, 4.0
    eps = 0.01

    f = make_function(expr)

    start = time.perf_counter()
    x_min, f_min, pts, iters = global_min_piyavskii(f, a, b, eps)
    elapsed = time.perf_counter() - start

    print("Функция: f(x) =", expr)
    print(f"Отрезок: [{a}, {b}]")
    print(f"Точность eps = {eps}")
    print()
    print(f"Приближённый минимум:")
    print(f"  x* ≈ {x_min:.6f}")
    print(f"  f(x*) ≈ {f_min:.6f}")
    print(f"Число итераций (вычислений функции): {iters}")
    print(f"Время работы: {elapsed:.6f} с")

    visualize(f, a, b, x_min, f_min, pts)
