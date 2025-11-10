import math
import time
import numpy as np
import matplotlib.pyplot as plt


# ---------- 1. Создание функции из строки ----------

def make_function(expr):
    """
    Превращает строку expr в функцию f(x).
    Разрешены sin, cos, exp, log, sqrt, pi, abs и переменная x.
    Пример:
        f = make_function("x**2 - 10*cos(2*pi*x) + 10")
        print(f(0))  # 0
    """
    # compile переводит строку expr в объект скомпилированного кода
    code = compile(expr, "<string>", "eval")

    safe_math = {
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "exp": math.exp,
        "log": math.log,
        "sqrt": math.sqrt,
        "pi": math.pi,
        "abs": abs,
    }

    # Внутри make_function мы объявляем вложенную функцию f(x)
    # eval(code, ...) - вычисляет скомпилированное выражение.
    # {"__builtins__": None} - отключаем доступ к встроенным функциям Python (безопасность).
    # Третий аргумент — словарь «локальных» переменных:
    # **safe_math — разворачиваем наш словарь с sin, cos, pi, ...
    # "x": x — добавляем переменную x, равную значению аргумента.
    def f(x):
        return eval(code, {"__builtins__": None}, {**safe_math, "x": x})

    return f


# ---------- 2. Метод Пиявского–Шуберта ----------

def piyavskii_shubert(f, a, b, L, eps=0.01, max_iter=1000):
    """
    Метод Пиявского–Шуберта для глобального поиска минимума
    липшицевой функции f на отрезке [a, b] с константой Липшица L.
    """
    start_time = time.perf_counter()

    # список уже исследованных точек по x, сначала только a и b
    X = [a, b]
    F = [f(a), f(b)]
    iterations = 2

    while iterations < max_iter:
        f_best = min(F)
        # минимальная нижняя оценка среди интервалов
        m_best = float("inf")
        x_new = None

        for i in range(len(X) - 1):
            xi, xj = X[i], X[i + 1]
            fi, fj = F[i], F[i + 1]

            # точка пересечения конусов Липшица
            x_mid = (fi - fj) / (2 * L) + (xi + xj) / 2
            # нижняя оценка минимально возможного значения функции на интервале
            m_i = (fi + fj) / 2 - L * (xj - xi) / 2

            if m_i < m_best:
                m_best = m_i
                x_new = x_mid

        # Если зазор ≤ eps, значит, текущий минимум достаточно близок к глобальному, выходим из цикла
        gap = f_best - m_best
        if gap <= eps:
            break

        f_new = f(x_new)
        iterations += 1

        # находит позицию, куда нужно вставить x_new, чтобы список X оставался отсортированным
        insert_pos = np.searchsorted(X, x_new)
        X.insert(insert_pos, x_new)
        F.insert(insert_pos, f_new)

    elapsed = time.perf_counter() - start_time
    # индекс лучшей точки = индекс минимального значения в списке F
    best_idx = int(np.argmin(F))
    f_best = F[best_idx]

    return X, F, best_idx, iterations, elapsed, m_best, f_best


# ---------- 3. Визуализация ----------

def plot_results(f, a, b, X, F, best_idx, L, expr_str="", eps=0.01, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Переводим списки в массивы numpy
    X = np.array(X)
    F = np.array(F)

    # xs — равномерная сетка из 1000 точек на [a, b]
    xs = np.linspace(a, b, 1000)
    # ys — значения функции в этих точках
    ys = np.array([f(float(x)) for x in xs])

    # Строим нижнюю оценку функции (ломаную) в каждой точке xg
    lb = []
    for xg in xs:
        lb_val = max(F[i] - L * abs(xg - X[i]) for i in range(len(X)))
        lb.append(lb_val)
    lb = np.array(lb)

    ax.plot(xs, ys, label="f(x)", linewidth=2)
    ax.plot(xs, lb, linestyle="--", label="Нижняя оценка", linewidth=1)
    ax.scatter(X, F, s=30, label="Точки метода")

    x_star, f_star = X[best_idx], F[best_idx]
    ax.scatter([x_star], [f_star], color="red", s=60, label="Найденный минимум")
    ax.axvline(x_star, color="red", linestyle=":", linewidth=1)
    ax.axhline(f_star, color="red", linestyle=":", linewidth=1)

    title = "Метод Пиявского–Шуберта"
    if expr_str:
        title += f"\n f(x) = {expr_str}"
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.grid(True)
    ax.legend()


# ---------- 4. Вспомогательная функция: единый запуск ----------

def run_experiment(expr, a, b, L, eps):
    """
    Общая функция: принимает строку expr и параметры,
    запускает метод, печатает результаты и рисует график.
    """
    f = make_function(expr)

    # если пользователь случайно ввёл a > b
    if a > b:
        a, b = b, a
        print(f"[!] Поменяла концы местами: теперь отрезок [{a}, {b}]")

    X, F, best_idx, iterations, elapsed, m_best, f_best = piyavskii_shubert(
        f, a, b, L, eps=eps, max_iter=1000
    )
    # Координаты найденного минимума
    x_star = X[best_idx]
    f_star = F[best_idx]

    print("\n=== Результаты ===")
    print(f"Функция: f(x) = {expr}")
    print(f"Отрезок: [{a}, {b}]")
    print(f"L = {L}, eps = {eps}")
    print(f"x* ≈ {x_star:.6f}, f(x*) ≈ {f_star:.6f}")
    print(f"Итераций (вычислений f): {iterations}")
    print(f"Нижняя оценка m_best ≈ {m_best:.6f}")
    print(f"Зазор f_best - m_best ≈ {f_best - m_best:.6f}")
    print(f"Время: {elapsed:.6f} с")

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_results(f, a, b, X, F, best_idx, L, expr_str=expr, eps=eps, ax=ax)
    plt.show()


# ---------- 5. Режимы ----------

def mode_rastrigin():
    """
    Режим: функция Растригина.
    Пользователь может либо взять параметры по умолчанию,
    либо ввести свои a, b, L, eps.
    """
    expr = "x**2 - 10*cos(2*pi*x) + 10"
    print("=== Режим: функция Растригина (1D) ===")
    print(f"По умолчанию используется f(x) = {expr}")
    use_default = input("Использовать параметры по умолчанию? (y/n): ").strip().lower()

    if use_default == "y" or use_default == "д":
        a, b = -5.12, 5.12
        L = 80.0
        eps = 0.01
    else:
        a = float(input("Левый конец отрезка a: "))
        b = float(input("Правый конец отрезка b: "))
        L = float(input("Оценка константы Липшица L (> 0): "))
        eps = float(input("Точность eps (например, 0.01): "))

    run_experiment(expr, a, b, L, eps)


def mode_ackley():
    """
    Режим: одномерное сечение функции Экли.
    Пользователь может либо взять параметры по умолчанию,
    либо ввести свои.
    """
    expr = (
        "-20*exp(-0.2*sqrt(0.5*x**2)) "
        "- exp(0.5*(cos(2*pi*x) + 1)) "
        "+ 20 + exp(1)"
    )
    print("=== Режим: функция Экли (Ackley), 1D-сечение ===")
    print("Берём стандартную 2D Ackley, фиксируем y = 0.")
    print(f"По умолчанию используется f(x) = {expr}")
    use_default = input("Использовать параметры по умолчанию? (y/n): ").strip().lower()

    if use_default == "y" or use_default == "д":
        a, b = -5.0, 5.0
        L = 20.0
        eps = 0.01
    else:
        a = float(input("Левый конец отрезка a: "))
        b = float(input("Правый конец отрезка b: "))
        L = float(input("Оценка константы Липшица L (> 0): "))
        eps = float(input("Точность eps (например, 0.01): "))

    run_experiment(expr, a, b, L, eps)


def mode_custom():
    """
    Полностью пользовательский режим:
    пользователь вводит f(x), [a, b], L, eps.
    """
    print("=== Пользовательский режим (произвольная функция) ===")
    print("Пример функции: x**2 - 10*cos(2*pi*x) + 10")
    print("Можно использовать: sin, cos, exp, log, sqrt, pi, abs и переменную x.\n")

    expr = input("Введите выражение для f(x): ").strip()
    a = float(input("Левый конец отрезка a: "))
    b = float(input("Правый конец отрезка b: "))
    L = float(input("Оценка константы Липшица L (> 0): "))
    eps = float(input("Точность eps (например, 0.01): "))

    run_experiment(expr, a, b, L, eps)


# ---------- 6. Главное меню ----------

if __name__ == "__main__":
    print("Выберите режим работы программы:")
    print("1 — Тест на функции Растригина")
    print("2 — Тест на функции Экли")
    print("3 — Своя (произвольная) функция")

    mode = input("Ваш выбор (1/2/3): ").strip()

    if mode == "1":
        mode_rastrigin()
    elif mode == "2":
        mode_ackley()
    elif mode == "3":
        mode_custom()
    else:
        print("Неизвестный выбор, запускаю пользовательский режим по умолчанию.\n")
        mode_custom()
