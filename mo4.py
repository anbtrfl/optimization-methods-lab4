import random

import numpy as np
import matplotlib.pyplot as plt
import scipy
from prettytable import PrettyTable
import time

function_calls = 0


def function(x):
    global function_calls
    function_calls += 1
    return x ** 2 + 4 * np.sin(5 * x) + 0.1 * x ** 4


def func_derivative(x):
    return 2 * x + 20 * np.cos(5 * x) + 0.4 * x ** 3


def Rosenbrock(x):
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2


def generate_point(s, bounds):
    new_s = s[:]
    idx = random.randint(0, len(s) - 1)
    new_s[idx] = np.random.uniform(bounds[idx][0], bounds[idx][1])
    return new_s

def accept(old_val, new_val, temp):
    if new_val < old_val:
        return True
    else:
        return random.random() < np.exp((old_val - new_val) / temp)

def annealing(f, bounds, temp, cooling, min_temp, iterations=1000):
    s = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * np.random.rand(len(bounds))
    best_solution = s
    current_temp = temp
    best_val = f(s)
    it = 0
    history = [s]
    history_f = [best_val]
    while it < iterations and current_temp > min_temp:
        solution = generate_point(s, bounds)
        old_val = f(s)
        new_val = f(solution)
        if accept(old_val, new_val, current_temp):
            s = solution
            history.append(s)
            history_f.append(new_val)
            if new_val < best_val:
                best_val = new_val
                best_solution = solution
        current_temp *= cooling
        it += 1
    return best_solution, best_val, history, history_f


def gradient_descent(objective, derivative, bounds, n_iterations, step_size):
    solution = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * np.random.rand(len(bounds))
    solution_eval = objective(solution)
    solutions = [solution_eval]
    for i in range(n_iterations):
        gradient = derivative(solution)
        solution = solution - step_size * gradient
        solution_eval = objective(solution)
        solutions.append(solution_eval)
    return [solution, solution_eval, solutions]


def nelder_mead(function, start_values, epochs=1000, eps=1e-3, eps_gradient=1e-10):
    res = scipy.optimize.minimize(function, start_values, method='Nelder-Mead',
                                  options={'xatol': eps_gradient, 'disp': True, 'return_all': True, 'maxiter': epochs})
    vecs = res.get('allvecs')
    return vecs


def measure_time(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time


START_LEFT = -3
START_RIGHT = 3
bounds = np.array([[START_LEFT, START_RIGHT]])
n_iterations = 1000
step_size = 0.1
temp = -10

best, score, scores, iterations = annealing(function, bounds, n_iterations, step_size, temp)

x_values = np.linspace(START_LEFT, START_RIGHT, n_iterations)
y_values = function(x_values)


plt.figure(figsize=(10, 6))
plt.plot(x_values, y_values, 'g-', label='Функция')
plt.scatter([point[0] for point in iterations], [function(point[0]) for point in iterations], c='r', marker='o', s=50, label='Итерации')
plt.xlabel('x', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.title('График функции', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=12)
plt.xlim(START_LEFT-1, START_RIGHT+1)
plt.ylim(min(y_values)-1, max(y_values)+1)
plt.show()
# -----------------------------------------------------------------------------------------------------------------------
n_iterations = 1000
START_LEFT = -3
START_RIGHT = 3
bounds = np.array([[START_LEFT, START_RIGHT]])
step_size_gd = 0.01
step_size_sa = 0.1
temp = -10


st1= time.time()
best_gd, score_gd, scores_gd = gradient_descent(function, func_derivative, bounds, n_iterations,
                                                step_size_gd)
et1 = time.time()

st2 = time.time()
best_sa, score_sa, scores_sa, it = annealing(function, bounds, n_iterations, step_size_sa, temp)
et2 = time.time()

st3 = time.time()
start_values = np.array([2.5])
vecs = nelder_mead(function, start_values, epochs=100, eps=1e-10, eps_gradient=1e-3)
et3 = time.time()



function_calls = 0

gd_calls = function_calls
gd_iterations = len(scores_gd)
gd_time = et1-st1

function_calls = 0

sa_calls = function_calls
sa_iterations = len(scores_sa)
sa_time = et2-st2

function_calls = 0

nm_calls = function_calls
nm_iterations = len(vecs[0])
nm_time = et3-st3

# Таблица сравнения
table = PrettyTable()
table.field_names = ["Метод", "Найденное решение", "Время (с)", "Количество вызовов", "Количество итераций"]
table.add_row(["Градиентный спуск", best_gd[0], gd_time, gd_calls, gd_iterations])
table.add_row(["Имитация отжига", best_sa[0], sa_time, sa_calls, sa_iterations])
table.add_row(["Нелдера-Мида", vecs[0][0], nm_time, nm_calls, nm_iterations])

# print(table)


# График сравнения скорости сходимости
# plt.figure(figsize=(10, 6))
# plt.plot(scores_gd, 'r-', label='Градиентный спуск')
# plt.plot(scores_sa, 'g-', label='Имитация отжига')
# plt.plot(vecs, 'b-', label='Нелдера-Мида')
# plt.xlabel('Итерация')
# plt.ylabel('')
# plt.title('Сравнение скорости сходимости градиентного спуска и имитации отжига')
# plt.legend()
# plt.grid(True)
# plt.show()
