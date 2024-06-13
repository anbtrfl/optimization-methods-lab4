import math
import tracemalloc
from time import time

import numpy as np
import optuna
import sympy
from matplotlib import pyplot as plt
from numpy import float64
from scipy import optimize
from sympy import *


class Statistic:
    def __init__(self):
        self.__start_time = None
        self.iterations = 0
        self.spent_time = 0.0
        self.memory = 0
        self.function_calculations = 0
        self.gradient_calculations = 0
        self.hessian_calculations = 0
        self.is_tracing_running = False

    def start_clock(self):
        self.__start_time = time()

    def stop_clock(self):
        self.spent_time += time() - self.__start_time
        self.__start_time = None

    def start_trace(self):
        tracemalloc.start()
        self.is_tracing_running = True

    def stop_trace(self):
        if self.is_tracing_running:
            res = tracemalloc.take_snapshot()
            stats = res.statistics(cumulative=True, key_type='filename')
            for stat in stats:
                self.memory += stat.size
            tracemalloc.stop()
            self.is_tracing_running = False

    def print_stat(self):
        print('time:           ', self.spent_time)
        print('memory:         ', self.memory)
        print('function_calls: ', self.function_calculations)
        print('gradient_calls: ', self.gradient_calculations)
        print('hessian_calls:  ', self.hessian_calculations)
        print('iterations:     ', self.iterations)


def difficult_func(func, vars, name):
    dfdx = lambdify(vars, diff(func, vars[0]))
    dfdy = lambdify(vars, diff(func, vars[1]))

    hess = hessian(func, vars)
    print(hess)
    hess = lambdify(vars, hess, modules='sympy')
    func = lambdify(vars, func, 'numpy')

    def f(args):
        return func(args[0], args[1])

    def grad(x):
        return np.array([dfdx(x[0], x[1]), dfdy(x[0], x[1])], dtype=float64)

    def f_hessian(x):
        res = hess(x[0], x[1])
        result = np.array([res[:2], res[2:]], dtype=float64)
        return result

    print(f_hessian(np.array([0, 0])))
    return [f, grad, f_hessian, name]


def not_working():
    x, y = symbols('x, y')
    func = 0.1 * (x ** 4 + y ** 4) + x * y ** 2 + 2 * x + 2 * y
    return difficult_func(func, np.array([x, y]), "func")


def distribution_function():
    x, y = symbols('x, y')
    func = 5 * (2 * x ** 2 - 1) * y * sympy.exp(-x ** 2 - y ** 2)
    return difficult_func(func, np.array([x, y]),
                          "5 * (2x^2 - 1)y * exp(-x^2-y^2)")


def rosenbrock(x):
    return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2


def grad_rosenbrock(arg):
    x = arg[0]
    y = arg[1]
    dx = 2 * (200 * (x ** 3) - 200 * x * y + x - 1)
    dy = 200 * (y - (x ** 2))
    return np.array([dx, dy])


def hessian_rosenbrock(arg):
    x = arg[0]
    y = arg[1]
    return np.array([
        np.array([-400 * (y - x ** 2) + 800 * x ** 2 + 2, -400 * x]),
        np.array([-400 * x, 200])
    ])


def ackley():
    x, y = symbols('x, y')
    func = -20.0 * exp(-0.2 * sqrt(0.5 * (x ** 2 + y ** 2))) - exp(
        0.5 * (cos(2 * pi * x) + cos(2 * pi * x))) + math.e + 20
    return difficult_func(func, np.array([x, y]),
                          "ackley")


def notpolinom():
    x, y = symbols('x, y')
    func = 5 * x * y * exp(-x ** 2 - y ** 2)
    return difficult_func(func, np.array([x, y]),
                          "5 * x * y * exp(-x ** 2 - y ** 2)")


def reversed_f():
    x, y = symbols('x, y')
    func = -1 * (2 * exp(-((x - 1) / 2) ** 2 - ((y - 1) / 1) ** 2) + 3 * exp(
        -((x - 2) / 3) ** 2 - ((y - 3) / 2) ** 2))
    return difficult_func(func, np.array([x, y]), "notpolinom2")


def golden_ratio(function, left_border, right_border, x, p, stat, eps=1e-8):
    phi = (1 + np.sqrt(5)) / 2
    a = left_border
    b = right_border
    iterations = 0
    calculations = 0
    segments = [(a, b)]

    c_1 = b - (b - a) / phi
    c_2 = a + (b - a) / phi

    stat.function_calculations += 2
    calc_result = [function(x + p * c_1), function(x + p * c_2)]
    calculations += 2

    while (b - a) / 2 >= eps:
        iterations += 1
        calculations += 1
        stat.function_calculations += 1
        if calc_result[0] > calc_result[1]:
            a = c_1
            c_1 = c_2
            c_2 = b - (c_1 - a)
            calc_result[0] = calc_result[1]
            calc_result[1] = function(x + p * c_2)
        else:
            b = c_2
            c_2 = c_1
            c_1 = a + b - c_2
            calc_result[1] = calc_result[0]
            calc_result[0] = function(x + p * c_1)
        segments.append((a, b))

    c = (b + a) / 2

    return c


def dichotomies(f, a, b, stat, eps=1e-8):
    while abs(b - a) > eps:
        c = (a + b) / 2
        delta = (b - a) / 8
        f1 = f(c - delta)
        f2 = f(c + delta)
        stat.function_calculations += 2
        if f1 < f2:
            b = c
        else:
            a = c
    return (a + b) / 2


def gradient_descend(f, f_grad, hessian, start_point, eps=1e-8):
    stat = Statistic()
    stat.start_trace()
    stat.start_clock()
    history = [start_point]
    x = start_point
    for epoch in range(1, 1501):
        grad_value = f_grad(x)
        lr = dichotomies(lambda t: f(x - t * grad_value), 0, 2, stat)
        x = x - lr * grad_value
        history.append(x)
        if epoch > 0 and np.linalg.norm(f_grad(x)) < eps:
            break
        stat.gradient_calculations += 1
        stat.iterations += 1
    stat.stop_trace()
    stat.stop_clock()
    return history[-1], stat, history


def positive_matrix(m):
    eigenvalues = np.linalg.eigvals(m)
    if all(eigenvalues > 0):
        return m
    else:
        return m + 2 * (-min(eigenvalues)) * np.eye(m.shape[0])


def newton(f, f_grad, f_hessian, start_point, eps=1e-8):
    cur_x = start_point
    stat = Statistic()
    stat.start_trace()
    stat.start_clock()
    xs = [start_point]
    cur_grad = 1

    while stat.iterations < 100 and np.linalg.norm(cur_grad) > eps:
        xs.append(cur_x)
        stat.iterations += 1
        cur_hessian = f_hessian(cur_x)
        stat.hessian_calculations += 1
        pos_hessian = positive_matrix(cur_hessian)
        cur_grad = f_grad(cur_x)
        if np.linalg.det(pos_hessian) != 0:
            direction = np.linalg.inv(pos_hessian) @ cur_grad
        else:
            direction = cur_grad
        alpha = golden_ratio(f, 0, 5, cur_x, -direction, stat)
        cur_x = cur_x - direction * alpha
        stat.gradient_calculations += 1
    stat.stop_trace()
    stat.stop_clock()
    xs.append(cur_x)
    return cur_x, stat, xs


def newton_with_constant_step(f, f_grad, f_hessian, start_point, step, eps=1e-8):
    cur_x = start_point
    stat = Statistic()
    stat.start_trace()
    stat.start_clock()
    xs = [start_point]
    cur_grad = 1

    while stat.iterations < 100 and np.linalg.norm(cur_grad) > eps:
        xs.append(cur_x)
        stat.iterations += 1
        cur_hessian = f_hessian(cur_x)
        stat.hessian_calculations += 1
        pos_hessian = positive_matrix(cur_hessian)
        cur_grad = f_grad(cur_x)
        if np.linalg.det(pos_hessian) != 0 and stat.iterations > 3:
            direction = np.linalg.inv(pos_hessian) @ cur_grad
        else:
            direction = cur_grad
        cur_x = cur_x - direction * step
        stat.gradient_calculations += 1
    stat.stop_trace()
    stat.stop_clock()
    xs.append(cur_x)
    return cur_x, stat, xs


def wolf_condition(c1, c2, grad, f, xk, ak, direction, stat):
    grad_xk = grad(xk)
    new_xk = xk + ak * direction
    armiho_condition = f(new_xk) <= f(xk) + c1 * ak * np.dot(grad_xk, direction)
    curvature_condition = abs(np.dot(grad(new_xk), direction)) <= c2 * abs(np.dot(grad_xk, direction))
    stat.function_calculations += 2
    stat.gradient_calculations += 2
    return (armiho_condition and curvature_condition) or ak < 1e-10


def backtracking_line_search(f, grad, a1, a2, xk, beta, direction, stat):
    ak = 1
    while not wolf_condition(a1, a2, grad, f, xk, ak, direction, stat):
        ak *= beta
    return ak


def newton_with_wolf(f, f_grad, f_hessian, start_point, c1, c2, beta, eps=1e-8):
    cur_x = start_point
    stat = Statistic()
    stat.start_trace()
    stat.start_clock()
    xs = []
    cur_grad = f_grad(cur_x)
    while stat.iterations < 100 and np.linalg.norm(cur_grad) > eps:
        xs.append(cur_x)
        stat.iterations += 1
        cur_hessian = f_hessian(cur_x)
        stat.hessian_calculations += 1
        pos_hessian = positive_matrix(cur_hessian)
        if np.linalg.det(pos_hessian) != 0 and stat.iterations > 3:
            direction = np.linalg.inv(pos_hessian) @ cur_grad
        else:
            direction = cur_grad
        alpha = backtracking_line_search(f, f_grad, c1, c2, cur_x, beta, -direction, stat)
        cur_x = cur_x - direction * alpha
        cur_grad = f_grad(cur_x)
        stat.gradient_calculations += 1
    stat.stop_trace()
    stat.stop_clock()
    xs.append(cur_x)
    return cur_x, stat, xs


def generic_quazi_optimize_function(f, f_grad, start_point, eps, method, f_hessian=None):
    stat = Statistic()
    stat.start_trace()
    stat.start_clock()
    xs = [start_point]
    result = optimize.minimize(fun=f, x0=start_point, method=method, jac=f_grad, hess=f_hessian, tol=eps,
                               callback=lambda el: xs.append(el),
                               options={'maxiter': 100000})
    stat.function_calculations = result.nfev
    stat.gradient_calculations = result.njev
    stat.iterations = result.nit
    stat.stop_trace()
    stat.stop_clock()
    return result.x, stat, xs


def newton_cg(f, f_grad, f_hessian, start_point, eps=1e-8):
    return generic_quazi_optimize_function(f, f_grad, start_point, eps, 'Newton-CG')


def quasinewton_BFGS(f, f_grad, f_hessian, start_point, eps=1e-8):
    return generic_quazi_optimize_function(f, f_grad, start_point, eps, 'BFGS')


def quasinewton_L_BFGS(f, f_grad, f_hessian, start_point, eps=1e-8):
    return generic_quazi_optimize_function(f, f_grad, start_point, eps, 'L-BFGS-B')


all_functions = [

    [rosenbrock, grad_rosenbrock, hessian_rosenbrock, "Rosenbrock"],

    distribution_function(),

    not_working(),

]

methods = [
    [gradient_descend, "Gradient descend"],
    # [newton, "Newton"],
    # [newton_with_constant_step, "Newton with constant step"],
    # [newton_with_wolf, "Newton with wolf"],
    # [newton_cg, "Newton-CG"],
    # [quasinewton_BFGS, "Quasinewton (BFGS)"],
    # [quasinewton_L_BFGS, "Quasinewton (L-BFGS-B)"]
]

results_for_graphs = []
stats_by_method = {
    gradient_descend: [],
    newton: [],
    newton_with_constant_step: [],
    # метод ньютона с условием вольфа
    newton_with_wolf: [],
    newton_cg: [],
    quasinewton_BFGS: [],
    quasinewton_L_BFGS: []
}


def print_res(x, f, stat):
    print('x: ', x)
    print('y: ', f(x))
    stat.print_stat()


def draw(diap, eps, function, start_values, title, xk, yk, fig, ax, ax2):
    x = np.arange(-diap[0], diap[0], diap[0] / 100)
    y = np.arange(-diap[1], diap[1], diap[1] / 100)
    X, Y = np.meshgrid(x, y)
    Z = function([X, Y])
    ax.plot(xk, yk, function(np.array([xk, yk])), color='blue')
    ax.scatter(xk[-1], yk[-1], color='red')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, color='cyan', edgecolor='none', alpha=0.3)

    # линии уровня
    cp = ax2.contour(X, Y, Z, levels=sorted(set(function(np.array([xk[i], yk[i]])) for i in range(len(xk)))))
    ax2.clabel(cp, inline=1, fontsize=10)
    ax2.plot(xk, yk, color='blue')
    ax2.scatter(xk[-1], yk[-1], color='red')
    fig.suptitle(f"{title}, epsilon = {eps}, начальная точка: {start_values}")


def draw_result(result, function, title, eps, point):
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2)
    xs = result[2]
    xk = [i[0] for i in xs]
    yk = [i[1] for i in xs]
    diapason = np.array([max(map(abs, xk)) + 3, max(map(abs, yk)) + 3])
    draw(diapason, eps, function, point, title, xk, yk, fig, ax, ax2)
    plt.show()


def print_result(method_name, method, point, stat, x, xs, best_params):
    print("=================================================================================")
    print("Function name: rosenbrock_function")
    print()
    print("Method: " + method_name)
    for param in best_params.keys():
        print(param + ":", best_params[param])
    print_res(x, rosenbrock, stat)
    print()
    results_for_graphs.append((point, method_name, "Rosenbrock", xs, rosenbrock))
    stats_by_method[method].append(stat)
    draw_result((x, stat, xs), rosenbrock, method_name + " для функции Rosenbrock", 1e-8, point)
    print()
    print()


def objective_wolf(trial):
    f = rosenbrock
    f_grad = grad_rosenbrock
    f_hessian = hessian_rosenbrock
    start_point_x = trial.suggest_float('start_point_x', -10.0, 10.0)
    start_point_y = trial.suggest_float('start_point_y', -10.0, 10.0)
    start_point = np.array([start_point_x, start_point_y])
    c1 = trial.suggest_float('c1', 0.0, 1.0)
    c2 = trial.suggest_float('c2', c1, 1.0)
    beta = trial.suggest_float('beta', 0.5, 1.0)
    eps = 1e-8
    cur_x, stat, xs = newton_with_wolf(f, f_grad, f_hessian, start_point, c1, c2, beta, eps)
    d = np.linalg.norm(np.array([1, 1]) - cur_x)
    trial.set_user_attr('stat', stat)
    trial.set_user_attr('x', cur_x)
    trial.set_user_attr('xs', xs)
    return d


def objective(trial):
    f = rosenbrock
    f_grad = grad_rosenbrock
    f_hessian = hessian_rosenbrock
    start_point_x = trial.suggest_float('start_point_x', -10.0, 10.0)
    start_point_y = trial.suggest_float('start_point_y', -10.0, 10.0)
    start_point = np.array([start_point_x, start_point_y])
    eps = 1e-8
    cur_x, stat, xs = method[0](f, f_grad, f_hessian, start_point, eps)
    d = np.linalg.norm(np.array([1, 1]) - cur_x)
    trial.set_user_attr('stat', stat)
    trial.set_user_attr('x', cur_x)
    trial.set_user_attr('xs', xs)
    return d


def objective_newton_constant_step(trial):
    f = rosenbrock
    f_grad = grad_rosenbrock
    f_hessian = hessian_rosenbrock
    start_point_x = trial.suggest_float('start_point_x', -10.0, 10.0)
    start_point_y = trial.suggest_float('start_point_y', -10.0, 10.0)
    start_point = np.array([start_point_x, start_point_y])
    step = trial.suggest_float('step', 1e-8, 1)
    eps = 1e-8
    cur_x, stat, xs = newton_with_constant_step(f, f_grad, f_hessian, start_point, step, eps)
    d = np.linalg.norm(np.array([1, 1]) - cur_x)
    trial.set_user_attr('stat', stat)
    trial.set_user_attr('x', cur_x)
    trial.set_user_attr('xs', xs)
    return d


def optimize_with_optuna(method_name, method, objective):
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=15)
    best_params = study.best_params
    point = np.array([best_params['start_point_x'], best_params['start_point_y']])
    stat = study.best_trial.user_attrs['stat']
    x = study.best_trial.user_attrs['x']
    xs = study.best_trial.user_attrs['xs']
    print_result(method_name, method, point, stat, x, xs, best_params)


for method in methods:
    print(method[1])
    if method[0] == newton_with_wolf:
        obj = objective_wolf
    elif method[0] == newton_with_constant_step:
        obj = objective_newton_constant_step
    else:
        obj = objective
    optimize_with_optuna(method[1], method[0], obj)
