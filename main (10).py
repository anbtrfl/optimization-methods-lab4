import matplotlib.pyplot as plt
import numpy as np
import optuna
import tensorflow as tf
from keras import Sequential
from keras.src.layers import Dense
from keras.src.losses import MeanSquaredError
from keras.src.optimizers import SGD

np.random.seed(101)


def generate_data(n_samples, n_features, noise):
    X = np.random.randn(n_samples, n_features)
    true_coef = np.random.randn(n_features)
    y = 0.5 * (X ** 2).dot(true_coef) + X.dot(true_coef) + noise * np.random.randn(n_samples)
    if n_features == 1:
        X = np.squeeze(X)
    return X, y, true_coef


def generate_data_5(n_samples, n_features, noise):
    X = np.random.randn(n_samples, n_features)
    true_coef = np.random.randn(n_features)
    y = 0.1 * ((X ** 5).dot(true_coef) - (X ** 4).dot(true_coef) - 10 * X.dot(true_coef)) - noise * np.random.randn(
        n_samples)
    if n_features == 1:
        X = np.squeeze(X)
    return X, y, true_coef


def generate_data_2():
    true_slope = 2
    true_intercept = 1
    X = np.random.rand(100, 1)
    noise = np.random.randn(100, 1) * 0.1
    y = true_slope * X + true_intercept + noise
    return X, y, true_slope, true_intercept


def compute_gradient(x, y, theta):
    prediction = x.dot(theta)
    error = prediction - y
    gradient = 2 * x.T.dot(error) / len(y)
    return gradient


def polynomial_regression(x, y, theta, learning_rate, iterations, regularization, coeff, batch_size, power=5):
    m = len(y)
    X = np.array([x ** i for i in range(power)]).T
    cost_history = np.zeros(iterations)
    theta_history = np.zeros((iterations, X.shape[1]))
    operation_count = 0
    for it in range(iterations):
        random_index = np.random.randint(m)
        xi = X[random_index:random_index + batch_size]
        yi = y[random_index:random_index + batch_size]

        gradient = compute_gradient(xi, yi, theta)

        gradient += coeff * regularization['gradient'](theta)

        operation_count += batch_size

        theta = theta - learning_rate * gradient

        cost = (1.0 / (2 * m)) * np.sum((X.dot(theta) - y) ** 2) + coeff * regularization['reg'](theta)
        cost_history[it] = cost
        theta_history[it, :] = theta.T

    return theta, cost_history, theta_history


def loss_function(prediction, actual):
    tf.reduce_mean(tf.squared_difference(prediction, actual))


def train_model(optimizer, X, y, epochs=100):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(1,)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    history = model.fit(X, y, epochs=epochs, verbose=0)
    prediction = model.predict(X)
    return model, history, prediction


def sgd_with_nesterov(params):
    return tf.keras.optimizers.SGD(learning_rate=params['learning_rate'],
                                   nesterov=params['nesterov'])


def sgd_with_momentum(params):
    return tf.keras.optimizers.SGD(learning_rate=params['learning_rate'],
                                   momentum=params['momentum'])


def sgd(params):
    return tf.keras.optimizers.SGD(learning_rate=params['learning_rate'])


def adagrad(params):
    return tf.keras.optimizers.Adagrad(learning_rate=params['learning_rate'])


def rmsprop(params):
    return tf.keras.optimizers.RMSprop(learning_rate=params['learning_rate'])


def adam(params):
    return tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])


colors = {
    'SGD': 'blue',
    'SGD with Nesterov': 'green',
    'SGD with Momentum': 'red',
    'AdaGrad': 'cyan',
    'RMSprop': 'magenta',
    'Adam': 'yellow',
}

optimizers = {
    'SGD': sgd,
    'SGD with Nesterov': sgd_with_nesterov,
    'SGD with Momentum': sgd_with_momentum,
    'AdaGrad': adagrad,
    'RMSprop': rmsprop,
    'Adam': adam,
}

hyperparameters = {
    'SGD': {'learning_rate': 0.01, 'decay': 1e-6},
    'SGD with Nesterov': {'learning_rate': 0.01, 'nesterov': True},
    'SGD with Momentum': {'learning_rate': 0.01, "momentum": 0.9},
    'AdaGrad': {'learning_rate': 0.5},
    'RMSprop': {'learning_rate': 0.01},
    'Adam': {'learning_rate': 0.01},
}

best_l1_ratio = 0.9


def draw(X, y, theta, m_power, name, lam, threshold=10):
    X_new = np.array([X ** i for i in range(m_power)]).T
    y_pred = X_new.dot(theta)
    mask = (y_pred < threshold) & (y_pred > -threshold)

    y_pred_filtered = y_pred[mask]
    X_filtered = X[mask]

    plt.scatter(X_filtered, y_pred_filtered, color='red',
                label=f'Polynom regression degree = {m_power - 1}, with {name} regularization, coeff = {lam}')
    plt.scatter(X, y)
    plt.legend()
    plt.show()


m_power = 5

regularization = {
    'L1': {'gradient': lambda x: np.sign(x), 'reg': lambda x: np.sum(np.abs(x)),
           'lambda': [10, 30, 35, 50, 70, 100, 130], 'iter': 450},
    'L2': {'gradient': lambda x: 2 * x, 'reg': lambda x: np.dot(x, x),
           'lambda': [10, 20, 40], 'iter': 10000},
    'Elastic': {
        'gradient': lambda x: (1 - best_l1_ratio) * np.sign(x) + best_l1_ratio * 2 * x,
        'reg': lambda x: (1 - best_l1_ratio) * np.sum(np.abs(x)) + best_l1_ratio * np.dot(x, x),
        'lambda': [25, 30, 40, 45, 51, 70],
        'iter': 2000
    },
    'None': {
        'gradient': lambda _: 0,
        'reg': lambda _: 0,
    }
}


def objective(trial):
    global m_power
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    regularization_coeff = trial.suggest_float('regularization_coeff', 0.0, 200.0)
    batch_size = trial.suggest_int('batch_size', 1, 100)
    m_power = trial.suggest_int('power', 1, 10)
    reg_name = trial.suggest_categorical('regularization', ['L1', 'L2', 'Elastic', 'None'])
    reg = regularization[reg_name]

    start = np.array([2.0] * m_power)
    theta, cost_history, _ = polynomial_regression(X, y, start, learning_rate, 1000,
                                                   reg, regularization_coeff, batch_size, m_power)

    return cost_history[-1]


X, y, slope, intercept = generate_data_2()


def create_model(learning_rate, momentum):
    model = Sequential()
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    optimizer = SGD(learning_rate=learning_rate, momentum=momentum, nesterov=True)
    model.compile(optimizer=optimizer, loss=MeanSquaredError())
    return model


def objective2(trial):
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    momentum = trial.suggest_float('momentum', 0.0, 1.0)

    model = create_model(learning_rate, momentum)
    history = model.fit(X, y, validation_data=(X, y), epochs=10, verbose=0)

    return history.history['val_loss'][-1]


study = optuna.create_study(direction='minimize')

study.optimize(objective2, n_trials=50)

best_params = study.best_params
print(f'Best parameters: {best_params}')
print(f'Best values: {study.best_value}')

model = create_model(best_params['learning_rate'], best_params['momentum'])
history = model.fit(X, y, epochs=100, verbose=0)

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'])
plt.title('Model loss during training')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper right')
plt.grid(True)
plt.show()
