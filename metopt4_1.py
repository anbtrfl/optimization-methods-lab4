import optuna
import matplotlib.pyplot as plt


# Define the Rosenbrock function
def rosenbrock(trial):
    a = 1.0
    b = 100.0
    x = trial.suggest_uniform('x', -2.0, 2.0)
    y = trial.suggest_uniform('y', -1.0, 3.0)
    return (a - x) ** 2 + b * (y - x **2) **2


# Create a study object
study = optuna.create_study()

# Specify the number of trials
study.optimize(rosenbrock, n_trials=1000)

# Get the best parameters
print(study.best_params)
print(study.best_value)

# Get the values at each trial
values = [trial.value for trial in study.trials]

# Plot the values
plt.figure(figsize=(10, 6))
plt.plot(values)
plt.xlabel('Iteration')
plt.ylabel('Value')
plt.title('Change in Value with Iterations')
plt.grid(True)
plt.show()