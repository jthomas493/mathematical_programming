# Databricks notebook source
# MAGIC %pip install scipy

# COMMAND ----------

# %restart_python

# COMMAND ----------

from scipy.optimize import minimize, NonlinearConstraint
import numpy as np

# COMMAND ----------

def objective_function(x):
    return x[0]**2 - 4*x[0] - 2*x[1]

def constraint_function(x):
    return x[0] + x[1]

def constraint_function2(x):
    return 2*x[0] + x[1] 

def constraint_function3(x):
    return -x[0] + 4*x[1]

initial_guess = [0.5, 0.5]

bounds = ([0,np.inf],[0,np.inf])

nlc1 = NonlinearConstraint(constraint_function, 0, 4) 
nlc2 = NonlinearConstraint(constraint_function2, 0, 5) 
nlc3 = NonlinearConstraint(constraint_function3, 2, np.inf) 

result = minimize(objective_function, initial_guess, constraints=[nlc1,nlc2,nlc3], method='SLSQP', bounds=bounds)
print(result)

# COMMAND ----------

def objective_function(x):
    return 4*x[0]**3 + 4*x[0] - 6*x[1]

def constraint_function(x):
    return x[0] + x[1]

def constraint_function2(x):
    return 2*x[0] + x[1] 

def constraint_function3(x):
    return -x[0] + 4*x[1]

initial_guess = [0.5, 0.5]

bounds = ([0,np.inf],[0,np.inf])

nlc1 = NonlinearConstraint(constraint_function, 0, 4) 
nlc2 = NonlinearConstraint(constraint_function2, 0, 5) 
nlc3 = NonlinearConstraint(constraint_function3, 2, np.inf) 

result = minimize(objective_function, initial_guess, constraints=[nlc1,nlc2,nlc3], method='SLSQP', bounds=bounds)
print(result)