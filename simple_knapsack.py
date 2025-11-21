# Databricks notebook source
# MAGIC %pip install pulp

# COMMAND ----------

# %restart_python

# COMMAND ----------

import pandas as pd
from pulp import *

# COMMAND ----------

limit = 10

# COMMAND ----------

A = LpVariable("A",lowBound=0,cat="Integer")
B = LpVariable("B",lowBound=0,cat="Integer")
C = LpVariable("C",lowBound=0,cat="Integer")
D = LpVariable("D",lowBound=0,cat="Integer")

# COMMAND ----------

prob = LpProblem("Knapsack",LpMaximize)

# COMMAND ----------

#Objective Function
prob += 30*A + 20*B + 15*C + 25*D

# COMMAND ----------

#Subject to
prob += 6*A + 4*B + 3*C + 4*D <= limit, "Weight Limit"

# COMMAND ----------

prob.solve()

# COMMAND ----------

print(f"Optimal Objective Value: {value(prob.objective)}")
print("Variable Values:")
for v in prob.variables():
    print(f"{v.name} = {v.varValue}")

# COMMAND ----------

print("Constraint Slack Values:")
for name, constraint in prob.constraints.items():
    print(f"Constraint: {name}, Slack: {constraint.slack}")
print("Shadow Prices : The Duals")
for name, constraint in prob.constraints.items():   
    print(f"Constraint:{name}, Dual: {constraint.pi}")