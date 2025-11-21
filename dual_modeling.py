# Databricks notebook source
# MAGIC %pip install pulp

# COMMAND ----------

# %restart_python

# COMMAND ----------

import pandas as pd
from pulp import *

# COMMAND ----------

x1 = LpVariable("x1", lowBound=0, cat=LpContinuous)
x2 = LpVariable("x2", lowBound=0, cat=LpContinuous)
x3 = LpVariable("x3", lowBound=0, cat=LpContinuous)
x4 = LpVariable("x4", lowBound=0, cat=LpContinuous)
x5 = LpVariable("x5", lowBound=0, cat=LpContinuous)

# COMMAND ----------

prob = LpProblem("quickModel",LpMaximize)

# COMMAND ----------

#Objective function
prob += 550*x1 + 600*x2 + 350*x3 + 400*x4 + 200*x5

# COMMAND ----------

#subject to
#Grinding
prob += lpSum(12*x1 + 20*x2 + 25*x4 + 15*x5) <= 288, "Grinding"
#Drilling
prob+= lpSum(10*x1 + 8*x2 + 16*x3)<= 192, "Drilling"
#Manpower
prob+= lpSum(20*x1 + 20*x2 + 20*x3 + 20*x4 + 20*x5)<= 384, "Manpower"

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

# COMMAND ----------

prob2 = LpProblem("reverse_q_Model",LpMinimize)

# COMMAND ----------

y1 = LpVariable("y1", lowBound=0,cat="Continuous")
y2 = LpVariable("y2", lowBound=0,cat="Continuous")
y3 = LpVariable("y3", lowBound=0,cat="Continuous")

# COMMAND ----------

#Objective function
prob2 += 288*y1 + 192*y2 + 384*y3

# COMMAND ----------

#Subject to
prob2+= lpSum(12*y1 + 10*y2 + 20*y3) >= 550, "Product 1 Profit"
prob2+= lpSum(20*y1 + 8*y2 + 20*y3) >= 600, "Product 2 Profit"
prob2+= lpSum(16*y2 + 20*y3) >= 350, "Product 3 Profit"
prob2+= lpSum(25*y1 + 20*y3) >= 400, "Product 4 Profit"
prob2+= lpSum(15*y1 + 20*y3) >= 200, "Product 5 Profit"

# COMMAND ----------

prob2.solve()

# COMMAND ----------

print(f"Optimal Objective Value: {value(prob2.objective)}")
print("Variable Values:")
for v in prob2.variables():
    print(f"{v.name} = {v.varValue}")

# COMMAND ----------

#Slack in the dual model indicates the cost of producing product 3,4,5

# COMMAND ----------

print("Constraint Slack Values:")
for name, constraint in prob2.constraints.items():
    print(f"Constraint: {name}, Slack: {constraint.slack}")