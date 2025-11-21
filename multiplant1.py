# Databricks notebook source
# MAGIC %pip install pulp

# COMMAND ----------

# %restart_python

# COMMAND ----------

import pandas as pd
from pulp import *

# COMMAND ----------

products = ["Standard", "Deluxe"]
work = ["Grinding", "Polishing"]
profit = {"Standard":10,"Deluxe":15}

# COMMAND ----------

Ahours = {
   ('Standard',"Grinding"):4.0, ("Standard","Polishing"):2.0,
   ("Deluxe","Grinding"):2.0, ("Deluxe","Polishing"):5.0
}

# COMMAND ----------

Bhours = {
   ('Standard',"Grinding"):5.0, ("Standard","Polishing"):5.0,
   ("Deluxe","Grinding"):3.0, ("Deluxe","Polishing"):6.0
}

# COMMAND ----------

Alimits = {"Grinding":80, "Polishing":60}
Blimits = {"Grinding":60, "Polishing":75}

# COMMAND ----------

# 2. Create the LP Problem
prob1 = LpProblem("factory_A",LpMaximize)
prob2 = LpProblem("factory_B",LpMaximize)

# COMMAND ----------

# 3. Define Decision Variables
make = pulp.LpVariable.dicts("Production", (p for p in products), lowBound=0,cat='Continuous')

# COMMAND ----------

# 4. Define the Objective Function
prob1 += lpSum([profit[p] * make[p] for p in products]), "Total Profit A"
prob2 += lpSum([profit[p] * make[p] for p in products]), "Total Profit B"

# COMMAND ----------

#5 constraint of raw material
prob1 += lpSum([4 * make[p] for p in products])<=75, "Raw A"
prob2 += lpSum([4 * make[p] for p in products])<=45, "Raw B"

# COMMAND ----------

# for w in work:
#     test = [Ahours[p,w] * make[p] for p in products]
# test 

# COMMAND ----------

#5 constraint of processing
for w in work: 
    prob1 += lpSum([Ahours[p,w] * make[p] for p in products])<= Alimits[w], f"{w} A"
    prob2 += lpSum([Bhours[p,w] * make[p] for p in products])<= Blimits[w], f"{w} B"

# COMMAND ----------

prob1.solve()

# COMMAND ----------

print("Status:", LpStatus[prob1.status])
print("Total Profit:", value(prob1.objective))

# 8. Create DataFrames
prod_data = {}

for p in products:
    prod_data[p] = make[p].varValue

print(prod_data)

# COMMAND ----------

prob2.solve()

# COMMAND ----------

print("Status:", LpStatus[prob2.status])
print("Total Profit:", value(prob2.objective))

# 8. Create DataFrames
prod_data = {}

for p in products:
    prod_data[p] = make[p].varValue
   
print(prod_data)