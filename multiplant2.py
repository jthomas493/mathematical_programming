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
factories = ["FactoryA", "FatoryB"]

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
prob = LpProblem("factories",LpMaximize)

# COMMAND ----------

# 3. Define Decision Variables
makeA = pulp.LpVariable.dicts("ProductionA", (p for p in products), lowBound=0,cat='Continuous')
makeB = pulp.LpVariable.dicts("ProductionB", (p for p in products), lowBound=0,cat='Continuous')
# processing = pulp.LpVariable.dicts("ProcessA", ((f,w) for f in factories for w in work), lowBound=0,cat='Continuous')

# COMMAND ----------

# 4. Define the Objective Function
prob += lpSum([profit[p] * makeA[p] for p in products] + [profit[p] * makeB[p] for p in products]), "Total Profit"

# COMMAND ----------

#5 constraint of raw material
prob += lpSum([4 * makeA[p] for p in products] + [4 * makeB[p] for p in products])<=120, "Raw"

# COMMAND ----------

#5 constraint of processing
for w in work: 
    prob += lpSum([Ahours[p,w] * makeA[p] for p in products])<= Alimits[w], f"{w} A"
    prob += lpSum([Bhours[p,w] * makeB[p] for p in products])<= Blimits[w], f"{w} B"

# COMMAND ----------

prob.solve()

# COMMAND ----------

print("Status:", LpStatus[prob.status])
print("Total Profit:", value(prob.objective))

# 8. Create DataFrames
Aprod_data = {}
Bprod_data = {}
# process_data = {}

for p in products:
    Aprod_data[p] = makeA[p].varValue
    Bprod_data[p] = makeB[p].varValue

# for f in factories:
#     process_data[f] = {}
#     for w in work:
#         process_data[f][w] = processing[f, w].varValue

print(Aprod_data)
print(Bprod_data)
# print(process_data)