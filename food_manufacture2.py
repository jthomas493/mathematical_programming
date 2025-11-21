# Databricks notebook source
# MAGIC %pip install pulp

# COMMAND ----------

# %restart_python

# COMMAND ----------

import pandas as pd
from pulp import *

# COMMAND ----------

oils = ['VEG1', 'VEG2', 'OIL1', 'OIL2', 'OIL3']
months = ['January', 'February', 'March', 'April', 'May', 'June']

# COMMAND ----------

prices = {
    ('VEG1', 'January'): 110, ('VEG1', 'February'): 130, ('VEG1', 'March'): 110, ('VEG1', 'April'): 120, ('VEG1', 'May'): 100, ('VEG1', 'June'): 90,
    ('VEG2', 'January'): 120, ('VEG2', 'February'): 130, ('VEG2', 'March'): 140, ('VEG2', 'April'): 110, ('VEG2', 'May'): 120, ('VEG2', 'June'): 100,
    ('OIL1', 'January'): 130, ('OIL1', 'February'): 110, ('OIL1', 'March'): 130, ('OIL1', 'April'): 120, ('OIL1', 'May'): 150, ('OIL1', 'June'): 140,
    ('OIL2', 'January'): 110, ('OIL2', 'February'): 90, ('OIL2', 'March'): 100, ('OIL2', 'April'): 120, ('OIL2', 'May'): 110, ('OIL2', 'June'): 80,
    ('OIL3', 'January'): 115, ('OIL3', 'February'): 115, ('OIL3', 'March'): 95, ('OIL3', 'April'): 125, ('OIL3', 'May'): 105, ('OIL3', 'June'): 135,
}

# COMMAND ----------

hardness = {'VEG1': 8.8, 'VEG2': 6.1, 'OIL1': 2.0, 'OIL2': 4.2, 'OIL3': 5.0}

# COMMAND ----------

veg_refining_capacity = 200
non_veg_refining_capacity = 250
storage_capacity = 1000
storage_cost = 5
selling_price = 150
initial_inventory = 500
target_inventory = 500
hardness_min = 3
hardness_max = 6
min_usage = 20

# COMMAND ----------

# 2. Create the LP Problem
prob = LpProblem("food_manufacture1",LpMaximize)

# COMMAND ----------

# 3. Define Decision Variables
buy = pulp.LpVariable.dicts("Buy", [(o, m) for o in oils for m in months], lowBound=0, cat='Continuous')
use = pulp.LpVariable.dicts("Use", [(o, m) for o in oils for m in months], lowBound=0, cat='Continuous')
store = pulp.LpVariable.dicts("Store", [(o, m) for o in oils for m in months], lowBound=0, cat='Continuous')
produce = pulp.LpVariable.dicts("Produce", [m for m in months], lowBound=0, cat='Continuous')
used_oil = pulp.LpVariable.dicts("Used_Oil", [(o, m) for o in oils for m in months], cat='Binary')

# COMMAND ----------

# 4. Define the Objective Function
prob += pulp.lpSum([selling_price * produce[m] for m in months]) - \
        pulp.lpSum([prices[o, m] * buy[o, m] for o in oils for m in months]) - \
        pulp.lpSum([storage_cost * store[o, m] for o in oils for m in months])

# COMMAND ----------

# 5. Define the Constraints
# Inventory Balance Constraints
for o in oils:
    for i, m in enumerate(months):
        if i == 0:  # January
            prob += initial_inventory + pulp.lpSum([buy[o, m]]) == pulp.lpSum([use[o, m]]) + pulp.lpSum([store[o, m]])
        else:
            prob += store[o, months[i-1]] + pulp.lpSum([buy[o, m]]) == pulp.lpSum([use[o, m]]) + pulp.lpSum([store[o, m]])

# COMMAND ----------

# Refining Capacity Constraints
for m in months:
    prob += pulp.lpSum([use['VEG1', m]]) + pulp.lpSum([use['VEG2', m]]) <= veg_refining_capacity
    prob += pulp.lpSum([use['OIL1', m]]) + pulp.lpSum([use['OIL2', m]]) + pulp.lpSum([use['OIL3', m]]) <= non_veg_refining_capacity

# COMMAND ----------

# Hardness Constraints
for m in months:
    prob += pulp.lpSum([hardness[o] * use[o, m] for o in oils]) >= hardness_min * produce[m]
    prob += pulp.lpSum([hardness[o] * use[o, m] for o in oils]) <= hardness_max * produce[m]

# COMMAND ----------

# Production = usage constraint
for m in months:
    prob += pulp.lpSum([use[o, m] for o in oils]) == produce[m]

# COMMAND ----------

# Final Inventory Constraint
for o in oils:
    prob += store[o, months[-1]] == target_inventory

# COMMAND ----------

#maximum 3 oils per month
for m in months:
    prob += pulp.lpSum([used_oil[o,m]for o in oils]) <= 3

# COMMAND ----------

# if used oil, use more than 20
for o in oils:
    for m in months:
        prob += use[o, m] <= storage_capacity * used_oil[o, m] # If use > 0, force used_oil = 1
        prob += use[o, m] >= min_usage * used_oil[o, m]

# COMMAND ----------

# If VEG1 or VEG2 are used, then OIL3 must be used
for m in months:
    prob += used_oil['VEG1', m] + used_oil['VEG2', m] <= 2 * used_oil['OIL3', m]

# COMMAND ----------

prob.solve()

# COMMAND ----------

# 7. Print the Results
print("Status:", LpStatus[prob.status])
print("Total Profit:", value(prob.objective))

# 8. Create DataFrames
buy_data = {}
use_data = {}
store_data = {}
used_oil_data = {}

for m in months:
    buy_data[m] = {}
    use_data[m] = {}
    store_data[m] = {}
    used_oil_data[m] = {} 

    for o in oils:
        buy_data[m][o] = buy[o, m].varValue
        use_data[m][o] = use[o, m].varValue
        store_data[m][o] = store[o, m].varValue
        used_oil_data[m][o] = used_oil[o, m].varValue

buy_df = pd.DataFrame(buy_data).T  # Transpose to have months as rows
use_df = pd.DataFrame(use_data).T  # Transpose to have months as rows
store_df = pd.DataFrame(store_data).T # Transpose to have months as rows
used_oil_df = pd.DataFrame(used_oil_data).T # Transpose to have months as rows

print("\n--- Buy DataFrame ---")
print(buy_df)

print("\n--- Use DataFrame ---")
print(use_df)

print("\n--- Store DataFrame ---")
print(store_df)

print("\n--- Used Oil DataFrame ---") #NEW
print(used_oil_df)