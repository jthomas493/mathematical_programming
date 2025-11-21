# Databricks notebook source
# MAGIC %pip install pulp

# COMMAND ----------

# %restart_python

# COMMAND ----------

import pandas as pd
from pulp import *

# COMMAND ----------

products = ['PROD1', 'PROD2', 'PROD3', 'PROD4', 'PROD5', 'PROD6', 'PROD7']
months = ['January', 'February', 'March', 'April', 'May', 'June']
profit = {'PROD1': 10, 'PROD2': 6, 'PROD3': 8, 'PROD4': 11, 'PROD5': 11,'PROD6':9, 'PROD7':3}

# COMMAND ----------

hours = {
    "Grinding":     {"PROD1": 0.5, "PROD2": 0.7, "PROD5": 0.3, "PROD6": 0.2, "PROD7": 0.5},
    "VerticalDrilling": {"PROD1": 0.1, "PROD2": 0.2, "PROD4": 0.3, "PROD6": 0.6},
    "HorizontalDrilling": {"PROD1": 0.2, "PROD3": 0.8, "PROD7": 0.6},
    "Boring":       {"PROD1": 0.05, "PROD2": 0.03, "PROD4": 0.07, "PROD5": 0.1, "PROD7": 0.08},
    "Planing":      {"PROD3": 0.01, "PROD5": 0.05, "PROD7": 0.05}
}


# COMMAND ----------

availability =  {
    ('January', 'GRINDING'): 1152.0, ('January', 'V_DRILL'): 768, ('January', 'H_DRILL'):1152, ('January', 'BORE'): 384, ('January', 'PLANE'): 384,
    ('February', 'GRINDING'): 1536.0, ('February', 'V_DRILL'): 768, ('February', 'H_DRILL'): 384, ('February', 'BORE'): 384, ('February', 'PLANE'): 384,
    ('March', 'GRINDING'): 1536.0, ('March', 'V_DRILL'): 768, ('March', 'H_DRILL'): 1152, ('March', 'BORE'): 0.0, ('March', 'PLANE'): 384,
    ('April', 'GRINDING'): 1536.0, ('April', 'V_DRILL'): 384, ('April', 'H_DRILL'): 1152, ('April', 'BORE'): 384, ('April', 'PLANE'): 384,
    ('May', 'GRINDING'): 1152.0, ('May', 'V_DRILL'): 384, ('May', 'H_DRILL'): 1152, ('May', 'BORE'): 384, ('May', 'PLANE'): 384,
    ('June', 'GRINDING'): 1536.0, ('June', 'V_DRILL'): 768, ('June', 'H_DRILL'): 768, ('June', 'BORE'): 384, ('June', 'PLANE'): 0.0
}

# COMMAND ----------

marketing = {
    "January": {"PROD1": 500, "PROD2": 1000, "PROD3": 300, "PROD4": 300, "PROD5": 800, "PROD6": 200, "PROD7": 100},
    "February": {"PROD1": 600, "PROD2": 500, "PROD3": 200, "PROD4": 0, "PROD5": 400, "PROD6": 300, "PROD7": 150},
    "March": {"PROD1": 300, "PROD2": 600, "PROD3": 0, "PROD4": 0, "PROD5": 500, "PROD6": 400, "PROD7": 100},
    "April": {"PROD1": 200, "PROD2": 300, "PROD3": 400, "PROD4": 500, "PROD5": 200, "PROD6": 0, "PROD7": 100},
    "May": {"PROD1": 0, "PROD2": 100, "PROD3": 500, "PROD4": 100, "PROD5": 1000, "PROD6": 300, "PROD7": 0},
    "June": {"PROD1": 500, "PROD2": 500, "PROD3": 100, "PROD4": 300, "PROD5": 1100, "PROD6": 500, "PROD7": 60}
}

# COMMAND ----------

availability = {
    "Grinding":     4,
    "VerticalDrilling": 2,
    "HorizontalDrilling": 3,
    "Boring":       1,
    "Planing":      1
}

# COMMAND ----------

downtime = {
    "January": {"Grinding": 1, "VerticalDrilling": 0, "HorizontalDrilling": 0, "Boring": 0, "Planing": 0},
    "February": {"Grinding": 0, "VerticalDrilling": 0, "HorizontalDrilling": 2, "Boring": 0, "Planing": 0},
    "March": {"Grinding": 0, "VerticalDrilling": 0, "HorizontalDrilling": 0, "Boring": 1, "Planing": 0},
    "April": {"Grinding": 0, "VerticalDrilling": 1, "HorizontalDrilling": 0, "Boring": 0, "Planing": 0},
    "May": {"Grinding": 1, "VerticalDrilling": 1, "HorizontalDrilling": 0, "Boring": 0, "Planing": 0},
    "June": {"Grinding": 0, "VerticalDrilling": 0, "HorizontalDrilling": 1, "Boring": 0, "Planing": 1}
}

# COMMAND ----------

storage_capacity = 100
storage_cost = 0.5
target_inventory = 50
working_days_per_month = 24
hours_per_day = 16

# COMMAND ----------

# 2. Create the LP Problem
prob = LpProblem("factory_planning",LpMaximize)

# COMMAND ----------

# 3. Define Decision Variables
production = pulp.LpVariable.dicts("Production", ((p, m) for p in products for m in months), lowBound=0,cat='Continuous')
inventory = pulp.LpVariable.dicts("Inventory", ((p, m) for p in products for m in months), lowBound=0, cat='Continuous')
sales = pulp.LpVariable.dicts("Sales", ((p, m) for p in products for m in months), lowBound=0, cat='Continuous')

# COMMAND ----------

# 4. Define the Objective Function
prob += pulp.lpSum([profit[p] * sales[p, m] - storage_cost * inventory[p, m]
                       for p in products for m in months]), "Total Profit"

# COMMAND ----------

# 5. Define the Constraints
# Inventory Balance Constraints
for p in products:
    for m_index, m in enumerate(months):
        if m_index == 0:
            # First month: initial inventory is zero
            prob += production[p, m] == sales[p, m] + inventory[p, m], f"InventoryBalance_{p}_{m}"
        else:
            m_prev = months[m_index - 1]
            prob += inventory[p, m_prev] + production[p, m] == sales[p, m] + inventory[p, m], f"InventoryBalance_{p}_{m}"

# COMMAND ----------

# Marketing Limits
for p in products:
    for m in months:
        prob += sales[p, m] <= marketing[m][p], f"MarketingLimit_{p}_{m}"

# COMMAND ----------

# Storage Limits
for p in products:
    for m in months:
        prob += inventory[p, m] <= storage_capacity, f"StorageLimit_{p}_{m}"

# COMMAND ----------

# Final Inventory Constraint
for p in products:
    prob += inventory[p, "June"] == target_inventory

# COMMAND ----------

# Machine Capacity Constraints
for m in months:
    available_hours = working_days_per_month * hours_per_day
    for machine in availability:
        machine_available = availability[machine] - downtime[m][machine]
        total_machine_hours = machine_available * available_hours
        machine_usage = pulp.lpSum([hours[machine].get(p, 0) * production[p, m] for p in products if machine in hours and p in hours[machine]]) # corrected to consider only the products the machine works on
        prob += machine_usage <= total_machine_hours, f"{machine}_{m}_Capacity"

# COMMAND ----------

prob.solve()

# COMMAND ----------

# 7. Print the Results
print("Status:", LpStatus[prob.status])
print("Total Profit:", value(prob.objective))

# 8. Create DataFrames
make_data = {}
sell_data = {}
store_data = {}

for m in months:
    make_data[m] = {}
    sell_data[m] = {}
    store_data[m] = {}
    for p in products:
        make_data[m][p] = production[p, m].varValue
        sell_data[m][p] = sales[p, m].varValue
        store_data[m][p] = inventory[p, m].varValue

make_df = pd.DataFrame(make_data).T  # Transpose to have months as rows
sell_df = pd.DataFrame(sell_data).T  # Transpose to have months as rows
store_df = pd.DataFrame(store_data).T # Transpose to have months as rows


print("\n--- Make DataFrame ---")
print(make_df)

print("\n--- Sell DataFrame ---")
print(sell_df)

print("\n--- Store DataFrame ---")
print(store_df)

# COMMAND ----------

print("Constraint Slack Values:")
for name, constraint in prob.constraints.items():
    print(f"Constraint: {name}, Slack: {constraint.slack}")
print("Shadow Prices : The Duals")
for name, constraint in prob.constraints.items():   
    print(f"Constraint:{name}, Dual: {constraint.pi}")