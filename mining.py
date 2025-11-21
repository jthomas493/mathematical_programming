# Databricks notebook source
# MAGIC %pip install pulp

# COMMAND ----------

# %restart_python

# COMMAND ----------

import pandas as pd
from pulp import *

# COMMAND ----------

mines = [1,2,3,4]
years = [1,2,3,4,5]
selling_price = 10
discount_rate = 0.1

# COMMAND ----------

#royalities to mine in millions
royalties = {
    1:5,
    2:4,
    3:4,
    4:5
}

# COMMAND ----------

#mine production limits in tons
production_limits = {
    1:2e6,
    2:2.5e6,
    3:1.3e6,
    4:3e6
}

# COMMAND ----------

#qualities of mines
qualities = {
    1:1.0,
    2:0.7,
    3:1.5,
    4:0.5,
}

# COMMAND ----------

#target qualites per year
target_qualities = {
    1:0.9,
    2:0.8,
    3:1.2,
    4:0.6,
    5:1.0,
}

# COMMAND ----------

prob = LpProblem("Mining_Problem", LpMaximize)

# COMMAND ----------

# Decision Variables
operate = pulp.LpVariable.dicts("Operate", [(m, y) for m in mines for y in years], lowBound=0, cat='Continuous')
isOpen = pulp.LpVariable.dicts("IsOpen", [(m, y) for m in mines for y in years], cat='Binary')
#closed for the whole duration
isClosed = pulp.LpVariable.dicts("IsClosed", [m for m in mines], cat='Binary') 

# COMMAND ----------

# Discount factor calculation
discount_factors = {year: 1 / (1 + discount_rate)**(year - 1) for year in years}

# COMMAND ----------

#Objective Function
prob += pulp.lpSum(
    [discount_factors[y] * (selling_price * operate[m, y] - royalties[m] * 1e6 * isOpen[m, y]) for m in mines for y in years]
) - pulp.lpSum([royalties[m]*1e6 * discount_factors[1] *isClosed[m] for m in mines]) , "Total_Discounted_Profit"


# COMMAND ----------

#Constraints
# Operating Limit
for y in years:
    prob += pulp.lpSum([isOpen[m, y] for m in mines]) <= 3, f"Operating_Limit_{y}"


# COMMAND ----------

# Production Limits
for m in mines:
    for y in years:
        prob += operate[m, y] <= production_limits[m] * isOpen[m,y], f"Production_Limit_{m}_{y}"  #A mine can only operate if it's open


# COMMAND ----------

# Quality Blend
for y in years:
    prob += pulp.lpSum([qualities[m] * operate[m, y] for m in mines]) == target_qualities[y] * pulp.lpSum([operate[m, y] for m in mines]), f"Quality_Blend_{y}"


# COMMAND ----------

# Royalties & isOpen Relationship (Key Constraint)
for m in mines:
    for y in years[:-1]: # All years except the last
        for future_y in years[y:]:
             prob += isOpen[m,y] >= isOpen[m, future_y], f"Royalties_{m}_{y}_{future_y}"


# COMMAND ----------

# Closure Constraints
for m in mines:
    for y in years:
        prob += operate[m,y] <= production_limits[m]*(1-isClosed[m]), f"Closure_limit_{m}_{y}"

    for y in years[:-1]:
        prob += isOpen[m,y] <= (1-isClosed[m]), f"Closure_open_{m}_{y}"

# COMMAND ----------

print(len(prob.constraints))

# COMMAND ----------

prob.solve()

# COMMAND ----------

total_profit = pulp.value(prob.objective)
print(f"Total Discounted Profit: £{total_profit/1e6:.2f} million") #in millions


# COMMAND ----------

for m in mines:
    print(f"\nMine {m}:")
    for y in years:
        op_amount = pulp.value(operate[m, y])
        is_open = pulp.value(isOpen[m, y])
        print(f"  Year {y}: Operate = {op_amount:.2f} tons, Open = {is_open}")
    is_perm_closed = pulp.value(isClosed[m])
    print(f"  Permanently Closed: {is_perm_closed}")

# COMMAND ----------

# Create DataFrame for operate
operate_data = {}
for m in mines:
    operate_data[m] = [pulp.value(operate[m, y]) for y in years]

df_operate = pd.DataFrame(operate_data, index=years)
df_operate.index.name = 'Year'
df_operate.columns.name = 'Mine'
print("\nOre Extracted (tons):")
print(df_operate)

# Create DataFrame for isOpen
is_open_data = {}
for m in mines:
    is_open_data[m] = [pulp.value(isOpen[m, y]) for y in years]

df_is_open = pd.DataFrame(is_open_data, index=years)
df_is_open.index.name = 'Year'
df_is_open.columns.name = 'Mine'
print("\nMine Open Status:")
print(df_is_open)


# Print Closure Status
closure_status = {m: pulp.value(isClosed[m]) for m in mines}
print("\nClosure Status:")
print(closure_status)

# Print Quality values
print("\nMine Qualities:")
print(qualities)

# Create DataFrame for target Qualities
qualities_series = pd.Series(target_qualities)
print("\nTarget Qualities:")
print(qualities_series)

# COMMAND ----------

print("Constraint Slack Values:")
for name, constraint in prob.constraints.items():
    if value(constraint.slack) > 0 or value(constraint.slack) < 0 :
     print(f"Constraint: {name}, Slack: {constraint.slack}")
print("Shadow Prices : The Duals")
for name, constraint in prob.constraints.items():
    if value(constraint.pi) > 0 or value(constraint.pi) < 0:   
     print(f"Constraint:{name}, Dual: {constraint.pi}")