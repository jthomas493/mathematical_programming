# Databricks notebook source
# MAGIC %pip install pulp

# COMMAND ----------

# %restart_python

# COMMAND ----------

import pandas as pd
from pulp import *

# COMMAND ----------

timeframe =[0,1,2,3]

# COMMAND ----------

skilled_requirements = [1000,1000,1500,2000]
semiskilled_requirements = [1500,1400,2000,2500]
unskilled_requirements = [2000,1000,500, 0]

# COMMAND ----------

#Strength of Force
SK = LpVariable.dicts("SkilledForce",[t for t in timeframe], lowBound=0, cat="Integer")
SS = LpVariable.dicts("SemiSkilledForce",[t for t in timeframe], lowBound=0, cat="Integer")
US = LpVariable.dicts("UnSkilledForce",[t for t in timeframe], lowBound=0, cat="Integer")


# COMMAND ----------

#Recruitment of Force
Recruit_SK = LpVariable.dicts("Recruit Skilled",[t for t in timeframe], lowBound=0, cat="Integer")
Recruit_SS = LpVariable.dicts("Recruit SemiSkilled",[t for t in timeframe], lowBound=0, cat="Integer")
Recruit_US = LpVariable.dicts("Recruit UnSkilled",[t for t in timeframe], lowBound=0, cat="Integer")

# COMMAND ----------

#Retrain of the Force
Retrain_US_SS = LpVariable.dicts("Retrain Unskilled to SemiSkilled",[t for t in timeframe], lowBound=0, cat="Integer")
Retrain_SS_SK = LpVariable.dicts("Retrain SemiSkilled to Skilled",[t for t in timeframe], lowBound=0, cat="Integer")

# COMMAND ----------

#Downgrading of Force
Downgrade_SK_SS = LpVariable.dicts("Downgrade Skilled to SemiSkilled",[t for t in timeframe], lowBound=0, cat="Integer")
Downgrade_SK_US = LpVariable.dicts("Downgrade Skilled to Unskilled",[t for t in timeframe], lowBound=0, cat="Integer")
Downgrade_SS_US = LpVariable.dicts("Downgrade SemiSkilled to UnSkilled",[t for t in timeframe], lowBound=0, cat="Integer")

# COMMAND ----------

#Redundancy of Force
Redundancy_SK = LpVariable.dicts("Redundant Skilled",[t for t in timeframe], lowBound=0, cat="Integer")
Redundancy_SS = LpVariable.dicts("Redundant SemiSkilled",[t for t in timeframe], lowBound=0, cat="Integer")
Redundancy_US = LpVariable.dicts("Redundant UnSkilled",[t for t in timeframe], lowBound=0, cat="Integer")

# COMMAND ----------

#Short time workforce
Short_time_SK = LpVariable.dicts("Short-time Skilled",[t for t in timeframe], lowBound=0, cat="Integer")
Short_time_SS = LpVariable.dicts("Short-time SemiSkilled",[t for t in timeframe], lowBound=0, cat="Integer")
Short_time_US = LpVariable.dicts("Short-time UnSkilled",[t for t in timeframe], lowBound=0, cat="Integer")

# COMMAND ----------

#Overmanning Workforce
Overmanning_SK = LpVariable.dicts("Overmanning Skilled",[t for t in timeframe], lowBound=0, cat="Integer")
Overmanning_SS = LpVariable.dicts("Overmanning SemiSkilled",[t for t in timeframe], lowBound=0, cat="Integer")
Overmanning_US = LpVariable.dicts("Overmanning UnSkilled",[t for t in timeframe], lowBound=0, cat="Integer")

# COMMAND ----------

#initialize problem
prob = LpProblem("Manpower", LpMinimize)

# COMMAND ----------

#Objective Function
prob += lpSum([Redundancy_SK[i] + Redundancy_SS[i] + Redundancy_US[i] for i in timeframe])

# COMMAND ----------

#Continuity
#continuity of skilled
for i in timeframe:
    if i == 0:
        prob += 1000 == SK[i]
    else:
        prob += 0.95*SK[i-1] + 0.9*Recruit_SK[i]+ 0.95*Retrain_SS_SK[i] - Downgrade_SK_SS[i] - Downgrade_SK_US[i] - Redundancy_SK[i] == SK[i]
#continuity of SemiSkilled
for i in timeframe:
    if i == 0:
        prob += 1500 == SS[i]
    else:
        prob += 0.95*SS[i-1] + 0.8*Recruit_SS[i]+ 0.95*Retrain_US_SS[i]- Retrain_SS_SK[i] + 0.5*Downgrade_SK_SS[i] - Downgrade_SS_US[i] - Redundancy_SS[i] == SS[i]
#continuity of UnSkilled
for i in timeframe:
    if i == 0:
        prob += 2000 == US[i]
    else:
        prob += 0.9*US[i-1] + 0.75*Recruit_US[i] - Retrain_US_SS[i] + 0.5*Downgrade_SK_US[i] + 0.5*Downgrade_SS_US[i] - Redundancy_US[i] == US[i]

# COMMAND ----------

#Retraining SemiSkilled Workers
for i in timeframe:
    prob += Retrain_SS_SK[i] - 0.25*SK[i] <= 0, f"SemiSkilled Retraining Limit year {i}"

# COMMAND ----------

#Overemanning
for i in timeframe:
    prob += Overmanning_SK[i] + Overmanning_SS[i] + Overmanning_US[i] <= 150, f"Overmanning Limit year {i}"

# COMMAND ----------

#Manning Requirements
for i in timeframe:
    prob += SK[i] - Overmanning_SK[i] - 0.5*Short_time_SK[i] == skilled_requirements[i], f"skilled_requirements year {i}"
    prob += SS[i] - Overmanning_SS[i] - 0.5*Short_time_SS[i] == semiskilled_requirements[i], f"semiskilled_requirements year {i}"
    prob += US[i] - Overmanning_US[i] - 0.5*Short_time_US[i] == unskilled_requirements[i], f"unskilled_requirements year {i}"

# COMMAND ----------

#Recruitment limits
for i in timeframe:
    prob += Recruit_SK[i] <= 500
    prob += Recruit_SS[i] <= 800
    prob += Recruit_US[i] <= 500
  

# COMMAND ----------

# Shorttime limits
for i in timeframe:
    prob += Short_time_SK[i] <= 50
    prob += Short_time_SS[i] <= 50
    prob += Short_time_US[i] <= 50

# COMMAND ----------

#Retraining UnSkilled Workers
for i in timeframe:
    prob += Retrain_US_SS[i] <= 200

# COMMAND ----------

print(prob.numVariables())
len(prob.constraints)

# COMMAND ----------

# prob.solve()
#stop solving after x Secconds
prob.solve(PULP_CBC_CMD(timeLimit=300))

# COMMAND ----------

print(f"Total Redundant personnel: {value(prob.objective)}")
print(f"""Total costs: {value(lpSum([ 
                         400 * Retrain_US_SS[i] + 500 * Retrain_SS_SK[i] + 200 * Redundancy_US[i] + 500 * Redundancy_SS[i] + 500 * Redundancy_SK[i]
                         + 500 * Short_time_US[i] + 400 * Short_time_SS[i] + 400 * Short_time_SK[i]
                         + 1500 * Overmanning_US[i] + 2000 * Overmanning_SS[i] + 3000 * Overmanning_SK[i]
                         for i in timeframe])):,.2f}
        """)

# COMMAND ----------

print("\n---Recruitment---")
for t in timeframe:
    print(f"Skilled Workers {t}: {Recruit_SK[t].varValue}")
print("\n")
for t in timeframe:
    print(f"SemiSkilled Workers {t}: {Recruit_SS[t].varValue}")
print("\n")
for t in timeframe:
    print(f"UnSkilled Workers {t}: {Recruit_US[t].varValue}")
print("\n")

# COMMAND ----------

print("\n---Retrain & Downgrade---")
for t in timeframe:
    print(f"Unskilled to SemiSkilled {t}: {Retrain_US_SS[t].varValue}")
print("\n")
for t in timeframe:
    print(f"SemiSkilled to Skilled {t}: {Retrain_SS_SK[t].varValue}")
print("\n")
for t in timeframe:
    print(f"SemiSkilled to Unskilled {t}: {Downgrade_SS_US[t].varValue}")
print("\n")
for t in timeframe:
    print(f"Skilled to Unskilled {t}: {Downgrade_SS_US[t].varValue}")
print("\n")
for t in timeframe:
    print(f"Skilled to SemiSkilled {t}: {Downgrade_SK_SS[t].varValue}")
print("\n")

# COMMAND ----------

print("\n---Redundancy---")
for t in timeframe:
    print(f"Redundant Skilled Workers {t}: {Redundancy_SK[t].varValue}")
print("\n")
for t in timeframe:
    print(f"Redundant SemiSkilled Workers {t}: {Redundancy_SS[t].varValue}")
print("\n")
for t in timeframe:
    print(f"Redundant UnSkilled Workers {t}: {Redundancy_US[t].varValue}")
print("\n")

# COMMAND ----------

print("\n---Shorttime---")
for t in timeframe:
    print(f"Short time Skilled Workers {t}: {Short_time_SK[t].varValue}")
print("\n")
for t in timeframe:
    print(f"Short time SemiSkilled Workers {t}: {Short_time_SS[t].varValue}")
print("\n")
for t in timeframe:
    print(f"Shor time UnSkilled Workers {t}: {Short_time_US[t].varValue}")
print("\n")

# COMMAND ----------

print("\n---Overmanning---")
for t in timeframe:
    print(f"Overmanned Skilled Workers {t}: {Overmanning_SK[t].varValue}")
print("\n")
for t in timeframe:
    print(f"Overmanned SemiSkilled Workers {t}: {Overmanning_SS[t].varValue}")
print("\n")
for t in timeframe:
    print(f"Overmanned UnSkilled Workers {t}: {Overmanning_US[t].varValue}")
print("\n")

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### Second Problem
# MAGIC #### Minimize cost

# COMMAND ----------

prob.setObjective( lpSum([ 
                         400 * Retrain_US_SS[i] + 500 * Retrain_SS_SK[i] + 200 * Redundancy_US[i] + 500 * Redundancy_SS[i] + 500 * Redundancy_SK[i]
                         + 500 * Short_time_US[i] + 400 * Short_time_SS[i] + 400 * Short_time_SK[i]
                         + 1500 * Overmanning_US[i] + 2000 * Overmanning_SS[i] + 3000 * Overmanning_SK[i]
                         for i in timeframe
]))

# COMMAND ----------

print(prob.numVariables())
len(prob.constraints)

# COMMAND ----------

# prob.solve()
#Set max solving time to 300 seconds
prob.solve(PULP_CBC_CMD(timeLimit=300))

# COMMAND ----------

print(f"Total Costs: {value(prob.objective):,.2f}")
print(f"Total Redundant Personnel: {value(lpSum([Redundancy_SK[i] + Redundancy_SS[i] + Redundancy_US[i] for i in timeframe]))}")

# COMMAND ----------

print("\n---Recruitment---")
for t in timeframe:
    print(f"Skilled Workers {t}: {Recruit_SK[t].varValue}")
print("\n")
for t in timeframe:
    print(f"SemiSkilled Workers {t}: {Recruit_SS[t].varValue}")
print("\n")
for t in timeframe:
    print(f"UnSkilled Workers {t}: {Recruit_US[t].varValue}")
print("\n")

# COMMAND ----------

print("\n---Retrain & Downgrade---")
for t in timeframe:
    print(f"Unskilled to SemiSkilled {t}: {Retrain_US_SS[t].varValue}")
print("\n")
for t in timeframe:
    print(f"SemiSkilled to Skilled {t}: {Retrain_SS_SK[t].varValue}")
print("\n")
for t in timeframe:
    print(f"SemiSkilled to Unskilled {t}: {Downgrade_SS_US[t].varValue}")
print("\n")
for t in timeframe:
    print(f"Skilled to Unskilled {t}: {Downgrade_SS_US[t].varValue}")
print("\n")
for t in timeframe:
    print(f"Skilled to SemiSkilled {t}: {Downgrade_SK_SS[t].varValue}")
print("\n")

# COMMAND ----------

print("\n---Redundancy---")
for t in timeframe:
    print(f"Redundant Skilled Workers {t}: {Redundancy_SK[t].varValue}")
print("\n")
for t in timeframe:
    print(f"Redundant SemiSkilled Workers {t}: {Redundancy_SS[t].varValue}")
print("\n")
for t in timeframe:
    print(f"Redundant UnSkilled Workers {t}: {Redundancy_US[t].varValue}")
print("\n")

# COMMAND ----------

print("\n---Shorttime---")
for t in timeframe:
    print(f"Short time Skilled Workers {t}: {Short_time_SK[t].varValue}")
print("\n")
for t in timeframe:
    print(f"Short time SemiSkilled Workers {t}: {Short_time_SS[t].varValue}")
print("\n")
for t in timeframe:
    print(f"Shor time UnSkilled Workers {t}: {Short_time_US[t].varValue}")
print("\n")

# COMMAND ----------

print("\n---Overmanning---")
for t in timeframe:
    print(f"Overmanned Skilled Workers {t}: {Overmanning_SK[t].varValue}")
print("\n")
for t in timeframe:
    print(f"Overmanned SemiSkilled Workers {t}: {Overmanning_SS[t].varValue}")
print("\n")
for t in timeframe:
    print(f"Overmanned UnSkilled Workers {t}: {Overmanning_US[t].varValue}")
print("\n")

# COMMAND ----------

