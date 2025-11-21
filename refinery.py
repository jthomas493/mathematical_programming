# Databricks notebook source
# MAGIC %pip install pulp

# COMMAND ----------

# %restart_python

# COMMAND ----------

from pulp import *

# COMMAND ----------

crude1_availability = 20_000
crude2_availability = 30_000
distilled_limit = 45_000
reformed_limit = 10_000
cracking_limit = 8_000
lube_oil_lowbound = 500
lube_oil_upbound = 1_000
regular_octane_limit = 84
premium_octane_limit = 94
vapor_pressure_limit = 1

# COMMAND ----------

crude1 = LpVariable("Crude 1", lowBound=0, cat="Continuous")
crude2 = LpVariable("Crude 2", lowBound=0, cat="Continuous")

LN  = LpVariable("light naphtha", lowBound=0, cat="Continuous")
MN = LpVariable("medium naphtha", lowBound=0, cat="Continuous")
HN = LpVariable("heavy naphtha", lowBound=0, cat="Continuous")

LO = LpVariable("light oil", lowBound=0, cat="Continuous")
HO = LpVariable("heavy oil", lowBound=0, cat="Continuous")

R = LpVariable("residuum", lowBound=0, cat="Continuous")

LNRG = LpVariable("light naphtha used to produce reformed gasoline", lowBound=0, cat="Continuous")
MNRG = LpVariable("medium naphtha used to produce reformed gasoline", lowBound=0, cat="Continuous")
HNRG = LpVariable("heavy naphtha used to produce reformed gasoline", lowBound=0, cat="Continuous")
RG = LpVariable("reformed gasoline", lowBound=0, cat="Continuous")

LOCGO = LpVariable("light oil used to produce cracked oil and cracked gasoline", lowBound=0, cat="Continuous")
HOCGO = LpVariable("heavy oil used to produce cracked oil and cracked gasoline", lowBound=0, cat="Continuous")

CG = LpVariable("cracked gasoline", lowBound=0, cat="Continuous")
CO = LpVariable("cracked oil", lowBound=0, cat="Continuous")

LNPMF = LpVariable("light naphtha used to produce premium motor fuel", lowBound=0, cat="Continuous")
LNRMF = LpVariable("light naphtha used to produce regular motor fuel", lowBound=0, cat="Continuous")
MNPMF = LpVariable("medium naphtha used to produce premium motor fuel", lowBound=0, cat="Continuous")
MNRMF = LpVariable("medium naphtha used to produce regular motor fuel", lowBound=0, cat="Continuous")
HNPMF = LpVariable("heavy naphtha used to produce premium motor fuel", lowBound=0, cat="Continuous")
HNRMF = LpVariable("heavy naphtha used to produce regular motor fuel", lowBound=0, cat="Continuous")

RGPMF = LpVariable("reformed gasoline used to produce premium motor fuel", lowBound=0, cat="Continuous")
RGRMF = LpVariable("reformed gasoline used to produce regular motor fuel", lowBound=0, cat="Continuous")

CGPMF = LpVariable("cracked gasoline used to produce premium motor fuel", lowBound=0, cat="Continuous")
CGRMF = LpVariable("cracked gasoline used to produce regular motor fuel", lowBound=0, cat="Continuous")

LOJF = LpVariable("light oil used to produce jet fuel", lowBound=0, cat="Continuous")
HOJF = LpVariable("heavy oil used to produce jet fuel", lowBound=0, cat="Continuous")

RJF = LpVariable("residuum used to produce jet fuel", lowBound=0, cat="Continuous")
COJF = LpVariable("cracked oil used to produce jet fuel", lowBound=0, cat="Continuous")

RLBO = LpVariable("residuum used to produce lube-oil", lowBound=0, cat="Continuous")

PMF = LpVariable("premium motor fuel", lowBound=0, cat="Continuous")
RMF = LpVariable("regular motor fuel", lowBound=0, cat="Continuous")

JF = LpVariable("jet fuel", lowBound=0, cat="Continuous")
FO = LpVariable("Fuel oil", lowBound=0, cat="Continuous")
LBO = LpVariable("Lube Oil", lowBound=500, upBound=1000, cat="Continuous")


# COMMAND ----------

prob = LpProblem("Refinery",LpMaximize)

# COMMAND ----------

#Objective Function
prob += 7 * PMF + 6 * RMF + 4 * JF + 3.5 * FO + 1.5 * LBO

# COMMAND ----------

#crude oil availability limit
prob += crude1 <= crude1_availability
prob += crude2 <= crude2_availability

# COMMAND ----------

#Distillation limit
prob += crude1 + crude2 <= distilled_limit
#reforming limit
prob += LNRG + MNRG + HNRG <= reformed_limit
#cracking limit 
prob += LOCGO + HOCGO <= cracking_limit


# COMMAND ----------

#continuities
#light naphtha
prob += -0.1 * crude1 - 0.15 * crude2 + LN == 0
#medium naphtha
prob += -0.2 * crude1 - 0.25 * crude2 + MN == 0 
#heavy naphtha
prob += -0.2 * crude1 - 0.18 * crude2 + HN == 0 
#light oil
prob += -0.12 * crude1 - 0.08 * crude2 + LO == 0 
#heavy oil
prob += -0.2 * crude1 - 0.19 * crude2 + HO == 0 
#residuum
prob += -0.13 * crude1 - 0.12 * crude2 + R == 0 

# COMMAND ----------

#reformed gasoline depends on naphtha used to reform
prob += -0.6 * LNRG - 0.52 * MNRG - 0.45 * HNRG + RG == 0

#cracked oil and cracked gasoline produced depend on LO and HO used
prob += - 0.68 * LOCGO - 0.75 * HOCGO + CO == 0
prob += - 0.28 * LOCGO - 0.2 * HOCGO + CG == 0

# COMMAND ----------

# lube-oil produced (and sold) is 0.5 times the quantity of residuum used
prob += -0.5 * RLBO + LBO == 0

# quantities of light naphtha used for reforming and blending are equal to the quantities available
prob += -LN + LNRG + LNPMF + LNPMF == 0 
prob += -MN + MNRG + MNPMF + MNPMF == 0 
prob += -HN + HNRG + HNPMF + HNPMF == 0 

# COMMAND ----------

#blending of fuel oil, the proportion of oil and residuum is fixed
prob += -LO + LOCGO + LOJF + 0.55 * FO == 0

prob += -HO + HOCGO + HOJF + .22 * FO == 0 

prob += -CO +  CG + COJF + .16 * FO == 0

prob += -R + RLBO + RJF + .05 * FO == 0

# COMMAND ----------

# The quantity of premium motor fuel produced is equal to the total quantity of its ingredients. This gives
prob += -LNPMF - MNPMF - HNPMF - RGPMF - CGPMF + PMF == 0
prob += -LNRMF - MNRMF - HNRMF - RGRMF - CGRMF + RMF == 0
prob += -LOJF - HOJF - RJF - COJF + JF == 0

# COMMAND ----------

# Premium motor fuel production must be at least 40% of regular motor fuel production
prob += PMF - 0.4 * RMF >=0

# COMMAND ----------

# Qualities  
#Premium octane
prob += -90 * LNPMF + 80 * MNPMF - 70 * HNPMF - 115 * RGPMF - 105 * CGPMF + 94 * PMF <= 0
#regular octane
prob += -90 * LNRMF + 80 * MNRMF - 70 * HNRMF - 115 * RGRMF - 105 * CGRMF + 84 * RMF <= 0
# jet vapour pressure
prob += -LOJF - 0.6 * HOJF - 1.5 * COJF - 0.05 * RJF + JF >= 0

# COMMAND ----------

prob.solve()

# COMMAND ----------

print(f"Total Profit: ${value(prob.objective):,.2f}")

# COMMAND ----------

for v in prob.variables():
    print(f"{v.name} = {v.varValue}")