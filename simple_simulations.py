# Databricks notebook source
# %restart_python

# COMMAND ----------

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from random import random, randint, choices

# COMMAND ----------


heads = 0
tails = 0
trials = 100000

for i in range(trials):
    num = random()
    if num > 0.5:
        tails += 1
    else:
        heads +=1

print(f"Number of Heads = {heads}\nPercentage of Heads = {heads/trials * 100:.2f}\n")
print(f"Number of Tails = {tails}\nPercentage of Tails = {tails/trials * 100:.2f}")

# COMMAND ----------

def simulate_model(matrix, initial_state,num_steps,state_labels):
    current_state = initial_state
    path = [state_labels[current_state]]

    for i in range(num_steps - 1):
        probabilities = matrix[current_state]
        next_state = choices(range(len(state_labels)), weights=probabilities, k=1)[0]
        current_state = next_state
        path.append(state_labels[current_state])
    return path

# COMMAND ----------

states = ["Sunny", "Rainy"]

transition_matrix = np.array([
    [0.8,0.2],
    [0.4,0.6]
])

# COMMAND ----------

plt.figure(figsize=(5,5))

G = nx.DiGraph()

for i , state_i in enumerate(states):
    for j, state_j in enumerate(states):
        if transition_matrix[i,j] > 0:
            G.add_edge(state_i,state_j,weight=transition_matrix[i,j])

pos = nx.circular_layout(G) # Or other layouts like spring_layout, spectral_layout
nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)
nx.draw_networkx_labels(G, pos)
nx.draw_networkx_edges(G, pos, arrowsize=20, edge_color='gray', connectionstyle='arc3,rad=0.2')
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.title("Sunny-Rainy Markov Network")
plt.show()           

# COMMAND ----------

initial_state = 0
steps = 50
simulation_path = simulate_model(transition_matrix,initial_state,steps,states)

print(f"If today is {states[initial_state]}, in {steps} days it will likely be {simulation_path[-1]}")

# COMMAND ----------

cumulative_count={}
results = []

for i in simulation_path:
    if i not in cumulative_count:
        cumulative_count[i] = 0
    cumulative_count[i] +=1
    results.append({i : cumulative_count[i]})

# COMMAND ----------

df = pd.DataFrame(results)
df_clean = df.interpolate(method="linear")

# COMMAND ----------

plt.plot(df_clean, label=["Sunny","Rainy"])
plt.xlabel("Number of Days")
plt.ylabel("Number of Occurances")
plt.legend(loc="upper left")
plt.title("Cummulative Weather days over time")
plt.show()

# COMMAND ----------

flea_states = ["Glee Flea", "Forget Flea", "Purpose Flea", "Skill Flea", "Dumb Flea", "Angry Flea", "Happ Flea"]
flea_matrix = np.array([
    [0.2,0.6,0.0,0.0,0.2,0.0,0.0],
    [0.0,0.0,1.0,0.0,0.0,0.0,0.0],
    [0.0,.66,0.0,.33,0.0,0.0,0.0],
    [0.0,1.0,0.0,0.0,0.0,0.0,0.0],
    [0.0,0.0,0.0,0.0,0.0,1.0,0.0],
    [0.0,0.0,0.0,0.0,0.0,0.0,1.0],
    [0.0,0.0,0.0,0.0,1.0,0.0,0.0]
])

# COMMAND ----------

plt.figure(figsize=(10,10))
flea_graph = nx.DiGraph()

for i , state_i in enumerate(flea_states):
    for j, state_j in enumerate(flea_states):
        if flea_matrix[i,j] > 0:
            flea_graph.add_edge(state_i,state_j,weight=flea_matrix[i,j])

pos = nx.planar_layout(flea_graph) # Or other layouts like spring_layout, spectral_layout, circular_layout, kamada_kawai_layout
nx.draw_networkx_nodes(flea_graph, pos, node_color='lightblue', node_size=500)
nx.draw_networkx_labels(flea_graph, pos)
nx.draw_networkx_edges(flea_graph, pos, arrowsize=20, edge_color='gray')
edge_labels = nx.get_edge_attributes(flea_graph, 'weight')
nx.draw_networkx_edge_labels(flea_graph, pos, edge_labels=edge_labels)
plt.title("Flea States Markov Network")
plt.show()   

# COMMAND ----------

initial_flea_state = 0
steps = 1000
flea_path = simulate_model(flea_matrix,initial_flea_state,steps,flea_states)

print(f"If the fleea starts as {flea_states[initial_flea_state]}, in {steps} hops the flea will likely be {flea_path[-1]}")

# COMMAND ----------

cum_count = {}
results2 = []

for i in flea_path:
    if i not in cum_count:
        cum_count[i] = 0
    cum_count[i] +=1
    results2.append({i:cum_count[i]})

# COMMAND ----------

df2 = pd.DataFrame(results2)
df2_clean = df2.interpolate(method="linear")

# COMMAND ----------

labels = set(flea_path)

plt.plot(df2_clean, label=labels)
plt.xlabel("Number of Hops")
plt.ylabel("Number of Occurances")
plt.legend(loc="upper left")
plt.title("Cummulative Flea Locations over time")
plt.show()