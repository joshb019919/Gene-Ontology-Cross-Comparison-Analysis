#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 09:23:00 2026

@author: danielhier
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Integrated GO Structure + HiG2Vec Geometry Analysis
Author: Daniel Hier (Integrated version)

This script:
1. Loads GO ontology
2. Builds subtree from root (depth k)
3. Computes structural metrics
4. Loads HiG2Vec embeddings
5. Computes hyperbolic radius + theta
6. Outputs ONE integrated CSV
"""

# ==========================================================
# IMPORTS
# ==========================================================

import pandas as pd
import numpy as np
import networkx as nx
import torch
from anytree import NodeMixin
from sklearn.decomposition import PCA
import os

# ==========================================================
# CONFIGURATION
# ==========================================================

GO_CSV_PATH = "data/GO_cellular_component_fully_normalized.csv"
HIG2VEC_PATH = "data/hig2vec_human_1000dim.pth"
OUTPUT_DIR = "results/"

ROOT_ID = "GO:0043231"
k = 1  # depth to parse

USE_HIG2VEC = True  # toggle embedding analysis

# ==========================================================
# LOAD GO DATA
# ==========================================================

df = pd.read_csv(GO_CSV_PATH)
df["Class ID"] = df["Class ID"].astype(str).str.strip()

label_lookup = dict(zip(df["Class ID"], df["Preferred Label"]))

# ==========================================================
# BUILD FULL GO DAG
# ==========================================================

def build_full_go_graph(df):
    G = nx.DiGraph()
    for _, row in df.iterrows():
        child = row["Class ID"]
        parents = str(row["Parents"]).split("|")
        for parent in parents:
            parent = parent.strip()
            if parent:
                G.add_edge(parent, child)
    return G

G_full = build_full_go_graph(df)

# ==========================================================
# ANYTREE NODE CLASS
# ==========================================================

class GONode(NodeMixin):
    def __init__(self, GO_id, label=None, parent=None):
        self.GO_id = GO_id
        self.label = label
        self.parent = parent
        self.tree_depth = 0

# ==========================================================
# BUILD SUBTREE (DEPTH k)
# ==========================================================

def add_children(parent_id, parent_node, current_depth):
    if current_depth >= k:
        return
    
    children_df = df[df["Parents"].str.contains(parent_id, na=False)]
    
    for _, row in children_df.iterrows():
        child_id = row["Class ID"]
        child_label = label_lookup.get(child_id, child_id)

        child_node = GONode(child_id, child_label, parent=parent_node)
        child_node.tree_depth = current_depth + 1

        add_children(child_id, child_node, current_depth + 1)

# Create root
root_label = label_lookup.get(ROOT_ID, ROOT_ID)
root = GONode(ROOT_ID, root_label)
root.tree_depth = 0

add_children(ROOT_ID, root, 0)

# ==========================================================
# COMPUTE STRUCTURAL METRICS
# ==========================================================

all_nodes = [root] + list(root.descendants)
total_subtree_size = len(all_nodes)

rows = []

for node in all_nodes:
    branching_factor = len(node.children)
    subtree_size_k = 1 + len(node.descendants)
    is_leaf = branching_factor == 0
    
    global_children = df["Parents"].str.contains(node.GO_id, na=False).sum()
    global_descendants = len(nx.descendants(G_full, node.GO_id))
    
    subtree_prop = subtree_size_k / total_subtree_size

    rows.append({
        "GO_ID": node.GO_id,
        "Label": node.label,
        "Tree_Depth": node.tree_depth,
        "Branching_Factor": branching_factor,
        "Subtree_Size_k": subtree_size_k,
        "Is_Leaf": is_leaf,
        "Global_Children_Count": global_children,
        "Global_Descendant_Count": global_descendants,
        "Subtree_Proportion_of_Total": round(subtree_prop, 5)
    })

structural_df = pd.DataFrame(rows)

# ==========================================================
# HIG2VEC EMBEDDING METRICS
# ==========================================================

if USE_HIG2VEC:

    print("Loading HiG2Vec...")
    model = torch.load(HIG2VEC_PATH, map_location="cpu")
    objects = model["objects"]
    embeddings = model["embeddings"].numpy()
    
    go_to_index = {go: i for i, go in enumerate(objects)}
    
    present_ids = [go for go in structural_df["GO_ID"] if go in go_to_index]
    
    if present_ids:
        indices = [go_to_index[go] for go in present_ids]
        vectors = embeddings[indices, :]
        
        norms = np.linalg.norm(vectors, axis=1)
        norms = np.clip(norms, None, np.nextafter(1.0, 0.0))
        hyperbolic_r = 2.0 * np.arctanh(norms)
        
        pca = PCA(n_components=2)
        coords_2d = pca.fit_transform(vectors)
        r_2d = np.hypot(coords_2d[:,0], coords_2d[:,1])
        theta = np.arctan2(coords_2d[:,1], coords_2d[:,0])
        
        embedding_df = pd.DataFrame({
            "GO_ID": present_ids,
            "hyperbolic_r": hyperbolic_r,
            "r_2d": r_2d,
            "theta": theta
        })
        
        structural_df = structural_df.merge(embedding_df, on="GO_ID", how="left")

# ==========================================================
# EXPORT FINAL INTEGRATED CSV
# ==========================================================

output_path = os.path.join(
    OUTPUT_DIR,
    f"GO_integrated_structure_embedding_{ROOT_ID.replace(':','')}_k_{k}.csv"
)

try:
    structural_df.to_csv(output_path, index=False, mode="x")
    print("\n✅ Integrated CSV exported to:")
except FileExistsError:
    print("File already exists, not written.")

print(output_path)
