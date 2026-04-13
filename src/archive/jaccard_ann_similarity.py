from collections import defaultdict
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import pairwise_distances

DATA = "./data/"
RESULTS = "./results/"
FIGURES = "./figures/"

gaf_cols = [
"DB","DB_Object_ID","Gene","Qualifier","GO_ID","DB_Reference",
"Evidence","With","Aspect","Name","Synonym","Type","Taxon",
"Date","Assigned_By","Extension","Gene_Product_Form_ID"
]

gaf = pd.read_csv(
    DATA + "goa_human.gaf",
    sep="\t",
    comment="!",
    header=None,
    dtype=str,
    names=gaf_cols,
    na_filter=False
)

target_terms = [
    "GO:0020022", "GO:0044222", "GO:0033099", "GO:0005801", "GO:0005783", 
    "GO:0005793", "GO:0042735", "GO:0010168", "GO:0044311", "GO:1990413", 
    "GO:0005794", "GO:0042566", "GO:0097708", "GO:0061468", "GO:0044227",
    "GO:0042579", "GO:0020009", "GO:0016007", "GO:0005739", "GO:0032047", 
    "GO:0070074", "GO:0033009", "GO:0005634", "GO:0097693", "GO:1990462", 
    "GO:0044223", "GO:0009536", "GO:0031094", "GO:0070088", "GO:0071212", 
    "GO:0005773"
]

# Prepare dict of ids with sets of genes
filtered = gaf[gaf["GO_ID"].isin(target_terms)]
print(filtered[filtered["Gene"] == ""])
go_to_genes = filtered.groupby("GO_ID")["Gene"].agg(set)
# print(go_to_genes.truncate(None))

# Confirm total terms with gene annotations
# print(len(go_to_genes))
# print(go_to_genes.keys())

# Confirm terms with no gene annotations
missing = set(target_terms) - set(go_to_genes.keys())
# print(missing)
# print(len(missing))

# Cross confirm that a GAF entry has no gene annotations
# print(gaf[gaf["Gene"] == "GO:0009536"]["Gene"])  # Empty series

# Compute binary membership matrix
terms = list(go_to_genes.keys())
all_genes = sorted(set().union(*go_to_genes.values))
binary = pd.DataFrame(0, index=terms, columns=all_genes)
for term, genes in go_to_genes.items():
    binary.loc[term, list(genes)] = 1
# print(binary.shape)  # 13 x 10069
# print(binary)        # binary.shape dataframe
# print(binary.index)  # Same as go_to_genes

df = pd.read_csv(DATA + "go_id_organelle_map.csv")  # header row becomes df.columns
d = [df[df[col].isin(binary.index)]["organelle"].to_list() for col in df.columns]
# print(d)

# # Compute similarity matrix (similarity matrix = 1 - Jaccard distance)
X = binary.to_numpy()         # Convert for below distancing
X_bin = (X > 0).astype(bool)  # Fixes autoconvert warning (nothing actually lost)
jaccard_matrix_rounded = 1 - pairwise_distances(X_bin, metric="jaccard")
jaccard_matrix = 1 - pairwise_distances(X_bin, metric="jaccard")
# print(jaccard_matrix.shape)
# print(jaccard_matrix)

jaccard_df = pd.DataFrame(jaccard_matrix, index=d[0], columns=d[0])
jaccard_df_rounded = pd.DataFrame(jaccard_matrix_rounded, index=d[0], columns=d[0]).round(4)
# print(jaccard_df)

# jaccard_df.to_csv(RESULTS + "gene_jaccard_matrix.csv", mode="w")
# jaccard_df_rounded.to_csv(RESULTS + "gene_jaccard_matrix_rounded.csv", mode="w")

# fig = plt.figure(figsize=(10,8))
# fig.tight_layout()

# sns.heatmap(
#     jaccard_df,
#     cmap="RdYlBu_r",
#     vmin=0,
#     vmax=1,
#     center=.5
# )

# plt.title("Jaccard GO Annotation Heatmap")
# fig.savefig(FIGURES + "jaccard_go_ann_heatmap.png", dpi=1200, bbox_inches="tight")
# g = sns.clustermap(jaccard_df, cmap="RdYlBu_r", figsize=(10,10), 
#                    vmin=0, vmax=1, center=0.5, cbar_kws={"shrink": 0.65, "aspect": 10})
# g.figure.suptitle("Jaccard GO Annotation Clustermap", y=1.02)
# g.savefig(FIGURES + "jaccard_go_ann_clustermap.png", dpi=1200, bbox_inches="tight")
# plt.close(g.figure)
