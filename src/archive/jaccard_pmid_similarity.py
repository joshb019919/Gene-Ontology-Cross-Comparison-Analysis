from collections import defaultdict
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import obonet
import pandas as pd
import seaborn as sns
from sklearn.metrics import pairwise_distances

DATA = "./data/"
RESULTS = "./results/"
FIGURES = "./figures/"

target_terms = [
"GO:0043231","GO:0070074","GO:0044311","GO:0020009","GO:0097693",
"GO:0033099","GO:0160045","GO:0061468","GO:1990413","GO:0016007",
"GO:0160208","GO:0160201","GO:0140494","GO:0031094","GO:0110143",
"GO:0010168","GO:0005634","GO:0097708","GO:1990462","GO:0033009",
"GO:0005801","GO:0071212","GO:0009536","GO:0005739","GO:0005773",
"GO:0042566","GO:0032047","GO:0042735","GO:0020022","GO:0005793",
"GO:0044222","GO:0044223","GO:0044227","GO:0042579","GO:0070088",
"GO:0005783","GO:0005794"
]

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

# Find only from those 37 ids
filtered = gaf[gaf["GO_ID"].isin(target_terms)]
s = (filtered.assign(DB_ref_split=filtered["DB_Reference"]
                     .str.split("|")).explode("DB_ref_split"))
s_pm = s[s["DB_ref_split"].str.startswith("PMID:", na=False)]
gaf_pm = filtered.loc[filtered.index.isin(s_pm.index)]
# print(gaf_pm)
go_to_pmids = s_pm.groupby("GO_ID")["DB_ref_split"].apply(
    lambda x: {p.replace("PMID:", "") for p in x}
)
# print(go_to_pmids)

# Compute binary membership matrix
go_to_pmids = go_to_pmids.to_dict()
terms = list(go_to_pmids.keys())
all_genes = sorted(set().union(*go_to_pmids.values()))
binary = pd.DataFrame(0, index=terms, columns=all_genes)
for term, genes in go_to_pmids.items():
    binary.loc[term, list(genes)] = 1
# print(binary.shape)
# # print(binary)
# # print(binary.index)

df = pd.read_csv(DATA + "go_id_organelle_map.csv")  # header row becomes df.columns
d = [df[df[col].isin(binary.index)]["organelle"].to_list() for col in df.columns]
# # print(d)

# # Compute similarity matrix (similarity matrix = 1 - Jaccard distance)
X = binary.to_numpy()
X_bin = (X > 0).astype(bool)
jaccard_matrix_rounded = 1 - pairwise_distances(X_bin, metric="jaccard")
jaccard_matrix = 1 - pairwise_distances(X_bin, metric="jaccard")
# print(jaccard_matrix.shape)
# # print(jaccard_matrix)

jaccard_df = pd.DataFrame(jaccard_matrix, index=d[0], columns=d[0])
jaccard_df_rounded = pd.DataFrame(jaccard_matrix_rounded, index=d[0], columns=d[0]).round(4)
# print(jaccard_df)

# Send to CSV
jaccard_df.to_csv(RESULTS + "pmid_jaccard_matrix.csv", mode="w")
jaccard_df_rounded.to_csv(RESULTS + "pmid_jaccard_matrix_rounded.csv", mode="w")

# Generate figures
fig = plt.figure(figsize=(10,8))
fig.tight_layout()

sns.heatmap(
    jaccard_df,
    cmap="RdYlBu_r"
)

plt.title("Jaccard GO PMID Heatmap")
fig.savefig(FIGURES + "jaccard_go_pmid_heatmap.png", dpi=1200, bbox_inches="tight")
g = sns.clustermap(jaccard_df, cmap="RdYlBu_r", figsize=(10,10))
g.figure.suptitle("Jaccard GO PMID Clustermap", y=1.02)
g.savefig(FIGURES + "jaccard_go_pmid_clustermap.png", dpi=1200, bbox_inches="tight")
plt.close(g.figure)