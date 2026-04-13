from scipy.cluster.hierarchy import dendrogram, ward
import matplotlib.pyplot as plt
import numpy as np

from archive.jaccard_ann_similarity import jaccard_df

df_matrix = jaccard_df.to_numpy()
organelle_labels = jaccard_df.index.to_list()

linkage_matrix = ward(df_matrix)
plt.figure(figsize=(10,7))
ddata = dendrogram(
	linkage_matrix,
	labels=organelle_labels,
	leaf_rotation=90,
	leaf_font_size=8,
)

# Label each merge (link) with its Ward distance value.
for x_coords, y_coords in zip(ddata["icoord"], ddata["dcoord"]):
	x = 0.5 * (x_coords[1] + x_coords[2])
	y = y_coords[1]
	plt.text(
		x,
		y,
		f"{y:.3f}",
		va="bottom",
		ha="center",
		fontsize=8,
		color="black"
	)

plt.title("Ward Method Dendrogram")
plt.ylabel("Distance")
plt.tight_layout()
plt.show()
