# ------------------- #
# Silhouette Analysis #
# ------------------- #
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

from ..gene_clustering import cartesian

# Use the cartesian coordinates from polar plot data
X = cartesian

range_n_clusters = [2, 3, 4, 5, 6]

for n_clusters in range_n_clusters:
	# Create subplot with 1 row, 2 cols
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18,7)

	# Silhouette plot first
    ax1.set_xlim([-0.3,1])  # Set to entire possible range to see
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])  # Demarcation

    # Compute Ward clusters for this n_clusters value.
    Z = linkage(X, method='ward')
    cluster_labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    cluster_labels_zero = cluster_labels - 1

	# Average value for all samples
    silhouette_avg = silhouette_score(X, cluster_labels)
    print(
		"For n_clusters =",
		n_clusters,
		"The average silhouette score is :",
		silhouette_avg
	)

	# Score for each sample
    sample_silhouette_values = np.asarray(silhouette_samples(X, cluster_labels), dtype=float)
    cmap = plt.get_cmap("nipy_spectral")

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = np.sort(sample_silhouette_values[cluster_labels_zero == i])

        size_cluster_i = ith_cluster_silhouette_values.size
        if size_cluster_i == 0:
			# Keep spacing consistent and show that this cluster has no assigned points.
            ax1.text(0.02, y_lower + 2, f"cluster {i}: empty", fontsize=7, color="gray")
            y_lower = y_lower + 10
            continue

        y_upper = y_lower + size_cluster_i

        color = cmap(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        if size_cluster_i == 1 and np.isclose(ith_cluster_silhouette_values[0], 0.0):
			# A singleton has silhouette 0 by definition; add an explicit marker/label.
            y_mid = y_lower + 0.5 * size_cluster_i
            ax1.scatter([0.0], [y_mid], color=color, s=16, zorder=3)
            ax1.text(0.02, y_mid, "singleton (s=0)", fontsize=7, va="center", color=color)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    denom = max(n_clusters - 1, 1)
    colors = cmap(cluster_labels_zero.astype(float) / denom)
    ax2.scatter(
		X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
    )

    # Label cluster centers using per-cluster means (Ward has no cluster_centers_ attribute).
    centers = []
    center_ids = []
    for i in range(n_clusters):
        mask = cluster_labels_zero == i
        if np.any(mask):
            centers.append(X[mask].mean(axis=0))
            center_ids.append(i)

    if centers:
        centers = np.asarray(centers)
        ax2.scatter(
            centers[:, 0],
            centers[:, 1],
            marker="o",
            c="white",
            alpha=1,
            s=200,
            edgecolor="k",
        )

        for idx, c in zip(center_ids, centers):
            ax2.scatter(c[0], c[1], marker="$%d$" % idx, alpha=1, s=50, edgecolor="k")

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Cartesian x (r cos theta)")
    ax2.set_ylabel("Cartesian y (r sin theta)")

    plt.suptitle(
		"Silhouette analysis for Ward linkage on Cartesian organelle coordinates with n_clusters = %d"
        % n_clusters,
        fontsize=14,
        fontweight="bold",
    )

plt.show()
