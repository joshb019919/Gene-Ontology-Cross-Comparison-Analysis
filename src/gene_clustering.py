import numpy as np
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Circle
import pandas as pd


def add_leader_labels(ax, x_vals, y_vals, labels, base_offset, cluster_labels=None, ring_geometry=None):
	"""Draw leader lines and labels with automatic collision detection and resolution."""
	fontsize = 5.5
	char_width = 0.6 * fontsize
	char_height = 1.2 * fontsize

	# Step 1: Compute initial label positions and directions.
	label_data = []
	for i, (x_pt, y_pt, label) in enumerate(zip(x_vals, y_vals, labels)):
		u_x, u_y = 1.0, 0.0
		if cluster_labels is not None and ring_geometry is not None:
			cluster_id = cluster_labels[i]
			geom = ring_geometry.get(cluster_id)
			if geom is not None:
				centroid = geom["centroid"]
				v_x = x_pt - centroid[0]
				v_y = y_pt - centroid[1]
				v_norm = np.hypot(v_x, v_y)
				if v_norm > 0:
					u_x = v_x / v_norm
					u_y = v_y / v_norm

		offset = base_offset * (1.0 + 0.18 * (i % 3))
		lx = x_pt + offset * u_x
		ly = y_pt + offset * u_y

		label_str = str(label)
		label_data.append({
			"index": i,
			"x_pt": x_pt,
			"y_pt": y_pt,
			"label": label_str,
			"u_x": u_x,
			"u_y": u_y,
			"offset": offset,
			"lx": lx,
			"ly": ly,
		})

	# Step 2: Estimate label bounding boxes and detect collisions.
	def bbox_from_label(lx, ly, label_str):
		"""Estimate bbox as (left, top, right, bottom) centered at (lx, ly)."""
		width = len(label_str) * char_width / 2.0
		height = char_height / 2.0
		return (lx - width, ly - height, lx + width, ly + height)

	def bboxes_overlap(bbox1, bbox2, margin=0.1):
		"""Check if two bboxes overlap (with small margin for safety)."""
		l1, t1, r1, b1 = bbox1
		l2, t2, r2, b2 = bbox2
		return not (r1 + margin < l2 or r2 + margin < l1 or b1 + margin < t2 or b2 + margin < t1)

	# Detect collisions and apply fixes.
	for i in range(len(label_data)):
		bbox_i = bbox_from_label(label_data[i]["lx"], label_data[i]["ly"], label_data[i]["label"])
		collides = False
		for j in range(i + 1, len(label_data)):
			bbox_j = bbox_from_label(label_data[j]["lx"], label_data[j]["ly"], label_data[j]["label"])
			if bboxes_overlap(bbox_i, bbox_j):
				collides = True
				# Rotate label i by a small angle to separate it.
				rotation = np.pi / 6.0  # 30 degrees
				u_x, u_y = label_data[i]["u_x"], label_data[i]["u_y"]
				cos_r, sin_r = np.cos(rotation), np.sin(rotation)
				u_x_rot = u_x * cos_r - u_y * sin_r
				u_y_rot = u_x * sin_r + u_y * cos_r
				label_data[i]["u_x"] = u_x_rot
				label_data[i]["u_y"] = u_y_rot
				label_data[i]["offset"] *= 1.15  # Slightly extend offset.
				label_data[i]["lx"] = label_data[i]["x_pt"] + label_data[i]["offset"] * u_x_rot
				label_data[i]["ly"] = label_data[i]["y_pt"] + label_data[i]["offset"] * u_y_rot
				break

	# Step 3: Draw all leader lines and labels.
	for item in label_data:
		lx = item["lx"]
		ly = item["ly"]
		u_x = item["u_x"]
		u_y = item["u_y"]
		x_pt = item["x_pt"]
		y_pt = item["y_pt"]
		label_str = item["label"]

		ax.plot([x_pt, lx], [y_pt, ly], color="black", linewidth=0.5, alpha=0.8, clip_on=False)
		ha = "left" if u_x > 0.15 else ("right" if u_x < -0.15 else "center")
		va = "bottom" if u_y > 0.2 else ("top" if u_y < -0.2 else "center")
		ax.text(lx, ly, label_str, fontsize=fontsize, ha=ha, va=va, clip_on=False)


def draw_cluster_rings(ax, x_vals, y_vals, cluster_labels, cmap="viridis", expansion=1.1, linewidth=1.8):
	"""Draw a light ring per cluster at expansion * farthest-point radius from centroid."""

	def compute_cluster_ring_geometry(x_vals, y_vals, cluster_labels, expansion=1.1):
		"""Return centroid/radius for each cluster using an expanded farthest-point radius."""
		xy = np.column_stack([x_vals, y_vals])
		geometry = {}
		for cluster_id in np.unique(cluster_labels):
			cluster_points = xy[cluster_labels == cluster_id]
			if cluster_points.shape[0] == 0:
				continue
			centroid = cluster_points.mean(axis=0)
			radius = expansion * np.linalg.norm(cluster_points - centroid, axis=1).max()
			if radius > 0:
				geometry[cluster_id] = {"centroid": centroid, "radius": radius}
		return geometry

	ring_geometry = compute_cluster_ring_geometry(x_vals, y_vals, cluster_labels, expansion=expansion)
	unique_clusters = np.unique(cluster_labels)
	norm = Normalize(vmin=np.min(unique_clusters), vmax=np.max(unique_clusters))
	color_map = plt.get_cmap(cmap)

	for cluster_id in unique_clusters:
		geom = ring_geometry.get(cluster_id)
		if geom is None:
			continue
		centroid = geom["centroid"]
		radius = geom["radius"]

		ring_rgb = np.array(color_map(norm(cluster_id) % 1.0)[:3])
		ring = Circle(
			(centroid[0], centroid[1]),
			radius,
			fill=False,
			edgecolor=ring_rgb,
			linewidth=linewidth,
			alpha=0.88,
			zorder=2,
		)
		ax.add_patch(ring)

	return ring_geometry


def plot_and_save(ax, x_vals, y_vals, cluster_labels, title, filename, draw_rings=True):
	"""Create, plot, and save a single clustering visualization."""
	ax.scatter(x_vals, y_vals, c=cluster_labels, cmap='tab10', s=10)
	ring_geometry = None
	if draw_rings:
		ring_geometry = draw_cluster_rings(ax, x_vals, y_vals, cluster_labels, cmap="tab10", expansion=1.1, linewidth=1.8)
	add_leader_labels(
		ax,
		x_vals,
		y_vals,
		labels,
		label_offset,
		cluster_labels=cluster_labels,
		ring_geometry=ring_geometry,
	)
	ax.set_title(title)
	ax.set_xlabel('x (r cos theta)')
	ax.set_ylabel('y (r sin theta)')

	plt.savefig(filename, dpi=300, bbox_inches='tight')


def save_cluster_assignments(organelle_names, cluster_labels, clustering_type, params):
	"""Save organelle-to-cluster assignments as CSV.
	
	Args:
		organelle_names: Series/array of organelle names.
		cluster_labels: Array of cluster assignments (one per organelle).
		clustering_type: String like "kmeans" or "wards_hierarchical" or "wards_distance".
		params: String like "6" or "5.0" for the parameter (n_clusters or distance threshold).
	"""
	df = pd.DataFrame({
		"organelle": organelle_names,
		"cluster": cluster_labels,
	})
	filepath = f"results/{clustering_type}_{params}.csv"
	df.to_csv(filepath, index=False)


# -------------------- #
# Convert to Cartesian #
# -------------------- #
# Polar coords
points_df = pd.read_csv("results/polar_plot_points.csv")
points_df_no_parent = points_df.iloc[1:-3].copy()
# print(points_df_no_parent)

# Separate r and theta and get organelle names for later labeling_no_parent
r = pd.to_numeric(points_df_no_parent["r"])
theta = pd.to_numeric(points_df_no_parent["theta"])  # in radians
labels = points_df_no_parent["organelle"]
organelle_names = points_df_no_parent["organelle"].astype(str)
# print(r)

# Convert to Cartesian coordinates
x = r * np.cos(theta)
y = r * np.sin(theta)
cartesian = np.column_stack([x, y])
cartesian = np.nan_to_num(cartesian, nan=0.0, posinf=0.0, neginf=0.0)
max_radius = max(float(np.nanmax(np.hypot(x, y))), 1.0)
label_offset = 0.18 * max_radius
# print(cartesian)

# ------------------ #
# K-Means Clustering #
# ------------------ #
kmeans_6 = KMeans(n_clusters=6, random_state=10)
kmeans_labels_6 = kmeans_6.fit_predict(cartesian)
kmeans_5 = KMeans(n_clusters=5, random_state=10)
kmeans_labels_5 = kmeans_5.fit_predict(cartesian)
kmeans_4 = KMeans(n_clusters=4, random_state=10)
kmeans_labels_4 = kmeans_4.fit_predict(cartesian)
kmeans_3 = KMeans(n_clusters=3, random_state=10)
kmeans_labels_3 = kmeans_3.fit_predict(cartesian)
kmeans_2 = KMeans(n_clusters=2, random_state=10)
kmeans_labels_2 = kmeans_2.fit_predict(cartesian)

# ------------------------- #
# Ward's Linkage Clustering #
# ------------------------- #
Z = linkage(cartesian, method='ward')
dendrogram(Z)
plt.title("Ward's Hierarchical Clustering Dendrogram")
plt.xlabel("Sample Index or (Cluster size)")
plt.ylabel("Distance")

# Cut the dendrogram to get n clusters
ward_labels_6 = fcluster(Z, t=6, criterion="maxclust")
ward_labels_5 = fcluster(Z, t=5, criterion="maxclust")
ward_labels_4 = fcluster(Z, t=4, criterion="maxclust")
ward_labels_3 = fcluster(Z, t=3, criterion="maxclust")
ward_labels_2 = fcluster(Z, t=2, criterion="maxclust")

# Cut at distance threshold
ward_labels_dist_5 = fcluster(Z, t=5.0, criterion="distance")
ward_labels_dist_3 = fcluster(Z, t=3.0, criterion="distance")

# ----- #
# Plots #
# ----- #

# K-Means clustering
fig, ax = plt.subplots(figsize=(10, 8))
plot_and_save(ax, x, y, kmeans_labels_6, 'K-Means (k=6)', 'figures/kmeans_6.png')
plt.tight_layout()
plt.close()

fig, ax = plt.subplots(figsize=(10, 8))
plot_and_save(ax, x, y, kmeans_labels_5, 'K-Means (k=5)', 'figures/kmeans_5.png')
plt.tight_layout()
plt.close()

fig, ax = plt.subplots(figsize=(10, 8))
plot_and_save(ax, x, y, kmeans_labels_4, 'K-Means (k=4)', 'figures/kmeans_4.png')
plt.tight_layout()
plt.close()

fig, ax = plt.subplots(figsize=(10, 8))
plot_and_save(ax, x, y, kmeans_labels_3, 'K-Means (k=3)', 'figures/kmeans_3.png')
plt.tight_layout()
plt.close()

fig, ax = plt.subplots(figsize=(10, 8))
plot_and_save(ax, x, y, kmeans_labels_2, 'K-Means (k=2)', 'figures/kmeans_2.png')
plt.tight_layout()
plt.close()

# Ward's Hierarchical clustering (maxclust)
fig, ax = plt.subplots(figsize=(10, 8))
plot_and_save(ax, x, y, ward_labels_6, "Ward's Hierarchical (k=6)", 'figures/wards_hierarchical_6.png')
plt.tight_layout()
plt.close()

fig, ax = plt.subplots(figsize=(10, 8))
plot_and_save(ax, x, y, ward_labels_5, "Ward's Hierarchical (k=5)", 'figures/wards_hierarchical_5.png')
plt.tight_layout()
plt.close()

fig, ax = plt.subplots(figsize=(10, 8))
plot_and_save(ax, x, y, ward_labels_4, "Ward's Hierarchical (k=4)", 'figures/wards_hierarchical_4.png')
plt.tight_layout()
plt.close()

fig, ax = plt.subplots(figsize=(10, 8))
plot_and_save(ax, x, y, ward_labels_3, "Ward's Hierarchical (k=3)", 'figures/wards_hierarchical_3.png')
plt.tight_layout()
plt.close()

fig, ax = plt.subplots(figsize=(10, 8))
plot_and_save(ax, x, y, ward_labels_2, "Ward's Hierarchical (k=2)", 'figures/wards_hierarchical_2.png')
plt.tight_layout()
plt.close()

# Ward's Hierarchical clustering (distance threshold)
fig, ax = plt.subplots(figsize=(10, 8))
plot_and_save(ax, x, y, ward_labels_dist_5, "Ward's Hierarchical (distance=5.0)", 'figures/wards_distance_5.0.png')
plt.tight_layout()
plt.close()

fig, ax = plt.subplots(figsize=(10, 8))
plot_and_save(ax, x, y, ward_labels_dist_3, "Ward's Hierarchical (distance=3.0)", 'figures/wards_distance_3.0.png')
plt.tight_layout()
plt.close()

# Original data (no clustering)
fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(x, y, c='gray', s=10)
add_leader_labels(ax, x, y, labels, label_offset)
ax.set_title('Original Data')
ax.set_xlabel('x (r cos theta)')
ax.set_ylabel('y (r sin theta)')
plt.tight_layout()
plt.savefig('figures/original_data.png', dpi=600, bbox_inches='tight')
plt.close()

# ---------- #
# Save Files #
# ---------- #
# K-Means clustering results
save_cluster_assignments(organelle_names, kmeans_labels_6, "kmeans", "6")
save_cluster_assignments(organelle_names, kmeans_labels_5, "kmeans", "5")
save_cluster_assignments(organelle_names, kmeans_labels_4, "kmeans", "4")
save_cluster_assignments(organelle_names, kmeans_labels_3, "kmeans", "3")
save_cluster_assignments(organelle_names, kmeans_labels_2, "kmeans", "2")

# Ward's Hierarchical clustering (maxclust) results
save_cluster_assignments(organelle_names, ward_labels_6, "wards_hierarchical", "6")
save_cluster_assignments(organelle_names, ward_labels_5, "wards_hierarchical", "5")
save_cluster_assignments(organelle_names, ward_labels_4, "wards_hierarchical", "4")
save_cluster_assignments(organelle_names, ward_labels_3, "wards_hierarchical", "3")
save_cluster_assignments(organelle_names, ward_labels_2, "wards_hierarchical", "2")

# Ward's Hierarchical clustering (distance threshold) results
save_cluster_assignments(organelle_names, ward_labels_dist_5, "wards_distance", "5.0")
save_cluster_assignments(organelle_names, ward_labels_dist_3, "wards_distance", "3.0")
