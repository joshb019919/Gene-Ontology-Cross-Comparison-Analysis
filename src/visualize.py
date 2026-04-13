import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load table of GO terms
df = pd.read_csv("results/polar_plot_points.csv")
df = df[df["organelle"] != "intracellular membrane-bounded organelles"]

# Exclude rows without a valid organelle label from all plots.
valid_organelle = df["organelle"].notna() & df["organelle"].astype(str).str.strip().ne("")
df = df.loc[valid_organelle].copy()

theta = df["theta"].values
r = df["r"].values
labels = df["organelle"].astype(str).values

##### Polar Scatter Plot #####
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, projection="polar")
ax.scatter(theta, r, s=10)
ax.set_title("Hyperbolic Embedding Polar Plot")
ax.set_ylim(0, 12.0)

# Annotate labels around local cluster(s)
max_r = np.nanmax(r) if len(r) > 0 else 1.0
if len(theta) > 0:
    ths = np.mod(theta, 2 * np.pi)
    order = np.argsort(ths)
    ths = ths[order]
    rs = r[order]
    labs = labels[order]

    # Build local clusters in angle/radius space.
    clusters = []
    curr = [0]
    ang_thresh = 0.10
    r_thresh = 0.10 * max_r
    for i in range(1, len(ths)):
        if abs(ths[i] - ths[i - 1]) <= ang_thresh and abs(rs[i] - rs[i - 1]) <= r_thresh:
            curr.append(i)
        else:
            clusters.append(curr)
            curr = [i]
    clusters.append(curr)

    label_records = []

    for g in clusters:
        n = len(g)
        g_th = ths[g]
        g_r = rs[g]
        g_lab = labs[g]

        # Cluster center in Cartesian coordinates.
        g_x = g_r * np.cos(g_th)
        g_y = g_r * np.sin(g_th)
        c_x = np.mean(g_x)
        c_y = np.mean(g_y)
        c_th = np.mod(np.arctan2(c_y, c_x), 2 * np.pi)
        c_r = float(np.hypot(c_x, c_y))

        if n == 1:
            # Single point: short offset from the point in its local outward direction.
            th_pt = g_th[0]
            r_pt = g_r[0]
            p_x = r_pt * np.cos(th_pt)
            p_y = r_pt * np.sin(th_pt)
            v_x = p_x - c_x
            v_y = p_y - c_y
            v_norm = np.hypot(v_x, v_y)
            if v_norm < 1e-9:
                v_x, v_y = np.cos(th_pt), np.sin(th_pt)
            else:
                v_x, v_y = v_x / v_norm, v_y / v_norm
            offset = 0.08 * max_r
            l_x = p_x + offset * v_x
            l_y = p_y + offset * v_y
            th_lbl = np.mod(np.arctan2(l_y, l_x), 2 * np.pi)
            r_lbl = float(np.hypot(l_x, l_y))
            label_records.append((th_pt, r_pt, th_lbl, r_lbl, g_lab[0], c_x, c_y))
            continue

        # Multi-point cluster: spread labels all around a local ring centered on cluster.
        ring_radius = (0.10 + 0.03 * min(n, 8)) * max_r
        slot_angles = c_th + np.linspace(0, 2 * np.pi, n, endpoint=False)

        # Match points to ring slots by local angle around the cluster center.
        local_pt_angles = np.mod(np.arctan2(g_y - c_y, g_x - c_x), 2 * np.pi)
        idx_local = np.argsort(local_pt_angles)
        idx_slots = np.argsort(np.mod(slot_angles, 2 * np.pi))

        for j in range(n):
            idx = idx_local[j]
            slot = slot_angles[idx_slots[j]]
            th_pt = g_th[idx]
            r_pt = g_r[idx]
            lbl = g_lab[idx]

            # Place label around cluster center (not around global polar frame).
            l_x = c_x + ring_radius * np.cos(slot)
            l_y = c_y + ring_radius * np.sin(slot)
            th_lbl = np.mod(np.arctan2(l_y, l_x), 2 * np.pi)
            r_lbl = float(np.hypot(l_x, l_y))
            label_records.append((th_pt, r_pt, th_lbl, r_lbl, lbl, c_x, c_y))

    # Collision-avoidance pass: push labels outward from each cluster center when too tight.
    if label_records:
        label_records.sort(key=lambda x: x[2])
        min_ang_sep = 0.07
        min_rad_sep = 0.06 * max_r
        push_step = 0.04 * max_r

        adjusted = []
        for rec in label_records:
            th_pt, r_pt, th_lbl, r_lbl, lbl, c_x, c_y = rec
            tries = 0
            while tries < 30:
                conflict = False
                for _, _, th2, r2, _, _, _ in adjusted:
                    dth = abs(th_lbl - th2)
                    dth = min(dth, 2 * np.pi - dth)
                    if dth < min_ang_sep and abs(r_lbl - r2) < min_rad_sep:
                        conflict = True
                        break
                if not conflict:
                    break

                # Push away from this cluster center (preserves around-cluster look).
                l_x = r_lbl * np.cos(th_lbl)
                l_y = r_lbl * np.sin(th_lbl)
                v_x = l_x - c_x
                v_y = l_y - c_y
                v_norm = np.hypot(v_x, v_y)
                if v_norm < 1e-9:
                    v_x, v_y = np.cos(th_lbl), np.sin(th_lbl)
                else:
                    v_x, v_y = v_x / v_norm, v_y / v_norm
                l_x += push_step * v_x
                l_y += push_step * v_y
                th_lbl = np.mod(np.arctan2(l_y, l_x), 2 * np.pi)
                r_lbl = float(np.hypot(l_x, l_y))
                tries += 1
            adjusted.append((th_pt, r_pt, th_lbl, r_lbl, lbl, c_x, c_y))

        stretched = []
        # Increase current leader length by 25% (1.5 -> 1.875).
        stretch_factor = 1.25
        for th_pt, r_pt, th_lbl, r_lbl, lbl, c_x, c_y in adjusted:
            p_x = r_pt * np.cos(th_pt)
            p_y = r_pt * np.sin(th_pt)
            l_x = r_lbl * np.cos(th_lbl)
            l_y = r_lbl * np.sin(th_lbl)

            # Make point-to-label segment approximately 1.5x longer.
            s_x = p_x + stretch_factor * (l_x - p_x)
            s_y = p_y + stretch_factor * (l_y - p_y)

            # Keep selected labels on the right side of their local cluster.
            key = str(lbl).lower()
            if key in {"plastid", "omegaso", "vacuole", "methane"}:
                y_offsets = {
                    "plastid": -0.06 * max_r,
                    "omegaso": -0.02 * max_r,
                    "vacuole": 0.02 * max_r,
                    "methane": 0.06 * max_r,
                }
                s_x = max(s_x, c_x + 0.14 * max_r)
                s_y = c_y + y_offsets.get(key, 0.0)

            s_th = np.mod(np.arctan2(s_y, s_x), 2 * np.pi)
            s_r = float(np.hypot(s_x, s_y))
            stretched.append((th_pt, r_pt, s_th, s_r, lbl, c_x, c_y))

        for th_pt, r_pt, th_lbl, r_lbl, lbl, _, _ in stretched:
            ax.plot([th_pt, th_lbl], [r_pt, r_lbl], color="black", linewidth=0.6, alpha=0.8, clip_on=False)
            key = str(lbl).lower()
            if key in {"plastid", "omegaso", "vacuole", "methane"}:
                ha = "left"
            else:
                cos_v = np.cos(th_lbl)
                if cos_v > 0.15:
                    ha = "left"
                elif cos_v < -0.15:
                    ha = "right"
                else:
                    ha = "center"
            ax.text(th_lbl, r_lbl, lbl, fontsize=5.5, ha=ha, va="center", clip_on=False)

# Save polar image
img_path = "results/polar_plot.png"
out_csv = "results/polar_plot_points.csv"
fig.savefig(img_path, dpi=300, bbox_inches="tight")
plt.close(fig)

##### Distance From Mean #####
mean_r = np.mean(r)
mean_theta = np.mean(theta)

df["delta_r"] = r - mean_r
df["delta_theta"] = theta - mean_theta

# Spread x positions so adjacent tick labels have visible separation.
x = np.arange(len(labels), dtype=float) * 1.35

fig, ax = plt.subplots(1,2, figsize=(24, 8), dpi=150)
fig.suptitle("GO Term Deviations from Mean Embedding Position", fontsize=16, y=0.98)
fig.text(0.5, 0.935, "Parent: intracellular membrane-bounded organelles", ha="center", va="center", fontsize=11)

ax[0].bar(x, df["delta_r"], width=0.80)
ax[0].set_xticks(x, labels)
ax[0].set_title("Deviation from Mean Radius")
ax[0].tick_params(axis="x", labelrotation=45, labelsize=7)
for tick in ax[0].get_xticklabels():
    tick.set_horizontalalignment("right")
    tick.set_rotation_mode("anchor")

if len(x) > 0:
    ax[0].set_xlim(x[0] - 0.8, x[-1] + 0.8)

ax[1].bar(x, df["delta_theta"], width=0.80)
ax[1].set_xticks(x, labels)
ax[1].set_title("Deviation from Mean Angle")
ax[1].tick_params(axis="x", labelrotation=45, labelsize=7)
for tick in ax[1].get_xticklabels():
    tick.set_horizontalalignment("right")
    tick.set_rotation_mode("anchor")

if len(x) > 0:
    ax[1].set_xlim(x[0] - 0.8, x[-1] + 0.8)

plt.subplots_adjust(top=0.88, bottom=0.36, wspace=0.39)
plt.show()
