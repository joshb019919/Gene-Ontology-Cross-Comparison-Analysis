from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.patches import Circle
from matplotlib.transforms import Bbox
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.cluster import KMeans


# ========================= #
# Shared plotting defaults  #
# ========================= #

@dataclass
class ClusterRingConfig:
    enabled: bool = True
    cmap: str = "tab10"
    expansion: float = 1.10
    linewidth: float = 1.8
    alpha: float = 0.88
    zorder: int = 2
    samples_per_ring: int = 361


@dataclass
class LabelStyle:
    fontsize: float = 6.0
    fontweight: str = "bold"
    color: str = "black"
    clip_on: bool = False
    zorder: int = 6
    bbox: Optional[dict] = field(
        default_factory=lambda: {
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.95,
            "pad": 0.15,
        }
    )


@dataclass
class LeaderLineStyle:
    color: str = "black"
    linewidth: float = 0.5
    alpha: float = 0.80
    clip_on: bool = False
    zorder: int = 4


@dataclass
class ScatterStyle:
    size: float = 14.0
    alpha: float = 0.95
    zorder: int = 3
    edgecolors: str = "none"


@dataclass
class LabelLayoutConfig:
    base_offset: Optional[float] = None
    offset_as_fraction_of_radius: float = 0.18
    leader_length_scale: float = 1.5
    radial_step_fraction: float = 0.06
    angular_step_deg: float = 6.0
    max_angular_steps: int = 9
    max_radial_steps: int = 12
    bbox_margin_pts: float = 1.5
    line_start_pad_pts: float = 1.5
    line_end_pad_pts: float = 0.8
    cluster_fallback_to_global_center: bool = True
    draw_debug_boxes: bool = False


@dataclass
class PolarClusterConfig:
    angle_threshold: float = 0.10
    radius_threshold_fraction: float = 0.10
    sort_by_theta: bool = True
    radial_limit: float = 12.0
    theta_zero_location: Optional[str] = None
    theta_direction: int = 1


# ========================= #
# Shared annotation engine  #
# ========================= #

class ClusterAnnotationEngine:
    """Reusable engine for rings, labels, and leader lines in Cartesian space.

    Polar plots reuse this engine by converting polar points to Cartesian for all
    geometry calculations, then drawing the final results back in polar coords.
    """

    def __init__(
        self,
        ring_config: Optional[ClusterRingConfig] = None,
        label_style: Optional[LabelStyle] = None,
        leader_style: Optional[LeaderLineStyle] = None,
        layout: Optional[LabelLayoutConfig] = None,
    ) -> None:
        self.ring_config = ring_config or ClusterRingConfig()
        self.label_style = label_style or LabelStyle()
        self.leader_style = leader_style or LeaderLineStyle()
        self.layout = layout or LabelLayoutConfig()

    def build_cluster_colors(self, cluster_labels: np.ndarray, cmap_name: Optional[str] = None) -> dict:
        ids = np.unique(cluster_labels.astype(int)) if len(cluster_labels) else np.array([0], dtype=int)
        if len(ids) == 0:
            ids = np.array([0], dtype=int)
        vmin = float(ids.min())
        vmax = float(ids.max()) if len(ids) > 1 else float(ids.min() + 1)
        norm = Normalize(vmin=vmin, vmax=vmax)
        resolved_cmap = cmap_name if cmap_name is not None else (self.ring_config.cmap if self.ring_config is not None else "tab10")
        cmap = plt.get_cmap(resolved_cmap)
        colors = {int(cid): cmap(norm(int(cid))) for cid in ids}
        return {"norm": norm, "cmap": cmap, "colors": colors}

    def compute_cluster_centroid_geometry(
        self,
        x: np.ndarray,
        y: np.ndarray,
        cluster_labels: np.ndarray,
    ) -> Dict[int, dict]:
        xy = np.column_stack([x, y])
        geometry: Dict[int, dict] = {}
        for cluster_id in np.unique(cluster_labels.astype(int)):
            points = xy[cluster_labels == cluster_id]
            if len(points) == 0:
                continue
            geometry[int(cluster_id)] = {
                "centroid": points.mean(axis=0),
                "count": int(len(points)),
            }
        return geometry

    def compute_cluster_ring_geometry(
        self,
        x: np.ndarray,
        y: np.ndarray,
        cluster_labels: np.ndarray,
    ) -> Dict[int, dict]:
        xy = np.column_stack([x, y])
        geometry: Dict[int, dict] = {}
        expansion = self.ring_config.expansion if self.ring_config is not None else 1.0
        for cluster_id in np.unique(cluster_labels.astype(int)):
            points = xy[cluster_labels == cluster_id]
            if len(points) == 0:
                continue
            centroid = points.mean(axis=0)
            distances = np.linalg.norm(points - centroid, axis=1)
            radius = float(distances.max()) if len(distances) else 0.0
            geometry[int(cluster_id)] = {
                "centroid": centroid,
                "radius": max(radius * expansion, 1e-9),
                "count": int(len(points)),
            }
        return geometry

    def place_labels(
        self,
        ax: Axes,
        x: np.ndarray,
        y: np.ndarray,
        labels: Sequence[str],
        cluster_labels: np.ndarray,
        ring_geometry: Dict[int, dict],
    ) -> List[dict]:
        labels_arr = np.asarray(labels, dtype=object)
        fig = ax.figure
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        global_center = np.array([np.mean(x), np.mean(y)], dtype=float)
        max_radius = max(float(np.nanmax(np.hypot(x, y))), 1.0)
        base_offset = (
            self.layout.base_offset
            if self.layout.base_offset is not None
            else self.layout.offset_as_fraction_of_radius * max_radius
        )
        base_offset *= self.layout.leader_length_scale

        priorities = self._placement_priority(x, y, labels_arr, cluster_labels)
        order = np.argsort(priorities)[::-1]
        placements: List[Optional[dict]] = [None] * len(labels_arr)
        occupied: List[dict] = []

        angle_candidates = [0.0]
        for step in range(1, self.layout.max_angular_steps + 1):
            delta = np.deg2rad(self.layout.angular_step_deg * step)
            angle_candidates.extend([delta, -delta])

        for idx in order:
            cluster_id = int(cluster_labels[idx])
            point = np.array([x[idx], y[idx]], dtype=float)
            center = self._outward_center(cluster_id, ring_geometry, global_center)
            outward = self._unit_from_center(point, center)
            base_angle = float(np.arctan2(outward[1], outward[0]))

            best_noncolliding = None
            best_colliding = None
            best_colliding_score = np.inf

            for radial_step in range(self.layout.max_radial_steps + 1):
                offset = base_offset * (1.0 + radial_step * self.layout.radial_step_fraction)
                for angle_delta in angle_candidates:
                    theta = base_angle + angle_delta
                    direction = np.array([np.cos(theta), np.sin(theta)], dtype=float)
                    text_xy = point + direction * offset
                    text_meta = self._text_bbox_data(ax, text_xy, str(labels_arr[idx]), direction, renderer)
                    candidate = self._make_placement(idx, point, text_xy, direction, text_meta, str(labels_arr[idx]))
                    score = radial_step * 1000.0 + abs(angle_delta)

                    if self._collides(candidate["bbox"], occupied):
                        if score < best_colliding_score:
                            best_colliding_score = score
                            best_colliding = candidate
                        continue

                    best_noncolliding = candidate
                    break
                if best_noncolliding is not None:
                    break

            chosen = best_noncolliding if best_noncolliding is not None else best_colliding
            if chosen is None:
                fallback_xy = point + outward * base_offset
                text_meta = self._text_bbox_data(ax, fallback_xy, str(labels_arr[idx]), outward, renderer)
                chosen = self._make_placement(idx, point, fallback_xy, outward, text_meta, str(labels_arr[idx]))

            placements[idx] = chosen
            occupied.append(chosen)

        return [p for p in placements if p is not None]


    def place_labels_polar(
        self,
        ax: Axes,
        x: np.ndarray,
        y: np.ndarray,
        labels: Sequence[str],
        cluster_labels: np.ndarray,
        ring_geometry: Dict[int, dict],
    ) -> List[dict]:
        labels_arr = np.asarray(labels, dtype=object)
        fig = ax.figure
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        global_center = np.array([np.mean(x), np.mean(y)], dtype=float)
        max_radius = max(float(np.nanmax(np.hypot(x, y))), 1.0)
        base_offset = (
            self.layout.base_offset
            if self.layout.base_offset is not None
            else self.layout.offset_as_fraction_of_radius * max_radius
        )
        base_offset *= self.layout.leader_length_scale

        priorities = self._placement_priority(x, y, labels_arr, cluster_labels)
        order = np.argsort(priorities)[::-1]
        placements: List[Optional[dict]] = [None] * len(labels_arr)
        occupied: List[dict] = []

        angle_candidates = [0.0]
        for step in range(1, self.layout.max_angular_steps + 1):
            delta = np.deg2rad(self.layout.angular_step_deg * step)
            angle_candidates.extend([delta, -delta])

        for idx in order:
            cluster_id = int(cluster_labels[idx])
            point = np.array([x[idx], y[idx]], dtype=float)
            center = self._outward_center(cluster_id, ring_geometry, global_center)
            outward = self._unit_from_center(point, center)
            base_angle = float(np.arctan2(outward[1], outward[0]))

            best_noncolliding = None
            best_colliding = None
            best_colliding_score = np.inf

            for radial_step in range(self.layout.max_radial_steps + 1):
                offset = base_offset * (1.0 + radial_step * self.layout.radial_step_fraction)
                for angle_delta in angle_candidates:
                    theta = base_angle + angle_delta
                    direction = np.array([np.cos(theta), np.sin(theta)], dtype=float)
                    text_xy = point + direction * offset
                    text_meta = self._text_bbox_data_polar(ax, text_xy, str(labels_arr[idx]), direction, renderer)
                    candidate = self._make_placement(idx, point, text_xy, direction, text_meta, str(labels_arr[idx]))
                    score = radial_step * 1000.0 + abs(angle_delta)

                    if self._collides(candidate["bbox_display"], occupied, key="bbox_display"):
                        if score < best_colliding_score:
                            best_colliding_score = score
                            best_colliding = candidate
                        continue

                    best_noncolliding = candidate
                    break
                if best_noncolliding is not None:
                    break

            chosen = best_noncolliding if best_noncolliding is not None else best_colliding
            if chosen is None:
                fallback_xy = point + outward * base_offset
                text_meta = self._text_bbox_data_polar(ax, fallback_xy, str(labels_arr[idx]), outward, renderer)
                chosen = self._make_placement(idx, point, fallback_xy, outward, text_meta, str(labels_arr[idx]))

            placements[idx] = chosen
            occupied.append(chosen)

        return [p for p in placements if p is not None]

    def _text_bbox_data_polar(self, ax: Axes, text_xy: np.ndarray, label: str, direction: np.ndarray, renderer) -> dict:
        ha = "left" if direction[0] > 0.15 else ("right" if direction[0] < -0.15 else "center")
        va = "bottom" if direction[1] > 0.20 else ("top" if direction[1] < -0.20 else "center")
        th_text, r_text = self._xy_to_polar(text_xy)
        temp = ax.text(
            th_text,
            r_text,
            label,
            fontsize=self.label_style.fontsize,
            fontweight=self.label_style.fontweight,
            color=self.label_style.color,
            ha=ha,
            va=va,
            alpha=0.0,
            clip_on=self.label_style.clip_on,
            bbox=self.label_style.bbox,
            zorder=self.label_style.zorder,
        )
        bbox_display = temp.get_window_extent(renderer=renderer)
        temp.remove()
        margin_pixels = self.layout.bbox_margin_pts * ax.figure.dpi / 72.0
        bbox_display = Bbox.from_extents(
            bbox_display.x0 - margin_pixels,
            bbox_display.y0 - margin_pixels,
            bbox_display.x1 + margin_pixels,
            bbox_display.y1 + margin_pixels,
        )
        return {"bbox": bbox_display, "bbox_display": bbox_display, "ha": ha, "va": va}

    @staticmethod
    def _collides(bbox: Bbox, occupied: List[dict], key: str = "bbox") -> bool:
        return any(bbox.overlaps(item[key]) for item in occupied)

    def _draw_one_polar_label(self, ax: Axes, placement: dict) -> None:
        point = np.asarray(placement["point"], dtype=float)
        text_xy = np.asarray(placement["text_xy"], dtype=float)
        direction = np.asarray(placement["direction"], dtype=float)
        unit = direction / max(np.linalg.norm(direction), 1e-12)
        bbox_display = placement["bbox_display"]

        th_point, r_point = self._xy_to_polar(point)
        th_text, r_text = self._xy_to_polar(text_xy)

        start = self._offset_polar_point_by_points(ax, np.array([th_point, r_point], dtype=float), unit, self.layout.line_start_pad_pts)
        end = self._segment_end_before_bbox_display(ax, start, np.array([th_text, r_text], dtype=float), bbox_display, unit)

        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=self.leader_style.color,
            linewidth=self.leader_style.linewidth,
            alpha=self.leader_style.alpha,
            clip_on=self.leader_style.clip_on,
            zorder=self.leader_style.zorder,
        )
        ax.text(
            th_text,
            r_text,
            placement["label"],
            fontsize=self.label_style.fontsize,
            fontweight=self.label_style.fontweight,
            color=self.label_style.color,
            ha=placement["ha"],
            va=placement["va"],
            clip_on=self.label_style.clip_on,
            zorder=self.label_style.zorder,
            bbox=self.label_style.bbox,
        )

    def _offset_polar_point_by_points(self, ax: Axes, polar_tr: np.ndarray, unit_direction_xy: np.ndarray, pad_points: float) -> np.ndarray:
        display_xy = ax.transData.transform(polar_tr)
        pixel_pad = pad_points * ax.figure.dpi / 72.0
        shifted_display = display_xy + unit_direction_xy * pixel_pad
        return ax.transData.inverted().transform(shifted_display)

    def _segment_end_before_bbox_display(
        self,
        ax: Axes,
        start_polar: np.ndarray,
        text_polar: np.ndarray,
        bbox_display: Bbox,
        unit_direction_xy: np.ndarray,
    ) -> np.ndarray:
        start_disp = ax.transData.transform(start_polar)
        text_disp = ax.transData.transform(text_polar)
        dx = text_disp[0] - start_disp[0]
        dy = text_disp[1] - start_disp[1]
        candidates = []
        if abs(dx) > 1e-12:
            for x_edge in (bbox_display.x0, bbox_display.x1):
                t = (x_edge - start_disp[0]) / dx
                if 0.0 <= t <= 1.0:
                    y_at_t = start_disp[1] + t * dy
                    if bbox_display.y0 - 1e-9 <= y_at_t <= bbox_display.y1 + 1e-9:
                        candidates.append(t)
        if abs(dy) > 1e-12:
            for y_edge in (bbox_display.y0, bbox_display.y1):
                t = (y_edge - start_disp[1]) / dy
                if 0.0 <= t <= 1.0:
                    x_at_t = start_disp[0] + t * dx
                    if bbox_display.x0 - 1e-9 <= x_at_t <= bbox_display.x1 + 1e-9:
                        candidates.append(t)
        end_disp = start_disp + min(candidates) * np.array([dx, dy], dtype=float) if candidates else text_disp
        pad_pixels = self.layout.line_end_pad_pts * ax.figure.dpi / 72.0
        end_disp = end_disp - unit_direction_xy * pad_pixels
        return ax.transData.inverted().transform(end_disp)

    def draw_cartesian_rings(self, ax: Axes, ring_geometry: Dict[int, dict], color_info: dict) -> None:
        if self.ring_config is None or not self.ring_config.enabled:
            return
        for cluster_id, geom in ring_geometry.items():
            color = color_info["colors"][int(cluster_id)]
            centroid = geom["centroid"]
            radius = geom["radius"]
            ax.add_patch(
                Circle(
                    (centroid[0], centroid[1]),
                    radius,
                    fill=False,
                    edgecolor=color,
                    linewidth=self.ring_config.linewidth,
                    alpha=self.ring_config.alpha,
                    zorder=self.ring_config.zorder,
                )
            )

    def draw_labels_cartesian(self, ax: Axes, placements: List[dict]) -> None:
        for placement in placements:
            self._draw_one_cartesian_label(ax, placement)

    def draw_labels_polar(self, ax: Axes, placements: List[dict]) -> None:
        for placement in placements:
            self._draw_one_polar_label(ax, placement)

    def _draw_one_cartesian_label(self, ax: Axes, placement: dict) -> None:
        point = np.asarray(placement["point"], dtype=float)
        text_xy = np.asarray(placement["text_xy"], dtype=float)
        direction = np.asarray(placement["direction"], dtype=float)
        unit = direction / max(np.linalg.norm(direction), 1e-12)
        bbox = placement["bbox"]

        start = self._offset_by_points(ax, point, unit, self.layout.line_start_pad_pts)
        end = self._segment_end_before_bbox(ax, start, text_xy, bbox, unit)

        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=self.leader_style.color,
            linewidth=self.leader_style.linewidth,
            alpha=self.leader_style.alpha,
            clip_on=self.leader_style.clip_on,
            zorder=self.leader_style.zorder,
        )
        ax.text(
            text_xy[0],
            text_xy[1],
            placement["label"],
            fontsize=self.label_style.fontsize,
            fontweight=self.label_style.fontweight,
            color=self.label_style.color,
            ha=placement["ha"],
            va=placement["va"],
            clip_on=self.label_style.clip_on,
            zorder=self.label_style.zorder,
            bbox=self.label_style.bbox,
        )
        if self.layout.draw_debug_boxes:
            x0, y0, x1, y1 = bbox.x0, bbox.y0, bbox.x1, bbox.y1
            ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color="red", linewidth=0.3, zorder=10)

    def _placement_priority(
        self,
        x: np.ndarray,
        y: np.ndarray,
        labels: np.ndarray,
        cluster_labels: np.ndarray,
    ) -> np.ndarray:
        points = np.column_stack([x, y])
        crowding = np.zeros(len(points), dtype=float)
        lengths = np.array([len(str(lbl)) for lbl in labels], dtype=float)
        for cluster_id in np.unique(cluster_labels):
            mask = cluster_labels == cluster_id
            pts = points[mask]
            if len(pts) <= 1:
                continue
            deltas = pts[:, None, :] - pts[None, :, :]
            dists = np.linalg.norm(deltas, axis=2)
            np.fill_diagonal(dists, np.inf)
            nearest = np.min(dists, axis=1)
            crowding[mask] = 1.0 / np.maximum(nearest, 1e-9)
        return crowding * 100.0 + lengths

    def _text_bbox_data(self, ax: Axes, text_xy: np.ndarray, label: str, direction: np.ndarray, renderer) -> dict:
        ha = "left" if direction[0] > 0.15 else ("right" if direction[0] < -0.15 else "center")
        va = "bottom" if direction[1] > 0.20 else ("top" if direction[1] < -0.20 else "center")
        temp = ax.text(
            text_xy[0],
            text_xy[1],
            label,
            fontsize=self.label_style.fontsize,
            fontweight=self.label_style.fontweight,
            color=self.label_style.color,
            ha=ha,
            va=va,
            alpha=0.0,
            clip_on=self.label_style.clip_on,
            bbox=self.label_style.bbox,
            zorder=self.label_style.zorder,
        )
        bbox_display = temp.get_window_extent(renderer=renderer)
        temp.remove()
        margin_pixels = self.layout.bbox_margin_pts * ax.figure.dpi / 72.0
        bbox_display = Bbox.from_extents(
            bbox_display.x0 - margin_pixels,
            bbox_display.y0 - margin_pixels,
            bbox_display.x1 + margin_pixels,
            bbox_display.y1 + margin_pixels,
        )
        bbox_data = bbox_display.transformed(ax.transData.inverted())
        return {"bbox": bbox_data, "ha": ha, "va": va}

    @staticmethod
    def _collides(bbox: Bbox, occupied: List[dict], key: str = "bbox") -> bool:
        return any(key in item and bbox.overlaps(item[key]) for item in occupied)

    @staticmethod
    def _make_placement(idx: int, point: np.ndarray, text_xy: np.ndarray, direction: np.ndarray, bbox_data: dict, label: str) -> dict:
        placement = {
            "index": idx,
            "point": point,
            "text_xy": text_xy,
            "direction": direction,
            "bbox": bbox_data["bbox"],
            "ha": bbox_data["ha"],
            "va": bbox_data["va"],
            "label": label,
        }
        if "bbox_display" in bbox_data:
            placement["bbox_display"] = bbox_data["bbox_display"]
        return placement

    def _segment_end_before_bbox(
        self,
        ax: Axes,
        start_data: np.ndarray,
        text_xy_data: np.ndarray,
        bbox_data: Bbox,
        unit_direction: np.ndarray,
    ) -> np.ndarray:
        start_disp = ax.transData.transform(start_data)
        text_disp = ax.transData.transform(text_xy_data)
        bbox_disp = bbox_data.transformed(ax.transData)
        dx = text_disp[0] - start_disp[0]
        dy = text_disp[1] - start_disp[1]
        candidates = []
        if abs(dx) > 1e-12:
            for x_edge in (bbox_disp.x0, bbox_disp.x1):
                t = (x_edge - start_disp[0]) / dx
                if 0.0 <= t <= 1.0:
                    y_at_t = start_disp[1] + t * dy
                    if bbox_disp.y0 - 1e-9 <= y_at_t <= bbox_disp.y1 + 1e-9:
                        candidates.append(t)
        if abs(dy) > 1e-12:
            for y_edge in (bbox_disp.y0, bbox_disp.y1):
                t = (y_edge - start_disp[1]) / dy
                if 0.0 <= t <= 1.0:
                    x_at_t = start_disp[0] + t * dx
                    if bbox_disp.x0 - 1e-9 <= x_at_t <= bbox_disp.x1 + 1e-9:
                        candidates.append(t)
        end_disp = start_disp + min(candidates) * np.array([dx, dy], dtype=float) if candidates else text_disp
        pad_pixels = self.layout.line_end_pad_pts * ax.figure.dpi / 72.0
        end_disp = end_disp - unit_direction * pad_pixels
        return ax.transData.inverted().transform(end_disp)

    def _offset_by_points(self, ax: Axes, xy_data: np.ndarray, unit_direction: np.ndarray, pad_points: float) -> np.ndarray:
        display_xy = ax.transData.transform(xy_data)
        pixel_pad = pad_points * ax.figure.dpi / 72.0
        shifted_display = display_xy + unit_direction * pixel_pad
        return ax.transData.inverted().transform(shifted_display)

    def _outward_center(self, cluster_id: int, ring_geometry: Dict[int, dict], global_center: np.ndarray) -> np.ndarray:
        if cluster_id in ring_geometry:
            return np.asarray(ring_geometry[cluster_id]["centroid"], dtype=float)
        if self.layout.cluster_fallback_to_global_center:
            return global_center
        return np.zeros(2, dtype=float)

    @staticmethod
    def _unit_from_center(point: np.ndarray, center: np.ndarray) -> np.ndarray:
        vector = point - center
        norm = np.linalg.norm(vector)
        if norm <= 1e-12:
            angle = np.arctan2(point[1], point[0])
            return np.array([np.cos(angle), np.sin(angle)], dtype=float)
        return vector / norm

    @staticmethod
    def _xy_to_polar(xy: np.ndarray) -> Tuple[float, float]:
        return float(np.mod(np.arctan2(xy[1], xy[0]), 2.0 * np.pi)), float(np.hypot(xy[0], xy[1]))


# ========================= #
# Plotter APIs              #
# ========================= #

@dataclass
class CartesianClusterPlotter:
    engine: ClusterAnnotationEngine = field(default_factory=ClusterAnnotationEngine)
    scatter_style: ScatterStyle = field(default_factory=ScatterStyle)

    def plot(
        self,
        ax: Axes,
        x: Sequence[float],
        y: Sequence[float],
        labels: Sequence[str],
        cluster_labels: Optional[Sequence[int]],
        title: str,
        xlabel: str = "x (r cos theta)",
        ylabel: str = "y (r sin theta)",
    ) -> dict:
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        labels_arr = np.asarray(labels, dtype=object)
        cluster_arr = np.zeros(len(x_arr), dtype=int) if cluster_labels is None else np.asarray(cluster_labels, dtype=int)

        color_info = self.engine.build_cluster_colors(cluster_arr)
        colors = [color_info["colors"][int(cid)] for cid in cluster_arr]
        ax.scatter(
            x_arr,
            y_arr,
            c=colors,
            s=self.scatter_style.size,
            alpha=self.scatter_style.alpha,
            edgecolors=self.scatter_style.edgecolors,
            zorder=self.scatter_style.zorder,
        )

        ring_geometry = self.engine.compute_cluster_ring_geometry(x_arr, y_arr, cluster_arr)
        self.engine.draw_cartesian_rings(ax, ring_geometry, color_info)
        placements = self.engine.place_labels(ax, x_arr, y_arr, labels_arr, cluster_arr, ring_geometry)
        self.engine.draw_labels_cartesian(ax, placements)

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return {"ring_geometry": ring_geometry, "placements": placements, "color_info": color_info}


@dataclass
class PolarClusterPlotter:
    engine: ClusterAnnotationEngine = field(default_factory=ClusterAnnotationEngine)
    scatter_style: ScatterStyle = field(default_factory=ScatterStyle)
    polar_config: PolarClusterConfig = field(default_factory=PolarClusterConfig)

    def compute_local_polar_clusters(self, theta: Sequence[float], r: Sequence[float]) -> np.ndarray:
        th = np.mod(np.asarray(theta, dtype=float), 2.0 * np.pi)
        rr = np.asarray(r, dtype=float)
        if len(th) == 0:
            return np.array([], dtype=int)

        order = np.argsort(th) if self.polar_config.sort_by_theta else np.arange(len(th))
        th_sorted = th[order]
        r_sorted = rr[order]
        max_r = max(float(np.nanmax(rr)), 1.0)
        ang_thresh = self.polar_config.angle_threshold
        r_thresh = self.polar_config.radius_threshold_fraction * max_r

        sorted_labels = np.zeros(len(th_sorted), dtype=int)
        cluster_id = 0
        sorted_labels[0] = cluster_id
        for i in range(1, len(th_sorted)):
            if abs(th_sorted[i] - th_sorted[i - 1]) <= ang_thresh and abs(r_sorted[i] - r_sorted[i - 1]) <= r_thresh:
                sorted_labels[i] = cluster_id
            else:
                cluster_id += 1
                sorted_labels[i] = cluster_id

        if len(th_sorted) > 1 and sorted_labels[0] != sorted_labels[-1]:
            wrap_dtheta = min(abs(th_sorted[0] + 2 * np.pi - th_sorted[-1]), abs(th_sorted[0] - (th_sorted[-1] - 2 * np.pi)))
            if wrap_dtheta <= ang_thresh and abs(r_sorted[0] - r_sorted[-1]) <= r_thresh:
                last_id = sorted_labels[-1]
                sorted_labels[sorted_labels == last_id] = sorted_labels[0]
                unique = np.unique(sorted_labels)
                remap = {old: new for new, old in enumerate(unique)}
                sorted_labels = np.array([remap[val] for val in sorted_labels], dtype=int)

        labels = np.empty_like(sorted_labels)
        labels[order] = sorted_labels
        return labels

    def plot(
        self,
        ax: Axes,
        theta: Sequence[float],
        r: Sequence[float],
        labels: Sequence[str],
        cluster_labels: Optional[Sequence[int]] = None,
        title: str = "Hyperbolic Embedding Polar Plot",
        radial_limit: Optional[float] = None,
    ) -> dict:
        theta_arr = np.mod(np.asarray(theta, dtype=float), 2.0 * np.pi)
        r_arr = np.asarray(r, dtype=float)
        labels_arr = np.asarray(labels, dtype=object)
        cluster_arr = self.compute_local_polar_clusters(theta_arr, r_arr) if cluster_labels is None else np.asarray(cluster_labels, dtype=int)

        x_arr = r_arr * np.cos(theta_arr)
        y_arr = r_arr * np.sin(theta_arr)
        ring_geometry = self.engine.compute_cluster_centroid_geometry(x_arr, y_arr, cluster_arr)
        color_info = self.engine.build_cluster_colors(cluster_arr, cmap_name="tab10")
        colors = [color_info["colors"][int(cid)] for cid in cluster_arr]

        ax.scatter(
            theta_arr,
            r_arr,
            c=colors,
            s=self.scatter_style.size,
            alpha=self.scatter_style.alpha,
            edgecolors=self.scatter_style.edgecolors,
            zorder=self.scatter_style.zorder,
        )

        placements = self.engine.place_labels_polar(ax, x_arr, y_arr, labels_arr, cluster_arr, ring_geometry)
        self.engine.draw_labels_polar(ax, placements)

        ax.set_title(title)
        ax.set_ylim(0, radial_limit if radial_limit is not None else self.polar_config.radial_limit)
        if self.polar_config.theta_zero_location is not None:
            ax.set_theta_zero_location(self.polar_config.theta_zero_location)
        ax.set_theta_direction(self.polar_config.theta_direction)
        return {
            "ring_geometry": ring_geometry,
            "placements": placements,
            "color_info": color_info,
            "cluster_labels": cluster_arr,
        }


# ========================= #
# Reusable workflows        #
# ========================= #

class GeneClusterWorkflow:
    def __init__(
        self,
        input_csv: str = "results/polar_plot_points.csv",
        results_dir: str = "results",
        figures_dir: str = "figures",
        cartesian_plotter: Optional[CartesianClusterPlotter] = None,
    ) -> None:
        self.input_csv = Path(input_csv)
        self.results_dir = Path(results_dir)
        self.figures_dir = Path(figures_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.cartesian_plotter = cartesian_plotter or CartesianClusterPlotter()

    def load_points(self) -> pd.DataFrame:
        return pd.read_csv(self.input_csv)

    def prepare_cartesian_data(
        self,
        df: pd.DataFrame,
        r_col: str = "r",
        theta_col: str = "theta",
        label_col: str = "organelle",
        trim_slice: Optional[slice] = slice(1, -3),
    ) -> dict:
        work = df.iloc[trim_slice].copy() if trim_slice is not None else df.copy()
        r = pd.to_numeric(work[r_col]).to_numpy()
        theta = pd.to_numeric(work[theta_col]).to_numpy()
        labels = work[label_col].astype(str).to_numpy()
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        xy = np.column_stack([x, y])
        xy = np.nan_to_num(xy, nan=0.0, posinf=0.0, neginf=0.0)
        return {
            "dataframe": work,
            "r": r,
            "theta": theta,
            "labels": labels,
            "x": xy[:, 0],
            "y": xy[:, 1],
            "cartesian": xy,
        }

    def compute_kmeans_labels(self, cartesian: np.ndarray, cluster_counts: Sequence[int]) -> Dict[int, np.ndarray]:
        return {
            int(k): KMeans(n_clusters=int(k), random_state=10).fit_predict(cartesian)
            for k in cluster_counts
        }

    def compute_ward_labels(self, cartesian: np.ndarray, cluster_counts: Sequence[int], distance_thresholds: Sequence[float]) -> dict:
        z = linkage(cartesian, method="ward")
        return {
            "linkage": z,
            "maxclust": {int(k): fcluster(z, t=int(k), criterion="maxclust") for k in cluster_counts},
            "distance": {float(t): fcluster(z, t=float(t), criterion="distance") for t in distance_thresholds},
        }

    def save_dendrogram(self, z: np.ndarray, output_path: str = "figures/wards_dendrogram.png") -> None:
        fig, ax = plt.subplots(figsize=(10, 6))
        dendrogram(z, ax=ax)
        ax.set_title("Ward's Hierarchical Clustering Dendrogram")
        ax.set_xlabel("Sample Index or (Cluster size)")
        ax.set_ylabel("Distance")
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def save_cluster_assignments(self, labels: Sequence[str], cluster_labels: Sequence[int], prefix: str, param: str) -> None:
        out = pd.DataFrame({"organelle": labels, "cluster": np.asarray(cluster_labels, dtype=int)})
        out.to_csv(self.results_dir / f"{prefix}_{param}.csv", index=False)

    def plot_cartesian(self, x: Sequence[float], y: Sequence[float], labels: Sequence[str], cluster_labels: Sequence[int], title: str, output_path: str) -> dict:
        fig, ax = plt.subplots(figsize=(10, 8))
        result = self.cartesian_plotter.plot(ax=ax, x=x, y=y, labels=labels, cluster_labels=cluster_labels, title=title)
        plt.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return result

    def run_all(
        self,
        cluster_counts: Sequence[int] = (6, 5, 4, 3, 2),
        distance_thresholds: Sequence[float] = (5.0, 3.0),
    ) -> dict:
        df = self.load_points()
        prepared = self.prepare_cartesian_data(df)
        x = prepared["x"]
        y = prepared["y"]
        labels = prepared["labels"]
        cartesian = prepared["cartesian"]

        kmeans = self.compute_kmeans_labels(cartesian, cluster_counts)
        ward = self.compute_ward_labels(cartesian, cluster_counts, distance_thresholds)
        self.save_dendrogram(ward["linkage"], str(self.figures_dir / "wards_dendrogram.png"))

        for k, cluster_labels in kmeans.items():
            self.plot_cartesian(x, y, labels, cluster_labels, f"K-Means (k={k})", str(self.figures_dir / f"kmeans_{k}.png"))
            self.save_cluster_assignments(labels, cluster_labels, "kmeans", str(k))

        for k, cluster_labels in ward["maxclust"].items():
            self.plot_cartesian(x, y, labels, cluster_labels, f"Ward's Hierarchical (k={k})", str(self.figures_dir / f"wards_hierarchical_{k}.png"))
            self.save_cluster_assignments(labels, cluster_labels, "wards_hierarchical", str(k))

        for dist, cluster_labels in ward["distance"].items():
            self.plot_cartesian(x, y, labels, cluster_labels, f"Ward's Hierarchical (distance={dist})", str(self.figures_dir / f"wards_distance_{dist}.png"))
            self.save_cluster_assignments(labels, cluster_labels, "wards_distance", str(dist))

        original_clusters = np.zeros(len(x), dtype=int)
        self.plot_cartesian(x, y, labels, original_clusters, "Original Data", str(self.figures_dir / "original_data.png"))

        return {"prepared": prepared, "kmeans": kmeans, "ward": ward}


class PolarPlotWorkflow:
    def __init__(
        self,
        input_csv: str = "results/polar_plot_points.csv",
        results_dir: str = "results",
        polar_plotter: Optional[PolarClusterPlotter] = None,
    ) -> None:
        self.input_csv = Path(input_csv)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.polar_plotter = polar_plotter or PolarClusterPlotter()

    def load_points(self) -> pd.DataFrame:
        return pd.read_csv(self.input_csv)

    def create_polar_plot(
        self,
        df: pd.DataFrame,
        theta_col: str = "theta",
        r_col: str = "r",
        label_col: str = "organelle",
        image_path: Optional[str] = None,
        radial_limit: Optional[float] = None,
        figsize: Tuple[float, float] = (7, 7),
        title: str = "Hyperbolic Embedding Polar Plot",
        cluster_labels: Optional[Sequence[int]] = None,
    ) -> dict:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="polar")
        result = self.polar_plotter.plot(
            ax=ax,
            theta=df[theta_col].to_numpy(),
            r=df[r_col].to_numpy(),
            labels=df[label_col].astype(str).to_numpy(),
            cluster_labels=cluster_labels,
            title=title,
            radial_limit=radial_limit,
        )
        if image_path is not None:
            fig.savefig(image_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return result

    def create_distance_from_mean_plot(
        self,
        df: pd.DataFrame,
        theta_col: str = "theta",
        r_col: str = "r",
        label_col: str = "organelle",
        output_path: Optional[str] = None,
        figsize: Tuple[float, float] = (10, 4),
    ) -> pd.DataFrame:
        out = df.copy()
        mean_r = float(np.mean(out[r_col]))
        mean_theta = float(np.mean(out[theta_col]))
        out["delta_r"] = out[r_col] - mean_r
        out["delta_theta"] = out[theta_col] - mean_theta

        labels = out[label_col].fillna("<missing>").map(str).tolist()
        x = np.arange(len(labels))

        fig, axes = plt.subplots(1, 2, figsize=figsize)
        axes[0].bar(x, out["delta_r"])
        axes[0].set_title("Deviation from Mean Radius")
        axes[0].tick_params(axis="x", rotation=90)

        axes[1].bar(x, out["delta_theta"])
        axes[1].set_title("Deviation from Mean Angle")
        axes[1].tick_params(axis="x", rotation=90)

        plt.tight_layout()
        if output_path is not None:
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out


if __name__ == "__main__":
    gene = GeneClusterWorkflow()
    gene.run_all()

    polar = PolarPlotWorkflow()
    df = polar.load_points()
    polar.create_polar_plot(df, image_path="results/polar_plot_refactored.png")
    polar.create_distance_from_mean_plot(df, output_path="results/polar_deviation_from_mean.png")
