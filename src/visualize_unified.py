from unified_cluster_plotting_fixed_v2 import PolarPlotWorkflow


if __name__ == "__main__":
    workflow = PolarPlotWorkflow(
        input_csv="results/polar_plot_points.csv",
        results_dir="results",
    )
    df = workflow.load_points()
    workflow.create_polar_plot(
        df,
        theta_col="theta",
        r_col="r",
        label_col="organelle",
        image_path="results/polar_plot.png",
        radial_limit=12.0,
        figsize=(7, 7),
    )
    workflow.create_distance_from_mean_plot(
        df,
        theta_col="theta",
        r_col="r",
        label_col="organelle",
        output_path="results/polar_deviation_from_mean.png",
        figsize=(10, 4),
    )
