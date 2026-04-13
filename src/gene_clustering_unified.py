from unified_cluster_plotting import GeneClusterWorkflow


if __name__ == "__main__":
    workflow = GeneClusterWorkflow(
        input_csv="results/polar_plot_points.csv",
        results_dir="results",
        figures_dir="figures",
    )
    workflow.run_all()
