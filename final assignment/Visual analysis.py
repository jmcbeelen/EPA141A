from ema_workbench import Model, MultiprocessingEvaluator, Policy, Scenario
from ema_workbench.util import ema_logging
from problem_formulation import get_model_for_problem_formulation
from ema_workbench.analysis import pairs_plotting
import seaborn as sns
import matplotlib.pyplot as plt
from ema_workbench.analysis import feature_scoring


# This sets up logging so you can see what's happening during execution
ema_logging.log_to_stderr(ema_logging.INFO)

# === Main Execution Block ===
if __name__ == "__main__":

    # === Load the model for Problem Formulation 2 ===
    model, _ = get_model_for_problem_formulation(2)

    # === Run 1000 experiments on 5 policies using parallel processing ===
    with MultiprocessingEvaluator(model, n_processes=-1) as evaluator:
        experiments, outcomes = evaluator.perform_experiments(scenarios=1000, policies=5)

    x = experiments  # Design matrix with uncertainties and policy
    y = outcomes     # Simulation results

    # === PART 1: Visual Exploration with Pairwise Scatter Plots ===

    # Create a pairwise scatter plot matrix grouped by policy
    fig, axes = pairs_plotting.pairs_scatter(x, y, group_by="policy", legend=False)
    fig.set_size_inches(14, 14)

    # Rotate axis labels for better readability
    for ax in fig.get_axes():
        ax.tick_params(axis='x', labelrotation=45)
        ax.tick_params(axis='y', labelsize=8)

    # Adjust layout to prevent overlap
    fig.subplots_adjust(bottom=0.22)

    # Save the figure
    plt.savefig("correlation.png", dpi=300, bbox_inches='tight')
    plt.show()

    # === PART 2: Extra Trees – Feature Importance Analysis ===

    # Clean the design matrix by removing policy column (not a true uncertainty)
    x_clean = x.drop(columns=["policy"], errors="ignore")

    # Compute feature importance scores (using Extra Trees) for each outcome
    fs = feature_scoring.get_feature_scores_all(x_clean, y)

    # Filter out levers so we keep only uncertainties in the heatmap
    lever_names = [l.name for l in model.levers]
    fs_uncertainties_only = fs.drop(index=lever_names, errors="ignore")

    # Plot the heatmap of feature importance
    plt.figure(figsize=(10, 8))
    sns.heatmap(fs_uncertainties_only, cmap="viridis", annot=True, fmt=".3f")
    plt.title("Feature Importance Scores")
    plt.xlabel("Outcomes")
    plt.ylabel("Uncertainties")
    plt.tight_layout()

    # Save the heatmap
    plt.savefig("Feature Importance Score.png", dpi=300)
    plt.show()








