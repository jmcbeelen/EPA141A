from ema_workbench import Model, MultiprocessingEvaluator, Policy, Scenario

from ema_workbench.em_framework.evaluators import perform_experiments
from ema_workbench.em_framework.samplers import sample_uncertainties
from ema_workbench.util import ema_logging
import time
from problem_formulation import get_model_for_problem_formulation
from ema_workbench.analysis import pairs_plotting
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from ema_workbench.analysis import feature_scoring


ema_logging.log_to_stderr(ema_logging.INFO)

if __name__ == "__main__":
    model, _ = get_model_for_problem_formulation(1)

    with MultiprocessingEvaluator(model, n_processes=-1) as evaluator:
        experiments, outcomes = evaluator.perform_experiments(scenarios=10, policies=10)

    x = experiments
    y = outcomes

    # --- Visualization 1: Pairwise scatterplot grouped by policy ---
    fig, axes = pairs_plotting.pairs_scatter(x, y, group_by="policy", legend=False)
    fig.set_size_inches(8, 8)
    fig.set_constrained_layout(True)
    plt.show()

    # Calculate feature importance scores for all outcomes
    # --- Step 1: Compute feature importance scores ---
    x_clean = x.drop(columns=["policy"], errors="ignore")
    fs = feature_scoring.get_feature_scores_all(x_clean, y)

    lever_names = [l.name for l in model.levers]
    fs_uncertainties_only = fs.drop(index=lever_names, errors="ignore")

    # --- Step 3: Plot heatmap ---
    plt.figure(figsize=(10, 8))
    sns.heatmap(fs_uncertainties_only, cmap="viridis", annot=True, fmt=".3f")
    plt.title("Feature Importance Scores ")
    plt.xlabel("Outcomes")
    plt.ylabel("Uncertainties")
    plt.tight_layout()
    plt.show()

    print(experiments.loc[0])




