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
    model, _ = get_model_for_problem_formulation(4)

    with MultiprocessingEvaluator(model, n_processes=-1) as evaluator:
        experiments, outcomes = evaluator.perform_experiments(scenarios=100, policies=3)

    # Now you can visualize or analyze results
    print(experiments.head())

    #fig, axes = pairs_plotting.pairs_scatter(experiments, outcomes, group_by="policy", legend=False)
    #fig.set_size_inches(8, 8)
    #plt.show()

    x = experiments
    y = outcomes

    fs = feature_scoring.get_feature_scores_all(x, y)
    sns.heatmap(fs, cmap="viridis", annot=True)
    plt.show()

