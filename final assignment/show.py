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
    model, _ = get_model_for_problem_formulation(3)

    with MultiprocessingEvaluator(model, n_processes=-1) as evaluator:
        experiments, outcomes = evaluator.perform_experiments(scenarios=10, policies=10)

    experiments_df = pd.DataFrame(experiments)
    experiments_df.to_csv("experiments.csv", index=False)
    outcomes_df = pd.DataFrame.from_dict(outcomes)
    outcomes_df.to_csv("outcomes.csv", index=False)

    print(experiments_df)
    print(outcomes_df)

