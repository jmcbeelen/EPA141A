#import
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
from ema_workbench.analysis import prim


def get_do_nothing_dict():
    return {l.name: 0 for l in model.levers}

if __name__ == "__main__":
    model, _ = get_model_for_problem_formulation(4)
    with MultiprocessingEvaluator(model, n_processes=-1) as evaluator:
        results = evaluator.perform_experiments(scenarios=100, policies=[Policy("baseline", **get_do_nothing_dict())] )

        experiments, outcomes = results

        #
        # hri_ok   = outcomes["Hydrological Resilience Index"][:, :-1, :].min(axis=(1,2)) >= 0
        # ead_ok   = outcomes["Expected Annual Damage"].sum(axis=1).max(axis=1) < 5e6
        # rfr_ok   = outcomes["RfR Total Costs"].flatten() < 50e6   # shape (n_exp,)
        #
        # y = hri_ok & ead_ok & rfr_ok
        #
        # cols_unc           = [c for c in experiments.columns if "_Bmax" in c or "_Brate" in c]
        # print (cols_unc)
