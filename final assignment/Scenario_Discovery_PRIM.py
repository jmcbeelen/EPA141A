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
    scenarios=100
    policies=4
    with MultiprocessingEvaluator(model, n_processes=-1) as evaluator:
        results = evaluator.perform_experiments(scenarios=scenarios, policies=policies )

        experiments, outcomes = results

        print(experiments)

        # ---- 2.1 Hydrological Resilience Index ----
        hri_sys = outcomes["Hydrological Resilience Index"][:, -1, :].mean(axis=1)  # (N,) systeembreed
        target_hri = hri_sys < 0

        # ---- 2.2 Expected Annual Damage ----
        ead_total = outcomes["Expected Annual Damage"].sum(axis=1)  # (N,)
        ead_q75 = np.percentile(ead_total, 75)
        target_ead = ead_total >= ead_q75

        # ---- 2.3 RfR Total Costs ----
        rfr_costs = outcomes["RfR Total Costs"].sum(axis=1)  # (N,)
        rfr_q75 = np.percentile(rfr_costs, 75)
        target_rfr = rfr_costs >= rfr_q75

        # ---- 2.4 Combineer tot één worst-case target ----
        y = target_hri | target_ead | target_rfr        # booleaanse vector (N,)
        print(f"worst-cases: {y.sum()} van {scenarios*policies} runs  ({y.mean()*100:.1f} %)")

        #Determine model uncertainties
        cols_unc = [u.name for u in model.uncertainties]
        X = experiments[cols_unc]

        #Start PRIM
        prim_alg = prim.Prim(
            X, y,
            threshold=0.8,
            peel_alpha=0.1,
            mass_min=0.05
        )

        box = prim_alg.find_box()

        box.show_tradeoff()
        plt.show()

        box.inspect()





