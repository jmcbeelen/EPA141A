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
import matplotlib

#matplotlib.use("Qt5Agg")  # of "Qt5Agg", afhankelijk van wat je geïnstalleerd hebt
import matplotlib.pyplot as plt
from ema_workbench.analysis import prim
from ema_workbench.analysis import dimensional_stacking


def get_do_nothing_dict():
    return {l.name: 0 for l in model.levers}

if __name__ == "__main__":
    model, _ = get_model_for_problem_formulation(5)
    scenarios=10000
    policies=4
    with MultiprocessingEvaluator(model, n_processes=-1) as evaluator:
        results = evaluator.perform_experiments(scenarios=scenarios, policies=policies )

        experiments, outcomes = results

        print(experiments)

        # ---- 2.1 Hydrological Resilience Index ----
        hri_sys = outcomes["A.2_Hydrological Resilience Index"].mean(axis=1)
        hri_q25 = np.percentile(hri_sys, 25)# (N,) systeembreed
        target_hri = hri_sys <= hri_q25

        # ---- 2.2 Expected Annual Damage ----
        ead_total = outcomes["A.2_Expected Annual Damage"].sum(axis=1)  # (N,)
        ead_q75 = np.percentile(ead_total, 75)
        target_ead = ead_total >= ead_q75

        # # ---- 2.3 RfR Total Costs ----
        # rfr_costs = outcomes["RfR Total Costs"].sum(axis=1)  # (N,)
        # rfr_q75 = np.percentile(rfr_costs, 75)
        # target_rfr = rfr_costs >= rfr_q75

        # ---- 2.4 Combineer tot één worst-case target ----
        y = target_hri | target_ead        # booleaanse vector (N,)
        print(f"worst-cases: {y.sum()} van {scenarios*policies} runs  ({y.mean()*100:.1f} %)")

        #Determine model uncertainties
        cols_unc = [u.name for u in model.uncertainties]
        X = experiments[cols_unc]

        #Start PRIM
        prim_alg = prim.Prim(
            X, y,
            threshold=0.6,
            peel_alpha=0.1,
            mass_min=0.05
        )

        box = prim_alg.find_box()
        box.inspect()
        box.show_tradeoff()

        box.inspect(style="graph")
        fig = box.show_pairs_scatter()


        lever_names = [lev.name for lev in model.levers]
        experiments_unc = experiments.drop(columns=lever_names)

        dimensional_stacking.create_pivot_plot(experiments_unc, y)
        plt.show()




