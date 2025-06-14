import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from ema_workbench import (Model, Policy, MultiprocessingEvaluator, Samplers)
from ema_workbench.analysis import feature_scoring
from ema_workbench.em_framework.salib_samplers import get_SALib_problem
from SALib.analyze import sobol
from problem_formulation import get_model_for_problem_formulation
from ema_workbench.analysis import feature_scoring

if __name__ == "__main__":

    # --- Load model and define zero policy ---
    model, planning_steps = get_model_for_problem_formulation(2)

    #zero_policy_dict = {lever.name: 0 for lever in model.levers}
    #zero_policy = Policy("zero_policy", **zero_policy_dict)

    # --- Define SALib problem ---
    problem = get_SALib_problem(model.uncertainties)

    # --- Run Sobol experiments ---
    n_scenarios = 10 # Must be a power of 2
    with MultiprocessingEvaluator(model, n_processes=-1) as evaluator:
        experiments, outcomes = evaluator.perform_experiments(
            scenarios=n_scenarios,
            policies=5,
            uncertainty_sampling=Samplers.SOBOL
        )

    # --- Analyze an outcome (e.g. Expected Annual Damage) ---
    Y = outcomes["System HRI (aggregate)"]

    Si = sobol.analyze(problem, Y, calc_second_order=True, print_to_console=True)

    # --- Visualize ---
    df = pd.DataFrame({
        "First Order": Si["S1"],
        "First Order Conf": Si["S1_conf"],
        "Total Order": Si["ST"],
        "Total Order Conf": Si["ST_conf"]
    }, index=problem["names"])

    df[["First Order", "Total Order"]].plot(
    kind="bar",
    yerr=df[["First Order Conf", "Total Order Conf"]].values.T,  # Add this
    figsize=(12, 6)
)
    plt.title("System HRI (aggregate)")
    plt.ylabel("Sobol Index")
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig("System HRI (aggregate).png", dpi=300)

    plt.show()


