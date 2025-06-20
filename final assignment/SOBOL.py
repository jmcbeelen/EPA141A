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

# === Main Entry Point ===
if __name__ == "__main__":

    # === 1. Load the DikeNetwork model and planning steps ===
    # Using problem formulation 2
    model, planning_steps = get_model_for_problem_formulation(2)

    # === 2. Define the SALib problem based on model uncertainties ===
    # This tells SALib which uncertainties to vary and how
    problem = get_SALib_problem(model.uncertainties)

    # === 3. Run Sobol sensitivity experiments ===
    # Sobol sampling requires the number of scenarios to be a power of 2
    n_scenarios = 20

    # Perform experiments using 5 policies across the Sobol-sampled scenarios
    with MultiprocessingEvaluator(model, n_processes=-1) as evaluator:
        experiments, outcomes = evaluator.perform_experiments(
            scenarios=n_scenarios,
            policies=5,
            uncertainty_sampling=Samplers.SOBOL
        )

    # === 4. Analyze sensitivity for a specific outcome ===
    # Select the outcome to analyze (can be changed as needed)
    Y = outcomes["System HRI (aggregate)"]

    # Run Sobol sensitivity analysis
    Si = sobol.analyze(problem, Y, calc_second_order=True, print_to_console=True)

    # === 5. Prepare results for visualization ===
    # Create a dataframe to store first-order and total-order indices + confidence intervals
    df = pd.DataFrame({
        "First Order": Si["S1"],
        "First Order Conf": Si["S1_conf"],
        "Total Order": Si["ST"],
        "Total Order Conf": Si["ST_conf"]
    }, index=problem["names"])

    # === 6. Plot Sobol indices ===
    df[["First Order", "Total Order"]].plot(
        kind="bar",
        yerr=df[["First Order Conf", "Total Order Conf"]].values.T,  # Add confidence intervals
        figsize=(12, 6)
    )

    plt.title("System HRI (aggregate)")
    plt.ylabel("Sobol Index")
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.tight_layout()

    # === 7. Save and show the figure ===
    plt.savefig("System HRI (aggregate).png", dpi=300)
    plt.show()


