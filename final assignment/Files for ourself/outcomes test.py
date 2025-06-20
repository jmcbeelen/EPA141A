from ema_workbench import Model, MultiprocessingEvaluator
from ema_workbench.util import ema_logging
from problem_formulation import get_model_for_problem_formulation

ema_logging.log_to_stderr(ema_logging.INFO)

if __name__ == "__main__":
    # Load the model for a specific problem formulation
    model, _ = get_model_for_problem_formulation(3)


    # Run a single experiment
    with MultiprocessingEvaluator(model, n_processes=1) as evaluator:
        experiments, outcomes = evaluator.perform_experiments(scenarios=10, policies=10)

    # Print the outcomes clearly
    print("\n--- Simulation Outcomes ---")
    for name, values in outcomes.items():
        print(f"{name}: {values[0]}")
