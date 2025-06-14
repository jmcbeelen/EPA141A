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
from ema_workbench import MultiprocessingEvaluator, ema_logging
from ema_workbench import MultiprocessingEvaluator

from ema_workbench import MultiprocessingEvaluator
from ema_workbench.em_framework.optimization import (HyperVolume, EpsilonProgress, ArchiveLogger)
from ema_workbench.em_framework.optimization import (EpsNSGAII)
import numpy as np


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

from ema_workbench.analysis import feature_scoring



if __name__ == "__main__":

    # --- Load your model ---
    model, planning_steps = get_model_for_problem_formulation(2)

    ema_logging.log_to_stderr(ema_logging.INFO)

    with MultiprocessingEvaluator(model) as evaluator:
        results1 = evaluator.optimize(nfe=5e3, searchover='levers',
                                      epsilons=[0.1, ] * len(model.outcomes))