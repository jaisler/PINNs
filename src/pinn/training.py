# SPDX-License-Identifier: MIT
import time

from src.utils import print_metrics_table
from src.utils import plot_history_training

def training_is_enabled(params):
    """
    Determine whether training is enabled for at least one optimizer.

    Training is considered enabled when either Adam or L-BFGS is enabled
    and its configured number of iterations is greater than zero.

    Parameters
    ----------
    params : dict
        Configuration parameters containing the optimizer settings.

    Returns
    -------
    bool
        ``True`` if Adam or L-BFGS training is enabled with a positive
        number of iterations; otherwise, ``False``.
    """

    adam_enabled = params["optimizer"]["adam"].get("enabled", False)
    lbfgs_enabled = params["optimizer"]["lbfgs"].get("enabled", False)

    adam_iterations = int(
        params["optimizer"]["adam"].get("iterations", 0)
    )

    lbfgs_iterations = int(
        params["optimizer"]["lbfgs"].get("iterations", 0)
    )

    return (
        adam_enabled and adam_iterations > 0
    ) or (
        lbfgs_enabled and lbfgs_iterations > 0
    )

def train_model(model, params):
    """
    Train model, save its checkpoints, and plot its loss history.

    Parameters
    ----------
    model : PhysicsInformedNN
        PhysicsInformedNN class object

    params : dict
        Configuration parameters dictionary.
    """

    # Check if training is enabled
    do_training = training_is_enabled(params)

    # Train
    if not do_training:
        print("---------------------------------------")
        print("Skipping training. Adam and LBFGS are disabled or both have "
              "zero iterations.")
        return

    start_time = time.time()                
    model.fit()
    elapsed = time.time() - start_time
    print("---------------------------------------")                
    print('Training time: %.4f' % (elapsed))

    # Save model
    if params['run']['checkpoint']['save_model']:
        model.save_model(
            params['paths']['model'], 
            params['files']['model_name']
        )

    # Plot history training
    plot_history_training(model, params)


def evaluate_data(model, data):
    """
    Evaluate the model using test dataset.
    
    Parameters
    ----------
    model : PhysicsInformedNN
        Initialized physics-informed neural network model. Object of the
        PhysicsInformedNN class.

    data : dict
        Dictionary containing the prepared training, validation, and
        collocation datasets.
    """

    test_data_available = (
        data["xtest"] is not None 
        and data["ytest"] is not None 
        and data["xtest"].shape[0] > 0
    )

    if not test_data_available:
        print("---------------------------------------")
        print("Skipping test evaluation.")
        print("No test data were created. "
              "This usually means N_test_data = 0 after the "
              "train/validation/test split.")
        return

    test_metrics = model.evaluate_data(
        data["xtest"], data["ytest"], data["rhotest"], data["utest"], 
        data["vtest"], data["ptest"], data["muttest"]
    )

    # Print metrics of the test dataset
    print_metrics_table(test_metrics, title="Test dataset metrics")