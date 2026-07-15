# SPDX-License-Identifier: MIT

import os
import pyvista as pv
 
import src.utils.plot as pl
from src.postprocessing import FlowFieldPostProcessor

def load_flowfield(params):
    """
    Load the CFD mesh/flowfield used for full-field prediction,
    VTK writing, and PyVista plotting.
    """

    flowfield_path = os.path.join(
        params['paths']['flow'],
        params['files']['flowfield'],
    )

    return pv.read(flowfield_path)


def run_flowfield_postprocessing(model, params):
    """
    Run the complete post-processing workflow on the full CFD mesh.

    This includes:
      1. loading the CFD flowfield,
      2. predicting the PINN/GNN solution on the full mesh,
      3. writing predicted and error fields,
      4. plotting simulation, prediction, and error fields.

    Parameters
    ----------
    model : PhysicsInformedNN
        Trained PINN/GNN model.

    params : dict
        Configuration dictionary.

    Returns
    -------
    postprocessor : FlowFieldPostProcessor
        Postprocessor object containing predicted and error fields.

    flowfield : pyvista.DataSet
        Loaded CFD mesh/flowfield.
    """

    if not params["run"]["routines"].get("inference", False): 
        print("---------------------------------------")
        print("Skipping flowfield post-processing.")
        print("Inference routine is disabled.")
        return None

    # Load CFD mesh/flowfield once
    flowfield = load_flowfield(params)

    # Post-processing on the full CFD mesh
    postprocessor = FlowFieldPostProcessor(
        model=model,
        flowfield=flowfield,
        params=params,
    )

    postprocessor.run(prefix=params["name"])

    do_plotting = (
        model.device_str == "cpu"
        and params["post_processing"].get("plot", True)
    )

    # Note that, after training, the model can be reused for post-processing 
    # without retraining.

    #To do this, keep the saved model checkpoint and the sampled data files. 
    # In a new run, disable sampling, load the existing data, load the saved 
    # model, and set the number of Adam and LBFGS iterations to zero. The code 
    # will reconstruct the PINN object, load the trained weights, skip the 
    # optimization step, and directly evaluate the model on the CFD mesh for 
    # VTK output and plotting.
    if do_plotting:
        # Plot CFD/simulation fields
        pl.plot_simulation_flow_pyvista(flowfield, params)

        # Plot predicted flow field
        pred_fields = postprocessor.get_predicted_fields()

        pl.plot_predicted_flow_pyvista(
            flowfield,
            pred_fields,
            params,
        )

        # Plot scaled absolute error flow field
        error_fields = postprocessor.get_error_fields()

        pl.plot_error_flow_pyvista(
            flowfield,
            error_fields,
            params,
            error_type=params["post_processing"]["error_type"],
        )

    else:
        print("---------------------------------------")
        print(
            "Skipping PyVista plotting. Plotting is only enabled when "
            "running in cpu device and "
            "params['post_processing']['plot'] is True."
        )

 