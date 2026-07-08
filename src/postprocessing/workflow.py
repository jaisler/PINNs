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
        params["paths"]["flow"],
        params["files"]["flowfield"],
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

    # Plot CFD/simulation fields
    pl.plot_simulation_flow(flowfield, params)

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