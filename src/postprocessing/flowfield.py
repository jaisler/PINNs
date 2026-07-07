# SPDX-License-Identifier: MIT

from pathlib import Path
import numpy as np

class FlowFieldPostProcessor:
    """
    Post-process predicted and reference CFD flow fields on the same mesh.

    The class:
        1. predicts the flow variables on the CFD mesh points,
        2. extracts the reference CFD fields from flowfield.point_data,
        3. computes error fields,
        4. scales absolute errors from 0 to 1,
        5. writes VTK files,
        6. computes mesh-based metrics.

    The CFD/reference solution is assumed to already be stored in `flowfield`.
    """

    def __init__(
        self,
        model,
        flowfield,
        params,
        eps=1.0e-12,
    ):
        """
        Parameters
        ----------
        model : PhysicsInformedNN
            Trained PINN/GNN/MLP model. It must provide model.predict(x, y).

        flowfield : pyvista.DataSet
            Original CFD mesh/flowfield.

        params : dict
            Configuration dictionary.

        eps : float
            Small value used to avoid division by zero.
        """

        if flowfield is None:
            raise ValueError(
                "FlowFieldPostProcessor received flowfield=None. "
                "The CFD mesh/flowfield must be loaded before post-processing."
            )

        self.model = model
        self.flowfield = flowfield
        self.params = params
        self.eps = eps

        self.output_dir = Path(params["paths"]["results"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.n_points = self.flowfield.n_points

        self.predicted_fields = {}
        self.reference_fields = {}
        self.error_fields = {}
        self.metrics = {}

    def run(self, prefix="flowfield"):
        """
        Run the complete mesh-based post-processing workflow.

        Steps
        -----
        1. Predict on all CFD mesh points.
        2. Extract reference CFD fields.
        3. Compute pointwise errors.
        4. Compute scalar metrics.
        5. Write predicted flowfield to VTK.
        6. Write reference/prediction/error fields to VTK.

        Parameters
        ----------
        prefix : str
            Prefix used for output VTK files.

        Returns
        -------
        dict
            Dictionary containing the written VTK file paths.
        """

        print("---------------------------------------")
        print("Mesh-based post-processing")

        # Get the predict flow field on the mesh points
        self.predict_on_mesh()
        # Extract reference flow field from CFD simulation
        self.extract_reference_fields()
        # Compute the error between the prediction and simulation flow field
        #self.compute_error_fields()
        # Compute the metrics
        #self.compute_metrics()
        
        # Write the prediction field to vtk file
        #self.write_prediction_vtk(prefix)
        # Write the error field to vtk file
        #self.write_error_vtk(prefix)
        # Print metrics
        #self.print_metrics()

    def predict_on_mesh(self):
        """
        Evaluate the trained model at all CFD mesh points.

        The prediction is performed on:
            xmesh = flowfield.points[:, 0]
            ymesh = flowfield.points[:, 1]

        The resulting fields are stored in self.predicted_fields.
        """

        xmesh = self.flowfield.points[:, 0]
        ymesh = self.flowfield.points[:, 1]

        equation = self.params["run"]["equation"]

        if equation == "Euler":
            rho_pred, u_pred, v_pred, p_pred = self.model.predict(
                xmesh,
                ymesh,
            )

            self.predicted_fields = {
                "rho": rho_pred,
                "u": u_pred,
                "v": v_pred,
                "p": p_pred,
            }

        elif equation == "RANS":
            rho_pred, u_pred, v_pred, p_pred, mut_pred = self.model.predict(
                xmesh,
                ymesh,
            )

            self.predicted_fields = {
                "rho": rho_pred,
                "u": u_pred,
                "v": v_pred,
                "p": p_pred,
                "mut": mut_pred,
            }

        else:
            raise ValueError(f"Unknown equation: {equation}.")

        # Conversion to 1D Numpy array
        for name in self.predicted_fields:
            value = self.predicted_fields[name]
            value = self.to_numpy_1d(value)
            self.predicted_fields[name] = value
    
def extract_reference_fields(self):
        """
        Extract the reference CFD fields from flowfield.point_data.

        The field names and components are read from:

            params["plot_flow"]["fields"]
            params["plot_flow"]["components"]

        Example for Euler:

            plot_flow:
              fields:
                - "Density"
                - "Pressure"
                - "Velocity"
                - "Velocity"

              components: [0, 0, 0, 1]

        This means:

            rho -> Density,  component 0
            p   -> Pressure, component 0
            u   -> Velocity, component 0
            v   -> Velocity, component 1
        """

        equation = self.params["run"]["equation"]

        if equation == "Euler":
            variables = ["rho", "p", "u", "v"]
        elif equation == "RANS":
            variables = ["rho", "p", "u", "v", "mut"]
        else:
            raise ValueError(f"Unknown equation: {equation}.")

        if "plot_flow" not in self.params:
            raise KeyError(
                "params does not contain 'plot_flow'. "
                "The reference VTK field names must be provided in "
                "params['plot_flow']['fields'] and "
                "params['plot_flow']['components']."
            )

        plot_cfg = self.params["plot_flow"]

        if "fields" not in plot_cfg:
            raise KeyError("params['plot_flow'] does not contain 'fields'.")

        if "components" not in plot_cfg:
            raise KeyError("params['plot_flow'] does not contain 'components'.")

        fields = plot_cfg["fields"]
        components = plot_cfg["components"]

        if len(fields) != len(components):
            raise ValueError(
                "params['plot_flow']['fields'] and "
                "params['plot_flow']['components'] must have the same length. "
                f"Got {len(fields)} fields and {len(components)} components."
            )

        if len(fields) < len(variables):
            raise ValueError(
                "Not enough reference fields were provided in params['plot_flow']. "
                f"Expected at least {len(variables)} fields for equation "
                f"{equation}, but got {len(fields)}."
            )

        for i in range(len(variables)):
            variable = variables[i]
            field_name = fields[i]
            component = components[i]

            # Get data from flow field
            data = np.asarray(self.flowfield.point_data[field_name])
            # Numpy conversion
            reference_value = self.to_numpy_1d(data[:, component])
            # Store value
            self.reference_fields[variable] = reference_value
