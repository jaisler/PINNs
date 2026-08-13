# SPDX-License-Identifier: MIT

from pathlib import Path
import numpy as np
import torch

from src.utils import compute_metrics as compute_scalar_metrics
from src.utils import print_metrics_table

class FlowFieldPostProcessor:
    """Compare predicted and reference flow fields on a CFD mesh."""

    def __init__(
        self,
        model,
        flowfield,
        params,
        eps=1.0e-12,
    ):
        """Initialize mesh-based post-processing.

        Parameters
        ----------
        model : PhysicsInformedNN
            Trained flow model.
        flowfield : pyvista.DataSet
            CFD mesh containing reference point fields.
        params : dict
            PIRFlow configuration.
        eps : float, optional
            Small denominator safeguard.
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

        print("---------------------------------------")
        print("Postprocessing flowfields ...")

    def run(self, prefix="flowfield"):
        """Run prediction, comparison, metrics, and VTK export.

        Parameters
        ----------
        prefix : str
            Output filename prefix.

        Returns
        -------
        None
            Results are stored on this object and written to disk.
        """

        # Get the predict flow field on the mesh points
        self.predict_on_mesh()
        # Extract reference flow field from CFD simulation
        self.extract_reference_fields()
        # Compute the error between the prediction and simulation flow field
        self.compute_error_fields()
        
        # Write comparison flow fields to vtk file
        # Ref, pred, error
        self.write_comparison_vtk(prefix)

        # Compute the metrics
        self.compute_metrics()
        # Print metrics
        self.print_metrics()

    def predict_on_mesh(self):
        """Evaluate the trained model at every CFD mesh point.

        Returns
        -------
        None
            Predictions are stored in ``predicted_fields``.
        """

        xmesh = self.flowfield.points[:, 0]
        ymesh = self.flowfield.points[:, 1]

        equation = self.params["run"]["equation"]

        if equation == "euler":
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

        elif equation == "rans":
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
        """Extract configured reference fields from mesh point data.

        Returns
        -------
        None
            Reference arrays are stored in ``reference_fields``.
        """

        equation = self.params["run"]["equation"]

        if equation == "euler":
            variables = ["rho", "p", "u", "v"]
        elif equation == "rans":
            variables = ["rho", "p", "u", "v", "mut"]
        else:
            raise ValueError(f"Unknown equation: {equation}.")

        if "post_processing" not in self.params:
            raise KeyError(
                "params does not contain 'post_processing'. "
                "The reference VTK field names must be provided in "
                "params['post_processing']['fields'] and "
                "params['post_processing']['components']."
            )

        plot_cfg = self.params["post_processing"]

        if "fields" not in plot_cfg:
            raise KeyError("params['post_processing'] does not contain 'fields'.")

        if "components" not in plot_cfg:
            raise KeyError("params['post_processing'] does not contain 'components'.")

        fields = plot_cfg["fields"]
        components = plot_cfg["components"]

        if len(fields) != len(components):
            raise ValueError(
                "params['post_processing']['fields'] and "
                "params['post_processing']['components'] must have the same length. "
                f"Got {len(fields)} fields and {len(components)} components."
            )

        if len(fields) < len(variables):
            raise ValueError(
                "Not enough reference fields were provided in params['post_processing']. "
                f"Expected at least {len(variables)} fields for equation "
                f"{equation}, but got {len(fields)}."
            )

        for i in range(len(variables)):
            variable = variables[i]
            field_name = fields[i]
            component = components[i]

            # Get data from flow field
            data = np.asarray(self.flowfield.point_data[field_name])

            # Scalar field, e.g. Density or Pressure: shape (N,)
            if data.ndim == 1:
                reference_value = data

            # Vector field, e.g. Velocity: shape (N, n_components)
            elif data.ndim == 2:
                if component >= data.shape[1]:
                    raise ValueError(
                        f"Field '{field_name}' has {data.shape[1]} components, "
                        f"but component {component} was requested."
                    )

                reference_value = data[:, component]

            else:
                raise ValueError(
                    f"Field '{field_name}' has unsupported shape {data.shape}."
                )

            # Convert to 1D NumPy array
            reference_value = self.to_numpy_1d(reference_value)
            # Store value
            self.reference_fields[variable] = reference_value

    def compute_error_fields(self):
        """Compute pointwise raw, absolute, relative, and scaled errors.

        Returns
        -------
        dict
            Error arrays keyed by variable and error type.
        """

        for name in self.predicted_fields:
            if name not in self.reference_fields:
                raise KeyError(
                    f"Predicted field '{name}' has no matching reference field."
                )

            pred = self.predicted_fields[name]
            ref = self.reference_fields[name]

            error = pred - ref
            # Absolute error
            abs_error = np.abs(error)
            # Relative error
            rel_error = abs_error / (np.abs(ref) + self.eps)
            # Scaled absolute error
            abs_error_01 = self.scale_01(abs_error)

            self.error_fields[f"{name}_error"] = error
            self.error_fields[f"{name}_abs_error"] = abs_error
            self.error_fields[f"{name}_rel_error"] = rel_error
            self.error_fields[f"{name}_abs_error_01"] = abs_error_01

        return self.error_fields
    
    def compute_metrics(self):
        """Compute scalar metrics for every predicted mesh field.

        Returns
        -------
        None
            Results are stored in ``metrics``.
        """

        for name in self.predicted_fields:
            pred_np = self.predicted_fields[name]
            ref_np = self.reference_fields[name]

            pred_torch = torch.tensor(
                pred_np,
                dtype=torch.float32,
            ).reshape(-1, 1)

            ref_torch = torch.tensor(
                ref_np,
                dtype=torch.float32,
            ).reshape(-1, 1)

            self.metrics[name] = compute_scalar_metrics(
                pred_torch,
                ref_torch,
                eps=self.eps,
            )

    def print_metrics(self):
        """Print the mesh-based metric table.

        Returns
        -------
        None
            Metrics are written to standard output.
        """

        print_metrics_table(
            self.metrics,
            title="Mesh-based post-processing metrics",
            rel_l2_percent=True,
        )

    def write_comparison_vtk(self, prefix):
        """Write reference, prediction, and error fields to one VTK file.

        Parameters
        ----------
        prefix : str
            Output filename prefix.

        Returns
        -------
        pathlib.Path
            Path of the written VTK file.
        """

        mesh = self.flowfield.copy(deep=True)

        for name in self.predicted_fields:
            mesh.point_data[f"{name}_ref"] = self.reference_fields[name]
            mesh.point_data[f"{name}_pred"] = self.predicted_fields[name]

        for name in self.error_fields:
            mesh.point_data[name] = self.error_fields[name]

        filename = self.output_dir / f"{prefix}_comparison.vtk"
        mesh.save(filename)

        print("---------------------------------------")
        print(f"Comparison flowfield written to: {filename}")

        return filename
    
    @staticmethod
    def to_numpy_1d(value):
        """Convert an array-like value to one-dimensional NumPy data.

        Parameters
        ----------
        value : array_like or torch.Tensor
            Value to convert.

        Returns
        -------
        numpy.ndarray
            Flattened array.
        """

        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()

        value = np.asarray(value)

        return value.reshape(-1)

    @staticmethod
    def scale_01(value):
        """Min-max scale an array to ``[0, 1]``.

        Parameters
        ----------
        value : array_like
            Values to scale.

        Returns
        -------
        numpy.ndarray
            Scaled values, or zeros for a constant field.
        """

        value = np.asarray(value)

        value_min = np.nanmin(value)
        value_max = np.nanmax(value)

        denominator = value_max - value_min

        if denominator < 1.0e-30:
            return np.zeros_like(value)

        return (value - value_min) / denominator
    
    def get_predicted_fields(self):
        """Return predicted mesh fields.

        Returns
        -------
        dict
            Predicted arrays keyed by variable.
        """
        return self.predicted_fields

    def get_reference_fields(self):
        """Return reference CFD fields.

        Returns
        -------
        dict
            Reference arrays keyed by variable.
        """
        return self.reference_fields

    def get_error_fields(self):
        """Return pointwise error fields.

        Returns
        -------
        dict
            Error arrays keyed by variable and error type.
        """
        return self.error_fields
