# SPDX-License-Identifier: MIT
from pathlib import Path
import pandas as pd

from .schlieren import generate_synthetic_schlieren
from .noise import add_noise

class ObservationData:
    """
    Load and organize observation data for the inverse problem.

    The class loads the enabled observations specified in the
    configuration. Supported observations include synthetic schlieren
    fields, velocity profiles, and pressure-tap measurements.
    """

    def __init__(self, params):
        """
        Initialize the observation-data loader.

        Parameters
        ----------
        params : dict
            Project configuration dictionary.

        """

        self.dims = params["geometry"]["dimension"]

        # Observation config
        self.config = params["identification"]["observations"]
        self.observations_directory = Path(params["paths"]["observations"])
        self.observations = {}

        # CFD flow field
        self.cfd_directory = Path(params["paths"]["flow"])
        self.cfd_filename = params["files"]["flowfield"]

        # Seed (noise generation)
        self.seed = params["seed"]

    def load_observation_data(self):
        """
        Load all enabled observation datasets.

        Returns
        -------
        self.observations : dict
            Observation data organized by observation type.
        """
        self.observations = {}

        if self.config["schlieren"]["enabled"]:
            self.observations["schlieren"] = (
                self._load_schlieren()
            )

        if self.config["velocity_profiles"]["enabled"]:
            self.observations["velocity_profiles"] = (
                self._load_velocity_profiles()
            )

        if self.config["pressure_taps"]["enabled"]:
            self.observations["pressure_taps"] = (
                self._load_pressure_taps()
            )

        return self.observations

    def _load_velocity_profiles(self):
        """Load the velocity-profile measurements.
        
        Returns
        -------
        velocity_profiles : list of dict
            Velocity profiles. Each dictionary contains coordinate
            arrays and one velocity-component array.
        """

        config = self.config["velocity_profiles"]
        n_files = config["n_files"]
        filename = self.config["velocity_profiles"]["filename"]

        components = ["u", "v", "w"][:self.dims]
        coordinates = ["x", "y", "z"][:self.dims]
        
        velocity_profiles = []

        for component in components:
            for index in range(n_files):
                file_path = (self.observations_directory
                             / f"{filename}_{component}_{index}.csv")

                data = pd.read_csv(file_path)

                # coordinates
                profile = {
                    coordinate: data[coordinate].to_numpy(dtype=float)
                    for coordinate in coordinates
                }
                # component
                profile[component] = (
                    data[component].to_numpy(dtype=float)
                )
            
                velocity_profiles.append(profile)

        return velocity_profiles       

    def _load_pressure_taps(self):
        """Load the pressure taps measurements.
        
        Returns
        -------
        pressure_taps : dict
            Pressure-tap coordinates and pressure values.
        """

        config = self.config["pressure_taps"]
        filename = config["filename"]

        file_path = self.observations_directory / f"{filename}.csv"
        data = pd.read_csv(file_path)

        coordinates = ["x", "y", "z"][:self.dims]
        
        pressure_taps = {
            coordinate: data[coordinate].to_numpy(dtype=float)
            for coordinate in coordinates
        }

        pressure_taps["p"] = data["p"].to_numpy(dtype=float)

        return pressure_taps
        
    def _load_schlieren(self):
        """ Generate noisy synthetic schlieren observations.

        Returns
        -------
        schlieren : dict
            Spatial coordinates and the selected synthetic schlieren
            field after applying the configured noise.
        """

        schlieren_config = self.config["schlieren"]
        file_path = self.cfd_directory / self.cfd_filename

        schlieren = generate_synthetic_schlieren(
            file_path=file_path,
            density_name=schlieren_config["density_name"],
            grad_type=schlieren_config["grad_type"],
            dims=self.dims,
        )

        noise_config = schlieren_config["noise"]
        schlieren[schlieren_config["grad_type"]] = add_noise(
            values=schlieren[schlieren_config["grad_type"]],
            noise_type=noise_config["type"],
            level=float(noise_config["level"]),
            seed=self.seed,
            nonnegative=(schlieren_config["grad_type"] == "magnitude")
        )

        return schlieren

    