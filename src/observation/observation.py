# SPDX-License-Identifier: MIT
from pathlib import Path
import pandas as pd

from .schlieren import generate_synthetic_schlieren
from .noise import add_noise

class ObservationData:
    """ Defines and loads measurement data."""
    def __init__(self, params):
        """Initialize an empty observation-data container."""

        # Observation config
        self.config = params["identification"]["observations"]
        self.data_directory = Path(params["paths"]["observations"])
        self.data = {}

        self.dims = params["geometry"]["dimension"]
        if self.dims not in (1, 2, 3):
            raise ValueError(f"Invalid problem dimension: {self.dims}")

        # CFD flow field
        self.cfd_directory = Path(params["flow"])
        self.cfd_filename = params["files"]["flowfield"]

    def load_observation_data(self):

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
        """Load the velocity-profile measurements."""

        config = self.config["velocity_profiles"]
        n_files = config["n_files"]
        filename = self.config["velocity_profiles"]["filename"]

        components = ["u", "v", "w"][:self.dims]
        coordinates = ["x", "y", "z"][:self.dims]
        
        velocity_profiles = []

        for component in components:
            for index in range(n_files):
                file_path = (self.data_directory
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
        """Load the pressure taps measurements."""

        config = self.config["pressure_taps"]
        filename = config["filename"]

        file_path = self.data_directory / f"{filename}.csv"
        data = pd.read_csv(file_path)

        coordinates = ["x", "y", "z"][:self.dims]
        
        pressure_taps = {
            coordinate: data[coordinate].to_numpy(dtype=float)
            for coordinate in coordinates
        }

        pressure_taps["p"] = data["p"].to_numpy(dtype=float)

        return pressure_taps
        
def _load_schlieren(self):
    """Generate schlieren observations from a CFD density field."""

    config = self.config["schlieren"]

    file_path = self.cfd_directory / f"{self.cfd_filename}.csv"

    schlieren = generate_synthetic_schlieren(
        file_path=file_path,
        density_name=config["density_name"],
        gradient_type=config["grad_type"],
        dims=self.dims,
    )

    if not config["noise"]: 
        return add_noise(schlieren)
    else:
        return schlieren

    