# SPDX-License-Identifier: MIT
from pathlib import Path

import pandas as pd

class ObservationData:
    """
    Defines and loads measurement data.
    """
    def __init__(self, params):
        """Initialize an empty observation-data container."""

        # Observation config
        self.config = params["identification"]["observations"]
        self.data_directory = Path(params["paths"]["observations"])
        self.dims = params["geometry"]["dimension"]
        self.data = {}

        # CFD flow field
        self.cfd_directory = Path(params["flow"])

    def load_observation_data(self):

        self.observations = {}

        #if self.config["schlieren"]["enabled"]:
        #    observations["schlieren"] = schlieren_data

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

    file_path = self.data_directory / config["filename"]

    return generate_synthetic_schlieren(
        file_path=file_path,
        density_variable=config["density_variable"],
        dims=self.dims,
    )