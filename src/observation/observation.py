# SPDX-License-Identifier: MIT
from pathlib import Path
import numpy as np
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
        self.data = {}

    def load_observation_data(self):

        observations = {}

        #if self.config["schlieren"]["enabled"]:
        #    observations["schlieren"] = schlieren_data

        if self.config["velocity_profiles"]["enabled"]:
            observations["velocity_profiles"] = (
                self._load_velocity_profiles()
            )

        if self.config["pressure_taps"]["enabled"]:
            observations["pressure_taps"] = (
                self._load_pressure_taps()
            )

        return observations

    def _load_velocity_profiles(self):

        config = self.config["velocity_profiles"]
        components = config["components"]
        n_files = config["n_files"]
        filename = self.config["velocity_profiles"]["filename"]

        velocity_profiles = {
            component: []
            for component in components
        }

        for component in components:
            for j in range(n_files):

                data = pd.read_csv(self.data_directory + 
                                   filename +
                                   component +
                                   '_' +
                                   str(j) +
                                   '.csv')

            profile = {
                "x": data["x"].to_numpy(dtype=float),
                "y": data["y"].to_numpy(dtype=float),
                component: data[component].to_numpy(dtype=float),
            }

            velocity_profiles[component].append(profile)

        return velocity_profiles        

    def _load_pressure_taps(self):

        return pressure_taps


        
