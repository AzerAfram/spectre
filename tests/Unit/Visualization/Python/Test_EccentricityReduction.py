# Distributed under the MIT License.
# See LICENSE.txt for details.
# unit test for plot trajectories

import logging
import os
import shutil
import unittest

import numpy as np
from click.testing import CliRunner

import spectre.IO.H5 as spectre_h5
from spectre.Informer import unit_test_build_path, unit_test_src_path
from spectre.support.Logging import configure_logging
from spectre.Visualization.PlotTrajectories import plot_trajectories_command


class TestPlotTrajectories(unittest.TestCase):
    def setUp(self):
        self.test_dir = os.path.join(
            unit_test_build_path(), "Visualization", "PlotTrajectories"
        )
        self.h5_filename = os.path.join(
            self.test_dir, "TestPlotTrajectoriesReductions.h5"
        )
        shutil.rmtree(self.test_dir, ignore_errors=True)
        os.makedirs(self.test_dir, exist_ok=True)
        self.create_h5_file()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def create_h5_file(self):
        """Create the H5"""
        logging.info(f"Creating HDF5 file: {self.h5_filename}")

        # Generate mock-up inspiral data
        nsamples = 100
        dt = 0.02
        x0 = 0.35
        y0 = 0.35
        z0 = 0
        z1 = -9.0e-6
        cosAmp = 7.43
        sinAmp = 7.43
        cosFreq = 0.0173
        sinFreq = 0.0172

        # Define the spirals as functions
        def SpiralA(t):
            return np.array(
                [
                    x0 + cosAmp * np.cos(-cosFreq * t),
                    y0 - sinAmp * np.sin(sinFreq * t),
                    z0 + z1 * (1 - 0.1) * t,
                ]
            )

        def SpiralB(t):
            return np.array(
                [
                    -x0 + cosAmp * np.cos(np.pi + cosFreq * t),
                    -y0 + sinAmp * np.sin(sinFreq * t),
                    z0 + z1 * (1 + 0.1) * t,
                ]
            )

        # Generate time samples
        distances = np.zeros(nsamples)
        tTable = np.arange(0, (nsamples + 1) * dt, dt)
      
        # Compute distances between SpiralA and SpiralB over time
        for i, t in enumerate(times):
            posA = SpiralA(t)
            posB = SpiralB(t)
            distance = np.linalg.norm(posA - posB)
            distances[i] = distance
        
        # Calculate minimum and maximum distances
        r_min = np.min(distances)
        r_max = np.max(distances)
        
        # Compute semi-major axis
        a = (r_max + r_min) / 2
        
        # Compute eccentricity
        self.ecc = (r_max - r_min) / (r_max + r_min)

        # Map time to spiral points
        AhA_data = np.array([[t, *SpiralA(t), *SpiralA(t)] for t in tTable])
        AhB_data = np.array([[t, *SpiralB(t), *SpiralB(t)] for t in tTable])

        with spectre_h5.H5File(self.h5_filename, "w") as h5_file:
            # Insert dataset for AhA
            dataset_AhA = h5_file.insert_dat(
                "ApparentHorizons/ControlSystemAhA_Centers.dat",
                legend=[
                    "Time",
                    "GridCenter_x",
                    "GridCenter_y",
                    "GridCenter_z",
                    "InertialCenter_x",
                    "InertialCenter_y",
                    "InertialCenter_z",
                ],  # Legend for the dataset
                version=0,  # Version number
            )
            # Populate dataset with AhA data
            for data_point in AhA_data:
                dataset_AhA.append(data_point)
            # Close dataset for AhA
            h5_file.close_current_object()

            # Insert dataset for AhB
            dataset_AhB = h5_file.insert_dat(
                "ApparentHorizons/ControlSystemAhB_Centers.dat",
                legend=[
                    "Time",
                    "GridCenter_x",
                    "GridCenter_y",
                    "GridCenter_z",
                    "InertialCenter_x",
                    "InertialCenter_y",
                    "InertialCenter_z",
                ],  # Legend for the dataset
                version=0,  # Version number
            )
            # Populate dataset with AhB data
            for data_point in AhB_data:
                dataset_AhB.append(data_point)
            # Close dataset for AhB
            h5_file.close_current_object()
        logging.info(
            f"Successfully created and populated HDF5 file: {self.h5_filename}"
        )

    def test_cli(self):
        # call the spec ecc reduction function and get its output
        # test to see if the output ecc is similar enough to self.ecc


if __name__ == "__main__":
    configure_logging(log_level=logging.DEBUG)
    unittest.main(verbosity=2)
