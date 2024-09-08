#!/usr/bin/env python

# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
from pathlib import Path
from typing import Dict, Literal, Optional, Sequence, Tuple, Union

import click
import h5py
import numpy as np
import pandas as pd
import yaml

from spectre.Visualization.PlotTrajectories import import_A_and_B
from spectre.Visualization.ReadH5 import to_dataframe

logger = logging.getLogger(__name__)

# Input orbital parameters that can be controlled
OrbitalParams = Literal[
    "Omega0",
    "adot0",
    "D0",
]


def eccentricity_control(
    h5_files: Sequence[Union[str, Path]],
    id_input_file_path: Union[str, Path],
    subfile_name_aha: str = "ApparentHorizons/ControlSystemAhA_Centers.dat",
    subfile_name_ahb: str = "ApparentHorizons/ControlSystemAhB_Centers.dat",
    tmin: Optional[float] = None,
    tmax: Optional[float] = None,
    target_eccentricity: float = 0.0,
    target_mean_anomaly_fraction: Optional[float] = None,
    plot_output_dir: Optional[Union[str, Path]] = None,
    opt_freq_filter: bool = True,
    opt_varpro: bool = True,
) -> Tuple[float, float, Dict[OrbitalParams, float]]:
    """Compute eccentricity and new orbital parameters for a binary system

    The eccentricity is estimated from the trajectories of the binary objects
    and new orbital parameters are suggested to drive the orbit to the target
    eccentricity, using SpEC's OmegaDotEccRemoval.py. Currently supports only
    circular orbits (eccentricity = 0).

    Arguments:
      h5_files: Paths to the H5 files containing the trajectory data (e.g.
        BbhReductions.h5).
      id_input_file_path: Path to the initial data input file from which the
        evolution started. This file contains the initial data parameters that
        are being controlled.
      subfile_name_aha: (Optional) Name of the subfile containing the apparent
        horizon centers for object A.
      subfile_name_ahb: (Optional) Name of the subfile containing the apparent
        horizon centers for object B.
      tmin: (Optional) The lower time bound for the eccentricity estimate.
        Used to remove initial junk and transients in the data.
      tmax: (Optional) The upper time bound for the eccentricity estimate.
        A reasonable value would include 2-3 orbits.
      target_eccentricity: (Optional) The target eccentricity to drive the
        orbit to. Default is 0.0 (circular orbit).
      target_mean_anomaly_fraction: (Optional) The target mean anomaly of the
        orbit divided by 2 pi, so it is a number between 0 and 1. The value 0
        corresponds to the pericenter of the orbit (closest approach), the value
        0.5 corresponds to the apocenter of the orbit (farthest distance), and
        the value 1 corresponds to the pericenter again. Currently this is
        unused because only an eccentricity of 0 is supported.
      plot_output_dir: (Optional) Output directory for plots.
      opt_freq_filter: (Optional) Whether to apply frequency filter.
      opt_varpro: (Optional) Whether to apply variable projection.

    Returns:
        Tuple of eccentricity, eccentricity error, and dictionary of updated
        orbital parameters.
    """
    # Import from SpEC
    from OmegaDotEccRemoval import ComputeOmegaAndDerivsFromFile, performAllFits

    # Read initial data parameters from input file
    with open(id_input_file_path, "r") as open_input_file:
        _, id_input_file = yaml.safe_load_all(open_input_file)
    id_binary = id_input_file["Background"]["Binary"]
    Omega0 = id_binary["AngularVelocity"]
    adot0 = id_binary["Expansion"]
    D0 = id_binary["XCoords"][1] - id_binary["XCoords"][0]

    # Load trajectory data
    traj_A, traj_B = import_A_and_B(
        h5_files, subfile_name_aha, subfile_name_ahb
    )
    if tmin is not None:
        traj_A = traj_A[traj_A[:, 0] >= tmin]
        traj_B = traj_B[traj_B[:, 0] >= tmin]
    if tmax is not None:
        traj_A = traj_A[traj_A[:, 0] <= tmax]
        traj_B = traj_B[traj_B[:, 0] <= tmax]

    # Load horizon parameters from evolution data at reference time (tmin)
    def get_horizons_data(reductions_file):
        with h5py.File(reductions_file, "r") as open_h5file:
            horizons_data = []
            for ab in "AB":
                ah_subfile = open_h5file.get(f"ObservationAh{ab}.dat")
                if ah_subfile is not None:
                    horizons_data.append(
                        to_dataframe(ah_subfile)
                        .set_index("Time")
                        .add_prefix(f"Ah{ab} ")
                    )
            if not horizons_data:
                return None
            return pd.concat(horizons_data, axis=1)

    horizon_params = pd.concat(map(get_horizons_data, h5_files))
    if tmin is not None:
        horizon_params = horizon_params[horizon_params.index >= tmin]
    if horizon_params.empty:
        mA = id_binary["ObjectRight"]["KerrSchild"]["Mass"]
        mB = id_binary["ObjectLeft"]["KerrSchild"]["Mass"]
        sA = sB = None
        # raise ValueError("No horizon data found in time range.")
    else:
        mA = horizon_params["AhA ChristodoulouMass"].iloc[0]
        mB = horizon_params["AhB ChristodoulouMass"].iloc[0]
        sA = [
            horizon_params[f"AhA DimensionfulSpinVector_{xyz}"] for xyz in "xyz"
        ]
        sB = [
            horizon_params[f"AhB DimensionfulSpinVector_{xyz}"] for xyz in "xyz"
        ]

    # Call into SpEC's OmegaDotEccRemoval.py
    t, Omega, dOmegadt, OmegaVec = ComputeOmegaAndDerivsFromFile(traj_A, traj_B)
    eccentricity, delta_Omega0, delta_adot0, delta_D0, ecc_std_dev, _ = (
        performAllFits(
            XA=traj_A,
            XB=traj_B,
            t=t,
            Omega=Omega,
            dOmegadt=dOmegadt,
            OmegaVec=OmegaVec,
            mA=mA,
            mB=mB,
            sA=sA,
            sB=sB,
            IDparam_omega0=Omega0,
            IDparam_adot0=adot0,
            IDparam_D0=D0,
            tmin=tmin,
            tmax=tmax,
            tref=tmin,
            opt_freq_filter=opt_freq_filter,
            opt_varpro=opt_varpro,
            opt_type="bbh",
            opt_tmin=tmin,
            opt_improved_Omega0_update=True,
            check_periastron_advance=True,
            plot_output_dir=plot_output_dir,
            Source="",
        )
    )
    logger.info(
        f"Eccentricity estimate is {eccentricity:g} +/- {ecc_std_dev:e}."
        " Update orbital parameters as follows"
        f" for target eccentricity {target_eccentricity}:\n"
        f"Omega0 += {delta_Omega0:e} -> {Omega0 + delta_Omega0:g}\n"
        f"adot0 += {delta_adot0:e} -> {adot0 + delta_adot0:e}\n"
        f"D0 += {delta_D0:e} -> {D0 + delta_D0:g}"
    )
    return (
        eccentricity,
        ecc_std_dev,
        {
            "Omega0": Omega0 + delta_Omega0,
            "adot0": adot0 + delta_adot0,
            "D0": D0 + delta_D0,
        },
    )


@click.command(name="eccentricity-control", help=eccentricity_control.__doc__)
@click.argument(
    "h5_files",
    nargs=-1,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
)
@click.option(
    "--subfile-name-aha",
    "-A",
    default="ApparentHorizons/ControlSystemAhA_Centers.dat",
    show_default=True,
    help=(
        "Name of subfile containing the apparent horizon centers for object A."
    ),
)
@click.option(
    "--subfile-name-ahb",
    "-B",
    default="ApparentHorizons/ControlSystemAhB_Centers.dat",
    show_default=True,
    help=(
        "Name of subfile containing the apparent horizon centers for object B."
    ),
)
@click.option(
    "--id-input-file",
    "-i",
    "id_input_file_path",
    required=True,
    help="Input file with initial data parameters.",
)
@click.option(
    "--tmin",
    type=float,
    help=(
        "The lower time bound for the eccentricity estimate. Used to remove"
        " initial junk and transients in the data."
    ),
)
@click.option(
    "--tmax",
    type=float,
    help=(
        "The upper time bound for the eccentricity estimate. A reasonable value"
        " would include 2-3 orbits."
    ),
)
@click.option(
    "-o",
    "--plot-output-dir",
    type=click.Path(writable=True),
    help="Output directory for plots.",
)
@click.option(
    "--opt-freq-filter",
    default=True,
    help="Whether to apply frequency filter.",
)
@click.option(
    "--opt-varpro",
    default=True,
    help="Whether to apply variable projection.",
)
def eccentricity_control_command(**kwargs):
    _rich_traceback_guard = True  # Hide traceback until here
    eccentricity_control(**kwargs)
