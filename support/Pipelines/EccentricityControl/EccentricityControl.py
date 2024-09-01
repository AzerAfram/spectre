#!/usr/bin/env python

# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
import os
import sys
from pathlib import Path
from typing import Dict, Literal, Optional, Sequence, Tuple, Union

import click
import h5py
import matplotlib.pyplot as plt
import numpy as np
import rich
import scipy
from scipy import io, optimize

sys.path.append('/home/AzerAfram/SpECTRE/spectre/src/Visualization/Python/')

from Plot import (
    apply_stylesheet_command,
    show_or_save_plot_command,
)
from ReadH5 import available_subfiles

logger = logging.getLogger(__name__)


# Read .dat files from Reductions data: Inertial centers, Mass, Spin
def extract_data_from_file(h5_file, subfile_name, functions, x_axis):
    """Extract data from '.dat' datasets in H5 files"""

    with h5py.File(h5_file, "r") as h5file:
        # Print available subfiles and exit
        if subfile_name is None:
            import rich.columns

            rich.print(
                rich.columns.Columns(
                    available_subfiles(h5file, extension=".dat")
                )
            )
            return

        # Open subfile
        if not subfile_name.endswith(".dat"):
            subfile_name += ".dat"
        dat_file = h5file.get(subfile_name)
        if dat_file is None:
            raise click.UsageError(
                f"Unable to open dat file '{subfile_name}'. Available "
                f"files are:\n {available_subfiles(h5file, extension='.dat')}"
            )

        # Read legend from subfile
        legend = list(dat_file.attrs["Legend"])

        # Select x-axis
        if x_axis is None:
            x_axis = legend[0]
        elif x_axis not in legend:
            raise click.UsageError(
                f"Unknown x-axis '{x_axis}'. Available columns are: {legend}"
            )

        num_obs = len(dat_file[:, legend.index(x_axis)])
        out_table = np.reshape(dat_file[:, legend.index(x_axis)], (num_obs, 1))

        # Assemble in table
        for function, label in functions:
            if function not in legend:
                raise click.UsageError(
                    f"Unknown function '{function}'. "
                    f"Available functions are: {legend}"
                )

            out_table = np.append(
                out_table,
                np.reshape(dat_file[:, legend.index(function)], (num_obs, 1)),
                axis=1,
            )

        return out_table


# Compute separation norm
def compute_separation(h5_file, subfile_name_aha, subfile_name_ahb):
    """Compute coordinate separation"""

    functions = [
        ["InertialCenter_x", "x"],
        ["InertialCenter_y", "y"],
        ["InertialCenter_z", "z"],
    ]
    x_axis = "Time"

    # Extract data
    ObjectA_centers = extract_data_from_file(
        h5_file=h5_file,
        subfile_name=subfile_name_aha,
        functions=functions,
        x_axis=x_axis,
    )

    ObjectB_centers = extract_data_from_file(
        h5_file=h5_file,
        subfile_name=subfile_name_ahb,
        functions=functions,
        x_axis=x_axis,
    )

    if (
        subfile_name_aha is None
        or subfile_name_ahb is None
        or subfile_name_aha == subfile_name_ahb
    ):
        raise click.UsageError(
            f"Dat files '{subfile_name_aha}' and '{subfile_name_aha}' are the"
            " same or at least one of them is missing. Choose files for"
            " different objects. Available files are:\n"
            f" {available_subfiles(h5_file, extension='.dat')}"
        )

    # Compute separation
    num_obs = min(len(ObjectA_centers[:, 0]), len(ObjectB_centers[:, 0]))

    # Separation vector
    separation_vec = (
        ObjectA_centers[:num_obs, 1:] - ObjectB_centers[:num_obs, 1:]
    )

    # Compute separation norm
    separation_norm = np.zeros((num_obs, 2))
    separation_norm[:, 0] = ObjectA_centers[:num_obs, 0]
    separation_norm[:, 1] = np.linalg.norm(separation_vec, axis=1)

    return separation_norm


def compute_time_derivative_of_separation_in_window(data, tmin=None, tmax=None):
    """Compute time derivative of separation on time window"""
    traw = data[:, 0]
    sraw = data[:, 1]

    # Compute separation derivative
    dsdtraw = (sraw[2:] - sraw[0:-2]) / (traw[2:] - traw[0:-2])

    trawcut = traw[1:-1]

    # Apply time window: select values in [tmin, tmax]
    if tmin == None and tmax == None:
        if traw[-1] < 200:
            which_indices = trawcut > 20
        else:
            which_indices = trawcut > 60
    elif tmax == None:
        which_indices = trawcut > tmin
    else:
        which_indices = np.logical_and(trawcut > tmin, trawcut < tmax)

    dsdt = dsdtraw[which_indices]
    t = trawcut[which_indices]

    return t, dsdt


def fit_model(x, y, model):
    """Fit coordinate separation"""
    F = model["function"]
    inparams = model["initial guess"]

    errfunc = lambda p, x, y: F(p, x) - y
    p, success = optimize.leastsq(errfunc, inparams[:], args=(x, y))

    # Compute rms error of fit
    e2 = (errfunc(p, x, y)) ** 2
    rms = np.sqrt(sum(e2) / np.size(e2))

    return dict([("parameters", p), ("rms", rms), ("success", success)])


def compute_coord_sep_updates(
    x, y, model, initial_separation, initial_xcts_values=None
):
    """Compute updates for eccentricity control"""
    fit_results = fit_model(x=x, y=y, model=model)

    amplitude, omega, phase = fit_results["parameters"][:3]

    # Compute updates for Omega and expansion and compute eccentricity
    dOmg = amplitude / 2.0 / initial_separation * np.sin(phase)
    dadot = -amplitude / initial_separation * np.cos(phase)

    # The amplitude could be negative due to degeneracy with phase shifts
    # of pi/2
    # Make sure eccentricity estimate is positive
    eccentricity = np.abs(amplitude / initial_separation / omega)

    fit_results["eccentricity"] = eccentricity
    fit_results["xcts updates"] = dict(
        [("omega update", dOmg), ("expansion update", dadot)]
    )

    # Update xcts parameters if given
    if initial_xcts_values is not None:
        xcts_omega, xcts_expansion = initial_xcts_values
        fit_results["previous xcts values"] = dict(
            [("omega", xcts_omega), ("expansion", xcts_expansion)]
        )
        updated_xcts_omega = xcts_omega + dOmg
        updated_xcts_expansion = xcts_expansion + dadot
        fit_results["updated xcts values"] = dict(
            [
                ("omega", updated_xcts_omega),
                ("expansion", updated_xcts_expansion),
            ]
        )

    return fit_results


def coordinate_separation_eccentricity_control_digest(
    h5_file, x, y, data, functions, fig=None
):
    """Print and plot for eccentricity control"""

    if fig is not None:
        traw = data[:, 0]
        sraw = data[:, 1]
        # Plot coordinate separation
        ((ax1, ax2), (ax3, ax4)) = fig.subplots(2, 2)
        fig.suptitle(
            h5_file,
            color="b",
            size="large",
        )
        ax2.plot(traw, sraw, "k", label="s", linewidth=2)
        ax2.set_title("coordinate separation " + r"$ D $")

        # Plot derivative of coordinate separation
        ax1.plot(x, y, "k", label=r"$ dD/dt $", linewidth=2)
        ax1.set_title(r"$ dD/dt $")

        ax4.set_axis_off()

    logger.info("Eccentricity control summary")

    for name, func in functions.items():
        expression = func["label"]
        rms = func["fit result"]["rms"]

        logger.info(
            f"==== Function fitted to dD/dt: {expression:30s},  rms ="
            f" {rms:4.3g}  ===="
        )

        F = func["function"]
        eccentricity = func["fit result"]["eccentricity"]

        # Print fit parameters
        p = func["fit result"]["parameters"]
        logger.info("Fit parameters:")
        if np.size(p) == 3:
            logger.info(
                f"Oscillatory part: (B, w)=({p[0]:4.3g}, {p[1]:7.4f}), ",
                f"Polynomial part: ({p[2]:4.2g})",
            )
        else:
            logger.info(
                f"Oscillatory part: (B, w, phi)=({p[0]:4.3g}, {p[1]:6.4f},"
                f" {p[2]:6.4f}), "
            )
            if np.size(p) >= 4:
                logger.info(f"Polynomial part: ({p[3]:4.2g}, ")
                for q in p[4:]:
                    logger.info(f"{q:4.2g}")
                logger.info(")")

        # Print eccentricity
        logger.info(f"Eccentricity based on fit: {eccentricity:9.6f}")
        # Print suggested updates based on fit
        xcts_updates = func["fit result"]["xcts updates"]
        dOmg = xcts_updates["omega update"]
        dadot = xcts_updates["expansion update"]
        logger.info("Suggested updates based on fit:")
        logger.info(f"(dOmega, dadot) = ({dOmg:+13.10f}, {dadot:+8.6g})")

        if "updated xcts values" in func["fit result"]:
            xcts_omega = func["fit result"]["updated xcts values"]["omega"]
            xcts_expansion = func["fit result"]["updated xcts values"][
                "expansion"
            ]
            logger.info("Updated Xcts values based on fit:")
            logger.info(
                f"(Omega, adot) = ({(xcts_omega + dOmg):13.10f},"
                f" {(xcts_expansion + dadot):13.10g})"
            )

        # Plot
        if fig is not None:
            errfunc = lambda p, x, y: F(p, x) - y
            # Plot dD/dt
            ax1.plot(
                x,
                F(p, x),
                label=(
                    f"{expression:s} \n rms = {rms:2.1e}, ecc ="
                    f" {eccentricity:4.5f}"
                ),
            )
            ax_handles, ax_labels = ax1.get_legend_handles_labels()

            # Plot residual
            ax3.plot(x, errfunc(p, x, y), label=expression)
            ax3.set_title("Residual")

            ax4.legend(ax_handles, ax_labels)


def coordinate_separation_eccentricity_control(
    h5_file,
    subfile_name_aha: str = "ApparentHorizons/ControlSystemAhA_Centers.dat",
    subfile_name_ahb: str = "ApparentHorizons/ControlSystemAhB_Centers.dat",
    tmin: float = None,
    tmax: float = None,
    angular_velocity_from_xcts: float = None,
    expansion_from_xcts: float = None,
    fig: plt.Figure = None,
):
    """Compute updates based on fits to the coordinate separation for manual
    eccentricity control

    This routine applies a time window. (To avoid large initial transients
    and to allow the user to specify about 2 to 3 orbits of data.)

    Computes the coordinate separations between Objects A and B, as well as
    a finite difference approximation to the time derivative.

    It fits different models to the time derivative. (The amplitude of the
    oscillations is related to the eccentricity.)

    This function returns a dictionary containing data for the fits to all
    the models considered below. For each model, the results of the fit as
    well as the suggested updates for omega and the expansion provided.

    The updates are computed using Newtonian estimates.
    See ArXiv:gr-qc/0702106 and ArXiv:0710.0158 for more details.

    A summary is printed to screen and if a matplotlib figure is provided, a
    plot is generated. The latter is useful to decide between the updates of
    different models (look for small residuals at early times).

    See OmegaDoEccRemoval.py in SpEC for improved eccentricity control.

    """

    data = compute_separation(
        h5_file=h5_file,
        subfile_name_aha=subfile_name_aha,
        subfile_name_ahb=subfile_name_ahb,
    )

    # Get initial separation from data (unwindowed)
    initial_separation = data[:, 1][0]

    # Compute derivative in time window
    t, dsdt = compute_time_derivative_of_separation_in_window(
        data=data, tmin=tmin, tmax=tmax
    )

    # Collect initial xcts values (if given)
    if (
        angular_velocity_from_xcts is not None
        and expansion_from_xcts is not None
    ):
        initial_xcts_values = (
            angular_velocity_from_xcts,
            expansion_from_xcts,
        )
    else:
        initial_xcts_values = None

    # Define functions to model the time derivative of the separation
    functions = dict([])

    # ==== Restricted fit ====
    functions["H1"] = dict(
        [
            ("label", "B*cos(w*t+np.pi/2)+const"),
            (
                "function",
                lambda p, t: p[0] * np.cos(p[1] * t + np.pi / 2) + p[3],
            ),
            ("initial guess", [0, 0.010, np.pi / 2, 0]),
        ]
    )

    # ==== const + cos ====
    functions["H2"] = dict(
        [
            ("label", "B*cos(w*t+phi)+const"),
            ("function", lambda p, t: p[0] * np.cos(p[1] * t + p[2]) + p[3]),
            ("initial guess", [0, 0.010, 0, 0]),
        ]
    )

    # ==== linear + cos ====
    functions["H3"] = dict(
        [
            ("label", "B*cos(w*t+phi)+linear"),
            (
                "function",
                lambda p, t: p[3] + p[4] * t + p[0] * np.cos(p[1] * t + p[2]),
            ),
            (
                "initial guess",
                [
                    0,
                    0.017,
                    0,
                    0,
                    0,
                ],
            ),
        ]
    )

    # ==== quadratic + cos ====
    functions["H4"] = dict(
        [
            ("label", "B*cos(w*t+phi)+quadratic"),
            (
                "function",
                lambda p, t: p[3]
                + p[4] * t
                + p[5] * t**2
                + p[0] * np.cos(p[1] * t + p[2]),
            ),
            (
                "initial guess",
                [
                    0,
                    0.017,
                    0,
                    0,
                    0,
                    0,
                ],
            ),
        ]
    )

    # Fit and compute updates
    for name, func in functions.items():
        # We will handle H4 separately
        if name == "H4":
            continue

        func["fit result"] = compute_coord_sep_updates(
            x=t,
            y=dsdt,
            model=func,
            initial_separation=initial_separation,
            initial_xcts_values=initial_xcts_values,
        )

    # ==== quadratic + cos ====
    # Replace the initial guess with that of the linear solve
    iguess_len = len(functions["H3"]["initial guess"])
    functions["H4"]["initial guess"][0:iguess_len] = functions["H3"][
        "fit result"
    ]["parameters"][0:iguess_len]

    functions["H4"]["fit result"] = compute_coord_sep_updates(
        x=t,
        y=dsdt,
        model=functions["H4"],
        initial_separation=initial_separation,
        initial_xcts_values=initial_xcts_values,
    )

    # Print results and plot
    coordinate_separation_eccentricity_control_digest(
        h5_file=h5_file,
        x=t,
        y=dsdt,
        data=data,
        functions=functions,
        fig=fig,
    )

    return functions

# Input orbital parameters that can be controlled
OrbitalParams = Literal[
    "Omega0",
    "adot0",
    "D0",
]

def init_perform_all_fits(
    h5_files: Sequence[Union[str, Path]],
    id_input_file_path: Union[str, Path],
    subfile_name_aha: str = "ApparentHorizons/ControlSystemAhA_Centers.dat",
    subfile_name_ahb: str = "ApparentHorizons/ControlSystemAhB_Centers.dat",
    tmin: Optional[float] = None,
    tmax: Optional[float] = None,
    target_eccentricity: float = 0.0,
    target_mean_anomaly_fraction: Optional[float] = None,
    plot_output_dir: Optional[Union[str, Path]] = None,
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
            opt_freq_filter=True,
            opt_varpro=True,
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

@click.command(name="eccentricity-control")
@click.argument(
    "h5_file",
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
    help="Input file with initial data parameters.",
)
@click.option(
    "--tmin",
    type=float,
    help=(
        "The lower time bound to start the fit. Used to remove initial junk"
        " and transients in the coordinate separations. Default tmin=20 (or 60)"
        " for tmax<200 (or >200)."
    ),
)
@click.option(
    "--tmax",
    type=float,
    help=(
        "The upper time bound to start the fit. A reasonable value would"
        " include 2-3 orbits."
    ),
)
@click.option(
    "--angular-velocity-from-xcts",
    type=float,
    help="Value of the angular velocity used in the Xcts file.",
)
@click.option(
    "--expansion-from-xcts",
    type=float,
    help="Value of the expansion velocity (adot) used in the Xcts file.",
)
@click.option(
    "-o",
    "--plot-output-dir",
    type=click.Path(writable=True),
    help="Output directory for plots.",
)
@click.option('--flag', 
              is_flag=True, 
              help="If set, the performAllFits() will be used")
@apply_stylesheet_command()
@show_or_save_plot_command()
def eccentricity_control_command(
    h5_file,
    subfile_name_aha,
    subfile_name_ahb,
    id_input_file_path,
    tmin,
    tmax,
    angular_velocity_from_xcts,
    expansion_from_xcts,
    plot_output_dir,
    flag,
):
    """Compute updates based on fits to the coordinate separation for manual
    eccentricity control

    Usage:

    Select an appropriate time window without large initial transients and about
    2 to 3 orbits of data.

    This script uses the coordinate separations between Objects A and B to
    compute a finite difference approximation to the time derivative.
    It then fits different models to it.

    For each model, the suggested updates dOmega and dadot based on Newtonian
    estimates are printed. Note that when all models fit the data adequately,
    their updates are similar. When they differ, examine the output plot to
    find a model that is good fit and has small residuals (especially at early
    times).

    Finally, replace the updated values to the angular velocity and expansion
    parameters (respectively) in the Xcts input file, or use the suggested
    updates to compute them (if the initial xcts parameters were not provided).

    See ArXiv:gr-qc/0702106 and ArXiv:0710.0158 for more details.

    Limitations:

    1) These eccentricity updates work only for non-precessing binaries.
    2) The time window is manually specified by the user.
    3) The coordinate separation is used, instead of the proper distance.

    See OmegaDoEccRemoval.py in SpEC for improved eccentricity control.

    """
    if (flag):
        init_perform_all_fits(
            h5_file,
            subfile_name_aha,
            subfile_name_ahb,
            id_input_file_path,
            tmin,
            tmax,
            plot_output_dir,
        )
        print("peformAllFitsFunc ...")

    else: 
        fig = plt.figure()
        functions = coordinate_separation_eccentricity_control(
            h5_file=h5_file,
            subfile_name_aha=subfile_name_aha,
            subfile_name_ahb=subfile_name_ahb,
            tmin=tmin,
            tmax=tmax,
            angular_velocity_from_xcts=angular_velocity_from_xcts,
            expansion_from_xcts=expansion_from_xcts,
            fig=fig,
        )
        return fig

if __name__ == "__main__":
    eccentricity_control_command(

    )
