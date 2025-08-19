import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mplbasketball.court3d import Court3D, draw_court_3d
from typing import Dict

logger = logging.getLogger(__name__)

def get_court_params(court_type: str = "nba", units: str = "ft", debug: bool = False) -> Dict[str, float]:
    """
    Build and return court parameters directly from mplbasketball's Court3D.
    Strictly uses the measurements provided by the library. No fallbacks.

    Returns:
        {
            "hoop_radius": float,   # radius in the same coordinate units used by your Court3D
        }
    """
    try:
        court = Court3D(court_type=court_type, units=units)
        court_params = court.court_parameters
        
        if 'hoop_radius' not in court_params:
            # We do NOT guess; we do NOT derive from diameter, etc. Hard error.
            raise KeyError(
                "Court3D does not provide 'hoop_radius' in court_parameters. "
                "Available keys: " + str(list(court_params.keys()))
            )
        
        hoop_radius = float(court_params['hoop_radius'])
        
        if debug:
            logger.debug(f"[court] type={court_type} units={units} hoop_radius={hoop_radius}")
        
        return {"hoop_radius": hoop_radius}
    except Exception as e:
        logger.error(f"Error getting court parameters: {e}")
        raise

def validate_court_params(court_params: Dict[str, float]) -> None:
    """
    Validate presence of required measurements. No defaults, no fallbacks.
    """
    missing = [k for k in ("hoop_radius",) if k not in court_params]
    if missing:
        raise KeyError(
            f"court_params missing required key(s): {missing}. "
            "We only use the library-provided measurements; none are inferred."
        )

def get_hoop_position(court_type: str = "nba", units: str = "ft", debug: bool = False) -> (float, float, float):
    """
    Calculate the 3D position of the basketball hoop based on court specifications.

    Parameters:
    - court_type (str): Type of the court ('nba', 'wnba', 'ncaa').
    - units (str): Units of measurement ('ft' or 'm').
    - debug (bool): Flag to enable debug logging.

    Returns:
    - x, y, z (float): Coordinates of the hoop in 3D space.
    """
    try:
        court = Court3D(court_type=court_type, units=units)
        params = court.court_parameters
        # The hoop is located a certain distance from the edge of the court
        x = params['court_dims'][0] / 2 - params['hoop_distance_from_edge']
        y = 0.0  # Centered along the y-axis
        z = params['hoop_height']
        if debug:
            logger.debug(f"Calculated hoop position at (x={x}, y={y}, z={z}) for court type '{court_type}' in '{units}' units.")
        return x, y, z
    except KeyError as e:
        logger.error(f"Key error when accessing court parameters: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_hoop_position: {e}")
        raise


def draw_court(ax: plt.Axes, court_type: str = "nba", units: str = "ft", debug: bool = False) -> None:
    """
    Draw the basketball court and hoops on the given axes.
    Uses ONLY real measurements from mplbasketball. No fallbacks.

    Parameters:
    - ax (plt.Axes): The Matplotlib 3D axis object.
    - court_type (str): Type of the court ('nba', 'wnba', 'ncaa').
    - units (str): Units of measurement ('ft' or 'm').
    - debug (bool): Flag to enable debug logging.

    Returns:
    - None
    """
    try:
        # Draw the court using mplbasketball
        draw_court_3d(ax, court_type=court_type, units=units, origin=np.array([0.0, 0.0]), line_width=2)
        if debug:
            logger.debug("Court drawn successfully using mplbasketball.")

        # Get court parameters using our validated function
        court_params = get_court_params(court_type=court_type, units=units, debug=debug)
        validate_court_params(court_params)
        
        if debug:
            logger.debug(f"Court Parameters in draw_court: {court_params}")

        # Get hoop position
        hoop_x, hoop_y, hoop_z = get_hoop_position(court_type=court_type, units=units, debug=debug)

        # Draw the hoop as a circle using the actual hoop_radius from the library
        hoop_radius = float(court_params["hoop_radius"])
        theta_circle = np.linspace(0, 2 * np.pi, 100)
        hoop_xs = hoop_x + hoop_radius * np.cos(theta_circle)
        hoop_ys = hoop_y + hoop_radius * np.sin(theta_circle)
        hoop_zs = np.full_like(hoop_xs, hoop_z)

        ax.plot(hoop_xs, hoop_ys, hoop_zs, c='orange', lw=3)
        if debug:
            logger.debug(f"Hoop drawn at position ({hoop_x}, {hoop_y}, {hoop_z}) with radius {hoop_radius}.")

        # Get full court parameters for additional features
        court = Court3D(court_type=court_type, units=units)
        full_court_params = court.court_parameters

        # Plot half-court line
        half_court_x = np.linspace(-full_court_params['court_dims'][0]/2, full_court_params['court_dims'][0]/2, 100)
        half_court_y = np.full_like(half_court_x, 0.0)
        half_court_z = np.full_like(half_court_x, 0.0)
        ax.plot(half_court_x, half_court_y, half_court_z, c='black', lw=2, linestyle='--', label='Half-Court Line')
        if debug:
            logger.debug("Half-court line plotted.")

        # Plot sidelines
        sideline_x = np.linspace(-full_court_params['court_dims'][0]/2, full_court_params['court_dims'][0]/2, 100)
        sideline_y_positive = np.full_like(sideline_x, full_court_params['court_dims'][1]/2)
        sideline_z = np.full_like(sideline_x, 0.0)
        ax.plot(sideline_x, sideline_y_positive, sideline_z, c='blue', lw=2, label='Sideline')
        if debug:
            logger.debug("Positive sideline plotted.")

        sideline_y_negative = np.full_like(sideline_x, -full_court_params['court_dims'][1]/2)
        ax.plot(sideline_x, sideline_y_negative, sideline_z, c='blue', lw=2, label='Sideline')
        if debug:
            logger.debug("Negative sideline plotted.")

        # Plot baselines
        baseline_y = np.linspace(-full_court_params['court_dims'][1]/2, full_court_params['court_dims'][1]/2, 100)
        baseline_z = np.full_like(baseline_y, 0.0)
        ax.plot(full_court_params['court_dims'][0]/2, baseline_y, baseline_z, c='green', lw=2, label='Baseline')
        if debug:
            logger.debug("Positive baseline plotted.")

        ax.plot(-full_court_params['court_dims'][0]/2, baseline_y, baseline_z, c='green', lw=2, label='Baseline')
        if debug:
            logger.debug("Negative baseline plotted.")

        if debug:
            logger.debug("Additional court features (half-court, sidelines, baselines) drawn successfully.")
    except KeyError as e:
        logger.error(f"Key error in draw_court: {e}")
        raise
    except Exception as e:
        logger.error(f"Error drawing court or hoop: {e}")
        raise
