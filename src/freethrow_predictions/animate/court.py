import logging
import numpy as np
import matplotlib.pyplot as plt
from mplbasketball.court3d import Court3D, draw_court_3d

logger = logging.getLogger(__name__)

# ---------- NEW HELPERS ----------

def _resolve_param(d: dict, *aliases: str, required: bool = False, name_for_error: str | None = None):
    """Return the first existing key from aliases. If required and none found, raise with available keys."""
    for key in aliases:
        if key in d:
            return d[key]
    if required:
        available = ", ".join(sorted(d.keys()))
        raise KeyError(
            f"Missing required court parameter {name_for_error or aliases[0]!r}. "
            f"Available keys: {available}"
        )
    return None


def _resolve_hoop_specs(court_params: dict, require: bool) -> tuple[float | None, float | None]:
    """
    Return (ring_inside_diameter, ring_height) if available.
    We DO NOT invent values; we only read from API with multiple aliases. If 'require' is True and
    diameter is not present, we raise. Otherwise we return (None, height_or_None).
    """
    # Height (commonly present)
    ring_h = _resolve_param(
        court_params,
        "hoop_height", "ring_height", "rim_height", "basket_ring_height",
        required=False, name_for_error="ring height"
    )

    # Try to get diameter first, then radius
    ring_d = _resolve_param(
        court_params,
        # diameter names
        "hoop_diameter", "rim_diameter", "ring_inside_diameter", "basket_ring_inside_diameter",
        "basket_ring_diameter", "goal_ring_diameter",
        required=False, name_for_error="ring inside diameter"
    )
    
    # If no diameter found, try radius and convert to diameter
    if ring_d is None:
        ring_r = _resolve_param(
            court_params,
            # radius names
            "hoop_radius", "rim_radius", "ring_radius", "basket_ring_radius",
            required=require, name_for_error="ring radius"
        )
        if ring_r is not None:
            ring_d = ring_r * 2.0  # Convert radius to diameter
    
    return ring_d, ring_h

# ---------- CHANGED FUNCTIONS ----------

def get_hoop_position(court_type: str = "nba", units: str = "ft", debug: bool = False) -> tuple[float, float, float]:
    """
    Compute hoop center using the API's court parameters. We rely on API-provided
    dimensions; we do not assume hard-coded numbers.
    """
    court = Court3D(court_type=court_type, units=units)
    params = court.court_parameters
    if debug:
        logger.debug(f"[get_hoop_position] court_parameters keys: {sorted(list(params.keys()))}")

    # Court dimensions - try to get court_dims tuple first, then individual parameters
    court_dims = _resolve_param(params, "court_dims", required=False)
    if court_dims is not None:
        court_len = court_dims[0]  # First element is length
        court_wid = court_dims[1]  # Second element is width
    else:
        # Fallback to individual parameters
        court_len = _resolve_param(params, "court_length", "court_dims_x", "court_dims_len", "court_dims_length", required=True, name_for_error="court length")
        court_wid = _resolve_param(params, "court_width", "court_dims_y", "court_dims_width", required=True, name_for_error="court width")

    # Distance of hoop center from the baseline/endline:
    # try several plausible names; require=True because we must know x
    dist_from_end = _resolve_param(
        params,
        "hoop_distance_from_edge", "rim_center_from_endline", "basket_center_from_endline",
        "ring_center_from_endline", "goal_center_from_endline",
        required=True, name_for_error="rim center from endline"
    )

    # y is centered by default; z (height) is informative but not strictly required for center
    _, ring_h = _resolve_hoop_specs(params, require=False)

    x = (court_len / 2.0) - float(dist_from_end)
    y = 0.0
    z = float(ring_h) if ring_h is not None else 0.0

    if debug:
        logger.debug(f"[get_hoop_position] court_len={court_len}, court_wid={court_wid}, dist_from_end={dist_from_end} → hoop=({x},{y},{z})")
    return x, y, z


def draw_court(
    ax: plt.Axes,
    court_type: str = "nba",
    units: str = "ft",
    debug: bool = False,
    draw_custom_hoop: str = "auto",  # "auto" | True | False
) -> None:
    """
    Draw the court via mplbasketball first. Optionally overlay a custom hoop ring *only if*
    the API exposes a ring diameter. We do NOT hard-code a diameter.
    """
    # 1) Let the library draw its court
    draw_court_3d(ax, court_type=court_type, units=units, origin=np.array([0.0, 0.0]), line_width=2)
    if debug:
        logger.debug("[draw_court] base court drawn.")

    court = Court3D(court_type=court_type, units=units)
    params = court.court_parameters
    if debug:
        logger.debug(f"[draw_court] court_parameters keys: {sorted(list(params.keys()))}")

    # 2) Decide whether we'll overlay our own ring
    must = (draw_custom_hoop is True)
    try:
        ring_diam, ring_h = _resolve_hoop_specs(params, require=must)
    except KeyError as e:
        logger.error(f"[draw_court] required hoop param missing: {e}")
        raise

    if draw_custom_hoop == "auto" and ring_diam is None:
        if debug:
            logger.debug("[draw_court] No ring diameter from API; skipping custom hoop overlay (auto mode).")
        return

    # 3) Compute hoop center
    hoop_x, hoop_y, hoop_z = get_hoop_position(court_type=court_type, units=units, debug=debug)

    # 4) Draw the ring only if we have a diameter
    if ring_diam is not None:
        hoop_radius = float(ring_diam) / 2.0

        # Optional sanity check (log only; do not convert units automatically)
        court_dims = _resolve_param(params, "court_dims", required=False)
        if court_dims is not None:
            court_len = court_dims[0]
        else:
            court_len = _resolve_param(params, "court_length", "court_dims_x", "court_dims_len", "court_dims_length", required=False)
        
        if court_len and hoop_radius > (court_len * 0.2):
            logger.warning(
                f"[draw_court] ring radius ({hoop_radius}) seems large relative to court length ({court_len}). "
                "Verify units coming from the API."
            )

        theta = np.linspace(0, 2 * np.pi, 100)
        xs = hoop_x + hoop_radius * np.cos(theta)
        ys = hoop_y + hoop_radius * np.sin(theta)
        zs = np.full_like(xs, hoop_z if ring_h is not None else hoop_z)

        ax.plot(xs, ys, zs, c="orange", lw=3)
        if debug:
            logger.debug(f"[draw_court] custom hoop overlay at ({hoop_x},{hoop_y},{hoop_z}); radius={hoop_radius}")
