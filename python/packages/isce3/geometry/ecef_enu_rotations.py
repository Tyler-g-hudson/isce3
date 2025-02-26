import isce3
import numpy as np


def enu_to_ecef_rotation(lon: float, lat: float) -> isce3.core.Quaternion:
    """
    Get a quaternion to rotate from East, North, Up (ENU) coordinates to Earth-Centered,
    Earth-Fixed (ECEF) coordinates.

    Parameters
    ----------
    lon : float
        The geodetic longitude of the origin of the ENU coordinate system, in radians.
    lat : float
        The geodetic latitude of the origin of the ENU coordinate system, in radians.

    Returns
    -------
    q : isce3.core.Quaternion
        A unit quaternion representing the rotation from ENU to ECEF coordinates.
    """
    # First, rotate clockwise about the East-axis by (pi/2 - lat) radians.
    # Then, rotate clockwise about the z-axis by (pi/2 + lon) radians.
    theta = 0.5 * np.pi - lat
    phi = 0.5 * np.pi + lon
    q1 = isce3.core.Quaternion(angle=theta, axis=(1.0, 0.0, 0.0))
    q2 = isce3.core.Quaternion(angle=phi, axis=(0.0, 0.0, 1.0))
    return q2 * q1


def ecef_to_enu_rotation(lon: float, lat: float) -> isce3.core.Quaternion:
    """
    Get a quaternion to rotate from Earth-Centered, Earth-Fixed (ECEF) coordinates to
    East, North, Up (ENU) coordinates.

    Parameters
    ----------
    lon : float
        The geodetic longitude of the origin of the ENU coordinate system, in radians.
    lat : float
        The geodetic latitude of the origin of the ENU coordinate system, in radians.

    Returns
    -------
    q : isce3.core.Quaternion
        A unit quaternion representing the rotation from ECEF to ENU coordinates.
    """
    # First, rotate counter-clockwise about the z-axis by (pi/2 + lon) radians.
    # Then, rotate counter-clockwise about the East-axis by (pi/2 - lat) radians.
    theta = 0.5 * np.pi - lat
    phi = 0.5 * np.pi + lon
    q1 = isce3.core.Quaternion(angle=-theta, axis=(1.0, 0.0, 0.0))
    q2 = isce3.core.Quaternion(angle=-phi, axis=(0.0, 0.0, 1.0))
    return q2 * q1
