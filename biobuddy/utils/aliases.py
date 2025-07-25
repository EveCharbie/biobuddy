from typing import TypeAlias, Iterable

import numpy as np

Point: TypeAlias = np.ndarray | Iterable[float]
Points: TypeAlias = np.ndarray | Iterable[Point]


def point_to_array(point: Point, name: str = "unknown") -> np.ndarray:
    """
    Convert a point to a numpy array

    Parameters
    ----------
    point
        The point to convert
    name
        The name of the point tu use in the error message is needed

    Returns
    -------
    np.ndarray
        The point as a numpy array
    """
    # TODO: move this to def initialize_point() and return None here instead
    if point is None:
        return np.ndarray((4, 0))

    if not isinstance(point, np.ndarray):
        point = np.array(point)

    if len(point.shape) == 0:
        raise RuntimeError(f"The point {name} must be a np.ndarray of shape (3,) or (4,), but received: {point}")
    error_message = f"The {name} must be a np.ndarray of shape (3,) or (4,), but received: {point.shape}"

    # Check the first dimension
    if len(point.shape) == 1:
        point = point[:, None]
    if point.shape[0] == 3:
        point = np.vstack((point, np.ones(point.shape[1])))
    if point.shape[0] != 4:
        raise RuntimeError(error_message)

    # Check the second dimension
    if len(point.shape) != 2:
        raise RuntimeError(error_message)
    if point.shape[1] != 1:
        raise RuntimeError(error_message)

    return point


def points_to_array(points: Points, name: str) -> np.ndarray:
    """
    Convert a list of points to a numpy array

    Parameters
    ----------
    points
        The points to convert
    name
        The name of the points to use in the error message if needed

    Returns
    -------
    np.ndarray
        The points as a numpy array
    """
    if points is None:
        return np.ndarray((4, 0))

    if isinstance(points, list):
        points = np.array(points)
        if points.shape[0] != 3 and points.shape[0] != 4:
            if points.shape[1] == 3 or points.shape[1] == 4:
                points = points.T
            else:
                raise RuntimeError(
                    f"The {name} must be a list of np.ndarray of shape (3,) or (4,), but received: {points.shape}"
                )

    if isinstance(points, np.ndarray):
        if len(points.shape) == 1:
            points = points[:, None]

        if len(points.shape) != 2 or points.shape[0] not in (3, 4):
            raise RuntimeError(
                f"The {name} must be a np.ndarray of shape (3,), (3, x) (4,) or (4, x), but received: {points.shape}"
            )

        if points.shape[0] == 3:
            points = np.vstack((points, np.ones(points.shape[1])))

        points[3] = 1  # Ensure the last row is all ones
        return points
    else:
        raise RuntimeError(f"The {name} must be a list or np.ndarray, but received: {type(points)}")
