from typing import Callable

import numpy as np

from biobuddy.biomechanical_model_real import BiomechanicalModelReal
from biobuddy.protocols import Data


class InertiaParametersReal:
    def __init__(
        self,
        mass: float = None,
        center_of_mass: np.ndarray = None,
        inertia: np.ndarray = None,
    ):
        """
        Parameters
        ----------
        mass
            The mass of the segment with respect to the full body
        center_of_mass
            The position of the center of mass from the segment coordinate system on the main axis
        inertia
            The inertia xx, yy and zz parameters of the segment
        """
        if not isinstance(mass, float):
            raise RuntimeError(f"The mass must be a float, not {type(mass)}")
        if not isinstance(center_of_mass, np.ndarray):
            raise RuntimeError(f"The center of mass must be a np.ndarray, not {type(center_of_mass)}")
        if center_of_mass.shape != (3, ):
            raise RuntimeError(f"The center of mass must be a np.ndarray of shape (3,) not {center_of_mass.shape}")
        if not isinstance(inertia, np.ndarray):
            raise RuntimeError(f"The inertia must be a np.ndarray, not {type(inertia)}")
        if inertia.shape != (3, 3):
            raise RuntimeError(f"The inertia must be a np.ndarray of shape (3, 3) not {inertia.shape}")

        self.mass = mass
        self.center_of_mass = center_of_mass
        self.inertia = inertia

    @staticmethod
    def from_data(
        data: Data,
        relative_mass: Callable,
        center_of_mass: Callable,
        inertia: Callable,
        kinematic_chain: BiomechanicalModelReal,
        parent_scs: "SegmentCoordinateSystemReal" = None,
    ):
        """
        This is a constructor for the InertiaParameterReal class.

        Parameters
        ----------
        data
            The data to pick the data from
        relative_mass
            The callback function that returns the relative mass of the segment with respect to the full body
        center_of_mass
            The callback function that returns the position of the center of mass
            from the segment coordinate system on the main axis
        inertia
            The callback function that returns the inertia xx, yy and zz parameters of the segment
        kinematic_chain
            The model as it is constructed at that particular time. It is useful if some values must be obtained from
            previously computed values
        parent_scs
            The segment coordinate system of the parent to transform the marker from global to local
        """

        mass = relative_mass(data.values, kinematic_chain)

        p: np.ndarray = center_of_mass(data.values, kinematic_chain)
        if not isinstance(p, np.ndarray):
            raise RuntimeError(f"The function {center_of_mass} must return a np.ndarray of dimension 4xT (XYZ1 x time)")
        if len(p.shape) == 1:
            p = p[:, np.newaxis]

        if len(p.shape) != 2 or p.shape[0] != 4:
            raise RuntimeError(f"The function {center_of_mass} must return a np.ndarray of dimension 4xT (XYZ1 x time)")

        p[3, :] = 1  # Do not trust user and make sure the last value is a perfect one
        com = (parent_scs.transpose if parent_scs is not None else np.identity(4)) @ p
        if np.isnan(com).all():
            raise RuntimeError(f"All the values for {com} returned nan which is not permitted")

        inertia: np.ndarray = inertia(data.values, kinematic_chain)

        return InertiaParametersReal(mass, com, inertia)

    @property
    def to_biomod(self):
        # Define the print function, so it automatically formats things in the file properly
        com = np.nanmean(self.center_of_mass, axis=1)[:3] if len(self.center_of_mass.shape) == 2 else self.center_of_mass[:3]

        out_string = f"\tmass\t{self.mass}\n"
        out_string += f"\tCenterOfMass\t{com[0]:0.5f}\t{com[1]:0.5f}\t{com[2]:0.5f}\n"
        if len(self.inertia.shape) == 2:
            if self.inertia.shape != (3, 3):
                raise ValueError("The inertia matrix must be of shape (3, 3) for a matrix or (3,) for the diagonal elements only.")
            out_string += f"\tinertia\n"
            out_string += f"\t\t{self.inertia[0, 0]}\t{self.inertia[0, 1]}\t{self.inertia[0, 2]}\n"
            out_string += f"\t\t{self.inertia[1, 0]}\t{self.inertia[1, 1]}\t{self.inertia[1, 2]}\n"
            out_string += f"\t\t{self.inertia[2, 0]}\t{self.inertia[2, 1]}\t{self.inertia[2, 2]}\n"
        elif len(self.inertia.shape) == 1:
            if self.inertia.shape != (3,):
                raise ValueError("The inertia matrix must be of shape (3, 3) for a matrix or (3,) for the diagonal elements only.")
            out_string += f"\tinertia_xxyyzz\t{self.inertia[0]}\t{self.inertia[1]}\t{self.inertia[2]}\n"
        return out_string
