from typing import Callable

import numpy as np

from ..biomechanical_model_real import BiomechanicalModelReal
from ....utils.protocols import Data
from ....utils.aliases import point_to_array


class MeshFileReal:
    def __init__(
        self,
        mesh_file_name: str,
        mesh_color: np.ndarray[float] = None,
        mesh_scale: np.ndarray[float] = None,
        mesh_rotation: np.ndarray[float] = None,
        mesh_translation: np.ndarray[float] = None,
    ):
        """
        Parameters
        ----------
        mesh_file_name
            The name of the mesh file
        mesh_color
            The color the mesh should be displayed in (RGB)
        mesh_scale
            The scaling that must be applied to the mesh (XYZ)
        mesh_rotation
            The rotation that must be applied to the mesh (Euler angles: XYZ)
        mesh_translation
            The translation that must be applied to the mesh (XYZ)
        """
        # TODO: create the def initialize_point(), to skip the if is None
        self.mesh_file_name = mesh_file_name
        self.mesh_color = mesh_color
        self.mesh_scale = None if mesh_scale is None else point_to_array(mesh_scale, "mesh_scale")
        self.mesh_rotation = None if mesh_rotation is None else point_to_array(mesh_rotation, "mesh_rotation")
        self.mesh_translation = (
            None if mesh_translation is None else point_to_array(mesh_translation, "mesh_translation")
        )

    @property
    def mesh_file_name(self) -> str:
        return self._mesh_file_name

    @mesh_file_name.setter
    def mesh_file_name(self, value: str):
        self._mesh_file_name = value

    @property
    def mesh_color(self) -> np.ndarray[float]:
        return self._mesh_color

    @mesh_color.setter
    def mesh_color(self, value: np.ndarray[float]):
        self._mesh_color = value

    @property
    def mesh_scale(self) -> np.ndarray:
        if self._mesh_scale is None:
            return np.ones((4, 1))
        else:
            return self._mesh_scale

    @mesh_scale.setter
    def mesh_scale(self, value: np.ndarray[float]):
        self._mesh_scale = None if value is None else point_to_array(value, "mesh_scale")

    @property
    def mesh_rotation(self) -> np.ndarray:
        if self._mesh_rotation is None:
            return np.zeros((4, 1))
        else:
            return self._mesh_rotation

    @mesh_rotation.setter
    def mesh_rotation(self, value: np.ndarray[float]):
        self._mesh_rotation = None if value is None else point_to_array(value, "mesh_rotation")

    @property
    def mesh_translation(self) -> np.ndarray:
        if self._mesh_translation is None:
            return np.zeros((4, 1))
        else:
            return self._mesh_translation

    @mesh_translation.setter
    def mesh_translation(self, value: np.ndarray[float]):
        self._mesh_translation = None if value is None else point_to_array(value, "mesh_translation")

    @staticmethod
    def from_data(
        data: Data,
        model: BiomechanicalModelReal,
        mesh_file_name: str,
        mesh_color: np.ndarray[float] | list[float] | tuple[float] = None,
        scaling_function: Callable = None,
        rotation_function: Callable = None,
        translation_function: Callable = None,
    ):
        """
        This is a constructor for the MeshFileReal class. It evaluates the functions that defines the mesh file to get
        actual characteristics of the mesh

        Parameters
        ----------
        data
            The data to pick the data from
        mesh_file_name
            The name of the mesh file
        mesh_color
            The color the mesh should be displayed in (RGB)
        scaling_function
            The function (f(m) -> np.ndarray, where m is a dict of markers (XYZ1 x time)) that defines the scaling
        rotation_function
            The function (f(m) -> np.ndarray, where m is a dict of markers (XYZ1 x time)) that defines the rotation
        translation_function
            The function (f(m) -> np.ndarray, where m is a dict of markers (XYZ1 x time)) that defines the translation
        """

        if not isinstance(mesh_file_name, str):
            raise RuntimeError("The mesh_file_name must be a string")

        if mesh_color is not None:
            mesh_color = np.array(mesh_color)
            if mesh_color.shape == (3, 1):
                mesh_color = mesh_color.reshape((3,))
            elif mesh_color.shape != (3,):
                raise RuntimeError("The mesh_color must be a vector of dimension 3 (RGB)")

        if scaling_function is None:
            mesh_scale = np.array([1, 1, 1])
        else:
            mesh_scale: np.ndarray = scaling_function(data.values, model)
            if not isinstance(mesh_scale, np.ndarray):
                raise RuntimeError(f"The scaling_function {scaling_function} must return a vector of dimension 3 (XYZ)")
            if mesh_scale.shape == (3, 1):
                mesh_scale = mesh_scale.reshape((3,))
            elif mesh_scale.shape != (3,):
                raise RuntimeError(f"The scaling_function {scaling_function} must return a vector of dimension 3 (XYZ)")

        if rotation_function is None:
            mesh_rotation = np.array([0, 0, 0])
        else:
            mesh_rotation: np.ndarray = rotation_function(data.values, model)
            if not isinstance(mesh_rotation, np.ndarray):
                raise RuntimeError(
                    f"The rotation_function {rotation_function} must return a vector of dimension 3 (XYZ)"
                )
            if mesh_rotation.shape == (3, 1):
                mesh_rotation = mesh_rotation.reshape((3,))
            elif mesh_rotation.shape != (3,):
                raise RuntimeError(
                    f"The rotation_function {rotation_function} must return a vector of dimension 3 (XYZ)"
                )

        if translation_function is None:
            mesh_translation = np.array([0, 0, 0])
        else:
            mesh_translation: np.ndarray = translation_function(data.values, model)
            if not isinstance(mesh_translation, np.ndarray):
                raise RuntimeError(
                    f"The translation_function {translation_function} must return a vector of dimension 3 (XYZ)"
                )
            if mesh_translation.shape == (3, 1):
                mesh_translation = mesh_translation.reshape((3,))
            elif mesh_translation.shape != (3,):
                raise RuntimeError(
                    f"The translation_function {translation_function} must return a vector of dimension 3 (XYZ)"
                )

        return MeshFileReal(
            mesh_file_name=mesh_file_name,
            mesh_color=mesh_color,
            mesh_scale=mesh_scale,
            mesh_rotation=mesh_rotation,
            mesh_translation=mesh_translation,
        )

    def to_biomod(self):
        # Define the print function, so it automatically formats things in the file properly
        out_string = ""
        out_string += f"\tmeshfile\t{self.mesh_file_name}\n"
        if self.mesh_color is not None:
            out_string += f"\tmeshcolor\t{self.mesh_color[0]}\t{self.mesh_color[1]}\t{self.mesh_color[2]}\n"
        if self.mesh_scale is not None:
            out_string += f"\tmeshscale\t{self.mesh_scale[0, 0]}\t{self.mesh_scale[1, 0]}\t{self.mesh_scale[2, 0]}\n"
        if self.mesh_rotation is not None and self.mesh_translation is not None:
            out_string += f"\tmeshrt\t{self.mesh_rotation[0, 0]}\t{self.mesh_rotation[1, 0]}\t{self.mesh_rotation[2, 0]}\txyz\t{self.mesh_translation[0, 0]}\t{self.mesh_translation[1, 0]}\t{self.mesh_translation[2, 0]}\n"
        elif self.mesh_rotation is not None or self.mesh_translation is not None:
            raise RuntimeError("The mesh_rotation and mesh_translation must be both defined or both undefined")
        return out_string
