from copy import deepcopy
from typing import Callable

import numpy as np

from ...components.real.biomechanical_model_real import BiomechanicalModelReal
from ...components.real.rigidbody.segment_real import (
    SegmentReal,
    InertialMeasurementUnitReal,
    InertiaParametersReal,
    MeshReal,
    SegmentCoordinateSystemReal,
    MarkerReal,
)
from ...components.real.muscle.muscle_real import MuscleReal, MuscleType, MuscleStateType
from ...components.generic.muscle.muscle_group import MuscleGroup
from ...components.real.muscle.via_point_real import ViaPointReal
from ...utils.named_list import NamedList


class EndOfFileReached(Exception):
    pass


class BiomodModelParser:
    def __init__(self, filepath: str):
        tokens = _tokenize_biomod(filepath=filepath)

        # Prepare the internal structure to hold the model
        self.gravity = None
        self.segments = NamedList[SegmentReal]()
        self.muscle_groups = NamedList[MuscleGroup]()
        self.muscles = NamedList[MuscleReal]()
        self.via_points = NamedList[ViaPointReal]()
        self.warnings = ""

        def next_token():
            nonlocal token_index
            token_index += 1
            if token_index >= len(tokens):
                raise EndOfFileReached()
            return tokens[token_index]

        # Parse the model
        biorbd_version = None
        gravity = None
        current_component = None
        token_index = -1
        try:
            while True:
                token = _read_str(next_token=next_token).lower()

                if current_component is None:
                    if token == "version":
                        if biorbd_version is not None:
                            raise ValueError("Version already defined")
                        biomod_version = _read_int(next_token=next_token)
                        # True for version 3 or less, False for version 4 or more
                        rt_in_matrix_default = biomod_version < 4
                    elif token == "gravity":
                        _check_if_version_defined(biomod_version)
                        if gravity is not None:
                            raise ValueError("Gravity already defined")
                        gravity = _read_float_vector(next_token=next_token, length=3)
                    elif token == "segment":
                        _check_if_version_defined(biomod_version)
                        current_component = SegmentReal(name=_read_str(next_token=next_token))
                        current_rt_in_matrix = rt_in_matrix_default
                    elif token == "imu":
                        _check_if_version_defined(biomod_version)
                        current_component = InertialMeasurementUnitReal(
                            name=_read_str(next_token=next_token), parent_name=""
                        )
                        current_rt_in_matrix = rt_in_matrix_default
                    elif token == "marker":
                        _check_if_version_defined(biomod_version)
                        current_component = MarkerReal(name=_read_str(next_token=next_token), parent_name="")
                    elif token == "musclegroup":
                        _check_if_version_defined(biomod_version)
                        current_component = MuscleGroup(
                            name=_read_str(next_token=next_token), origin_parent_name="", insertion_parent_name=""
                        )
                    elif token == "muscle":
                        _check_if_version_defined(biomod_version)
                        current_component = MuscleReal(
                            name=_read_str(next_token=next_token),
                            muscle_type = MuscleType.HILL_DE_GROOTE,
                            state_type = MuscleStateType.DEGROOTE,
                            muscle_group = "",
                            origin_position = None,
                            insertion_position = None,
                            optimal_length = None,
                            maximal_force = None,
                            tendon_slack_length = None,
                            pennation_angle = None,
                            maximal_excitation = None,
                        )
                    elif token == "viapoint":
                        _check_if_version_defined(biomod_version)
                        current_component = ViaPointReal(
                            name=_read_str(next_token=next_token),
                            parent_name="",
                            muscle_name="",
                            muscle_group="",
                            position = None,
                        )
                    else:
                        raise ValueError(f"Unknown component {token}")

                elif isinstance(current_component, SegmentReal):
                    if token == "endsegment":
                        self.segments.append(current_component)
                        current_component = None
                    elif token == "parent":
                        current_component.parent_name = _read_str(next_token=next_token)
                    elif token == "rtinmatrix":
                        current_rt_in_matrix = _read_bool(next_token=next_token)
                    elif token == "rt":
                        scs = _get_rt_matrix(next_token=next_token, current_rt_in_matrix=current_rt_in_matrix)
                        current_component.segment_coordinate_system = SegmentCoordinateSystemReal(
                            scs=scs, is_scs_local=True
                        )
                    elif token == "translations":
                        current_component.translations = _read_str(next_token=next_token)
                    elif token == "rotations":
                        current_component.rotations = _read_str(next_token=next_token)
                    elif token in ("mass", "com", "centerofmass", "inertia", "inertia_xxyyzz"):
                        if current_component.inertia_parameters is None:
                            current_component.inertia_parameters = InertiaParametersReal()

                        if token == "mass":
                            current_component.inertia_parameters.mass = _read_float(next_token=next_token)
                        elif token == "com" or token == "centerofmass":
                            com = _read_float_vector(next_token=next_token, length=3)
                            current_component.inertia_parameters.center_of_mass = com
                        elif token == "inertia":
                            inertia = _read_float_vector(next_token=next_token, length=9).reshape((3, 3))
                            current_component.inertia_parameters.inertia = inertia
                        elif token == "inertia_xxyyzz":
                            inertia = _read_float_vector(next_token=next_token, length=3)
                            current_component.inertia_parameters.inertia = np.diag(inertia)
                    elif token == "mesh":
                        if current_component.mesh is None:
                            current_component.mesh = MeshReal()
                        position = _read_float_vector(next_token=next_token, length=3).T
                        current_component.mesh.add_positions(position)
                    elif token == "mesh_file":
                        raise NotImplementedError()
                    else:
                        raise ValueError(f"Unknown information in segment")

                elif isinstance(current_component, InertialMeasurementUnitReal):
                    if token == "endimu":
                        if not current_component.parent_name:
                            raise ValueError(f"Parent name not found in imu {current_component.name}")
                        self.segments[current_component.parent_name].imus.append(current_component)
                        current_component = None
                    elif token == "parent":
                        current_component.parent_name = _read_str(next_token=next_token)
                    elif token == "rtinmatrix":
                        current_rt_in_matrix = _read_bool(next_token=next_token)
                    elif token == "rt":
                        scs = _get_rt_matrix(next_token=next_token, current_rt_in_matrix=current_rt_in_matrix)
                        current_component.scs = scs
                    elif token == "technical":
                        current_component.is_technical = _read_bool(next_token=next_token)
                    elif token == "anatomical":
                        current_component.is_anatomical = _read_bool(next_token=next_token)

                elif isinstance(current_component, MarkerReal):
                    if token == "endmarker":
                        if not current_component.parent_name:
                            raise ValueError(f"Parent name not found in marker {current_component.name}")
                        self.segments[current_component.parent_name].markers.append(current_component)
                        current_component = None
                    elif token == "parent":
                        current_component.parent_name = _read_str(next_token=next_token)
                    elif token == "position":
                        current_component.position = _read_float_vector(next_token=next_token, length=3)
                    elif token == "technical":
                        current_component.is_technical = _read_bool(next_token=next_token)
                    elif token == "anatomical":
                        current_component.is_anatomical = _read_bool(next_token=next_token)

                elif isinstance(current_component, MuscleGroup):
                    if token == "endmusclegroup":
                        if not current_component.insertion_parent_name:
                            raise ValueError(f"Insertion parent name not found in musclegroup {current_component.name}")
                        if not current_component.origin_parent_name:
                            raise ValueError(f"Origin parent name not found in musclegroup {current_component.name}")
                        self.muscle_groups.append(current_component)
                        current_component = None
                    elif token == "insertionparent":
                        current_component.insertion_parent_name = _read_str(next_token=next_token)
                    elif token == "originparent":
                        current_component.origin_parent_name = _read_str(next_token=next_token)

                elif isinstance(current_component, MuscleReal):
                    if token == "endmuscle":
                        if not current_component.muscle_type:
                            raise ValueError(f"Muscle type not found in muscle {current_component.name}")
                        if not current_component.state_type:
                            raise ValueError(f"Muscle state type not found in muscle {current_component.name}")
                        if not current_component.muscle_group:
                            raise ValueError(f"Muscle group not found in muscle {current_component.name}")
                        if current_component.origin_position is None:
                            raise ValueError(f"Origin position not found in muscle {current_component.name}")
                        if current_component.insertion_position is None:
                            raise ValueError(f"Insertion position not found in muscle {current_component.name}")
                        if current_component.optimal_length is None:
                            raise ValueError(f"Optimal length not found in muscle {current_component.name}")
                        if current_component.maximal_force is None:
                            raise ValueError(f"Maximal force not found in muscle {current_component.name}")
                        if current_component.tendon_slack_length is None:
                            raise ValueError(f"Tendon slack length not found in muscle {current_component.name}")
                        if current_component.pennation_angle is None:
                            raise ValueError(f"Pennation angle not found in muscle {current_component.name}")
                        self.muscles.append(current_component)
                        current_component = None
                    elif token == "type":
                        current_component.muscle_type = MuscleType(_read_str(next_token=next_token))
                    elif token == "statetype":
                        current_component.state_type = MuscleStateType(_read_str(next_token=next_token))
                    elif token == "musclegroup":
                        current_component.muscle_group = _read_str(next_token=next_token)
                    elif token == "originposition":
                        current_component.origin_position = _read_float_vector(next_token=next_token, length=3)
                    elif token == "insertionposition":
                        current_component.insertion_position = _read_float_vector(next_token=next_token, length=3)
                    elif token == "optimallength":
                        current_component.optimal_length = _read_float(next_token=next_token)
                    elif token == "maximalforce":
                        current_component.maximal_force = _read_float(next_token=next_token)
                    elif token == "tendonslacklength":
                        current_component.tendon_slack_length = _read_float(next_token=next_token)
                    elif token == "pennationangle":
                        current_component.pennation_angle = _read_float(next_token=next_token)
                    elif token == "maximal_excitation":
                        current_component.maximal_excitation = _read_float(next_token=next_token)

                elif isinstance(current_component, ViaPointReal):
                    if token == "endviapoint":
                        if not current_component.parent_name:
                            raise ValueError(f"Parent name not found in via point {current_component.name}")
                        if not current_component.muscle_name:
                            raise ValueError(f"Muscle name type not found in via point {current_component.name}")
                        if not current_component.muscle_group:
                            raise ValueError(f"Muscle group not found in muscle {current_component.name}")
                        self.via_points.append(current_component)
                        current_component = None
                    elif token == "parent":
                        current_component.parent_name = _read_str(next_token=next_token)
                    elif token == "muscle":
                        current_component.muscle_name = _read_str(next_token=next_token)
                    elif token == "musclegroup":
                        current_component.muscle_group = _read_str(next_token=next_token)
                    elif token == "position":
                        current_component.position = _read_float_vector(next_token=next_token, length=3)

                else:
                    raise ValueError(f"Unknown component {type(current_component)}")
        except EndOfFileReached:
            pass

    def to_real(self) -> BiomechanicalModelReal:
        model = BiomechanicalModelReal()

        # Add the segments
        for segment in self.segments:
            model.segments.append(deepcopy(segment))

        return model


def _tokenize_biomod(filepath: str) -> list[str]:
    # Load the model from the filepath
    with open(filepath, "r") as f:
        content = f.read()
    lines = content.splitlines()

    # Do a first pass to remove every commented content
    is_block_commenting = False
    line_index = 0
    for line_index in range(len(lines)):
        line = lines[line_index]
        # Remove everything after // or between /* */ (comments)
        if "/*" in line and "*/" in line:
            # Deal with the case where the block comment is on the same line
            is_block_commenting = False
            line = (line.split("/*")[0] + "" + line.split("*/")[1]).strip()
        if not is_block_commenting and "/*" in line:
            is_block_commenting = True
            line = line.split("/*")[0]
        if is_block_commenting and "*/" in line:
            is_block_commenting = False
            line = line.split("*/")[1]
        line = line.split("//")[0]
        line = line.strip()
        lines[line_index] = line
    tokens = lines

    # Make spaces also a separator
    tokens_tp: list[str] = []
    for line in tokens:
        tokens_tp.extend(line.split(" "))
    tokens = [token for token in tokens_tp if token != ""]

    # Make tabs also a separator
    tokens_tp: list[str] = []
    for token in tokens:
        tokens_tp.extend(token.split("\t"))
    tokens = [token for token in tokens_tp if token != ""]

    return tokens


def _check_if_version_defined(biomod_version: int):
    if biomod_version is None:
        raise ValueError("Version not defined")
    return


def _read_str(next_token: Callable) -> str:
    return next_token()


def _read_int(next_token: Callable) -> int:
    return int(next_token())


def _read_float(next_token: Callable) -> float:
    return float(next_token())


def _read_bool(next_token: Callable) -> bool:
    return next_token() == "1"


def _read_float_vector(next_token: Callable, length: int) -> np.ndarray:
    return np.array([_read_float(next_token=next_token) for _ in range(length)])


def _get_rt_matrix(next_token: Callable, current_rt_in_matrix: bool) -> np.ndarray:
    if current_rt_in_matrix:
        scs = SegmentCoordinateSystemReal.from_rt_matrix(
            rt_matrix=_read_float_vector(next_token=next_token, length=16).reshape((4, 4))
        )
    else:
        angles = _read_float_vector(next_token=next_token, length=3)
        angle_sequence = _read_str(next_token=next_token)
        translations = _read_float_vector(next_token=next_token, length=3)
        scs = SegmentCoordinateSystemReal.from_euler_and_translation(
            angles=angles, angle_sequence=angle_sequence, translations=translations
        )
    return scs.scs[:, :, 0]
