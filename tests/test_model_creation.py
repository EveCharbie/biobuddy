import os
import pytest

import ezc3d
import numpy as np
from biorbd import Model

from biobuddy import (
    BiomechanicalModelReal,
    SegmentReal,
    Translations,
    Rotations,
    MeshReal,
    MarkerReal,
    SegmentCoordinateSystemReal,
    DeLevaTable,
    BiomechanicalModel,
    Segment,
    Marker,
    SegmentCoordinateSystem,
    Axis,
    Mesh,
    C3dData,
    RangeOfMotion,
    Ranges,
    MeshFile,
    Contact,
    MuscleGroup,
    Muscle,
    MuscleType,
    MuscleStateType,
    ViaPoint,
)


def test_model_creation_from_static(remove_temporary: bool = True):
    """
    Produces a model from real data
    """

    kinematic_model_file_path = "temporary.bioMod"

    # Create a model holder
    bio_model = BiomechanicalModelReal()

    # The trunk segment
    bio_model.segments.append(
        SegmentReal(
            name="TRUNK",
            translations=Translations.YZ,
            rotations=Rotations.X,
            mesh=MeshReal(((0, 0, 0), (0, 0, 0.53))),
        )
    )
    bio_model.segments["TRUNK"].add_marker(MarkerReal(name="PELVIS", parent_name="TRUNK", position=np.array([0, 0, 0])))

    # The head segment
    bio_model.segments.append(
        SegmentReal(
            name="HEAD",
            parent_name="TRUNK",
            segment_coordinate_system=SegmentCoordinateSystemReal.from_euler_and_translation(
                np.array([0, 0, 0]), "xyz", np.array([0, 0, 0.53])
            ),
            mesh=MeshReal((np.array([0, 0, 0]), np.array([0, 0, 0.24]))),
        )
    )
    bio_model.segments["HEAD"].add_marker(
        MarkerReal(name="BOTTOM_HEAD", parent_name="HEAD", position=np.array([0, 0, 0]))
    )
    bio_model.segments["HEAD"].add_marker(
        MarkerReal(name="TOP_HEAD", parent_name="HEAD", position=np.array([0, 0, 0.24]))
    )
    bio_model.segments["HEAD"].add_marker(
        MarkerReal(name="HEAD_Z", parent_name="HEAD", position=np.array([0, 0, 0.24]))
    )
    bio_model.segments["HEAD"].add_marker(
        MarkerReal(name="HEAD_XZ", parent_name="HEAD", position=np.array([0.24, 0, 0.24]))
    )

    # The arm segment
    bio_model.segments.append(
        SegmentReal(
            name="UPPER_ARM",
            parent_name="TRUNK",
            segment_coordinate_system=SegmentCoordinateSystemReal.from_euler_and_translation(
                np.array([0, 0, 0]), "xyz", np.array([0, 0, 0.53])
            ),
            rotations=Rotations.X,
            mesh=MeshReal((np.array([0, 0, 0]), np.array([0, 0, -0.28]))),
        )
    )
    bio_model.segments["UPPER_ARM"].add_marker(
        MarkerReal(name="SHOULDER", parent_name="UPPER_ARM", position=np.array([0, 0, 0]))
    )
    bio_model.segments["UPPER_ARM"].add_marker(
        MarkerReal(name="SHOULDER_X", parent_name="UPPER_ARM", position=np.array([1, 0, 0]))
    )
    bio_model.segments["UPPER_ARM"].add_marker(
        MarkerReal(name="SHOULDER_XY", parent_name="UPPER_ARM", position=np.array([1, 1, 0]))
    )

    bio_model.segments.append(
        SegmentReal(
            name="LOWER_ARM",
            parent_name="UPPER_ARM",
            segment_coordinate_system=SegmentCoordinateSystemReal.from_euler_and_translation(
                np.array([0, 0, 0]), "xyz", np.array([0, 0, -0.28])
            ),
            mesh=MeshReal((np.array([0, 0, 0]), np.array([0, 0, -0.27]))),
        )
    )
    bio_model.segments["LOWER_ARM"].add_marker(
        MarkerReal(name="ELBOW", parent_name="LOWER_ARM", position=np.array([0, 0, 0]))
    )
    bio_model.segments["LOWER_ARM"].add_marker(
        MarkerReal(name="ELBOW_Y", parent_name="LOWER_ARM", position=np.array([0, 1, 0]))
    )
    bio_model.segments["LOWER_ARM"].add_marker(
        MarkerReal(name="ELBOW_XY", parent_name="LOWER_ARM", position=np.array([1, 1, 0]))
    )

    bio_model.segments.append(
        SegmentReal(
            name="HAND",
            parent_name="LOWER_ARM",
            segment_coordinate_system=SegmentCoordinateSystemReal.from_euler_and_translation(
                np.array([0, 0, 0]), "xyz", np.array([0, 0, -0.27])
            ),
            mesh=MeshReal((np.array([0, 0, 0]), np.array([0, 0, -0.19]))),
        )
    )
    bio_model.segments["HAND"].add_marker(MarkerReal(name="WRIST", parent_name="HAND", position=np.array([0, 0, 0])))
    bio_model.segments["HAND"].add_marker(
        MarkerReal(name="FINGER", parent_name="HAND", position=np.array([0, 0, -0.19]))
    )
    bio_model.segments["HAND"].add_marker(MarkerReal(name="HAND_Y", parent_name="HAND", position=np.array([0, 1, 0])))
    bio_model.segments["HAND"].add_marker(MarkerReal(name="HAND_YZ", parent_name="HAND", position=np.array([0, 1, 1])))

    # The thigh segment
    bio_model.segments.append(
        SegmentReal(
            name="THIGH",
            parent_name="TRUNK",
            rotations=Rotations.X,
            mesh=MeshReal((np.array([0, 0, 0]), np.array([0, 0, -0.42]))),
        )
    )
    bio_model.segments["THIGH"].add_marker(
        MarkerReal(name="THIGH_ORIGIN", parent_name="THIGH", position=np.array([0, 0, 0]))
    )
    bio_model.segments["THIGH"].add_marker(
        MarkerReal(name="THIGH_X", parent_name="THIGH", position=np.array([1, 0, 0]))
    )
    bio_model.segments["THIGH"].add_marker(
        MarkerReal(name="THIGH_Y", parent_name="THIGH", position=np.array([0, 1, 0]))
    )

    # The shank segment
    bio_model.segments.append(
        SegmentReal(
            name="SHANK",
            parent_name="THIGH",
            segment_coordinate_system=SegmentCoordinateSystemReal.from_euler_and_translation(
                np.array([0, 0, 0]), "xyz", np.array([0, 0, -0.42])
            ),
            rotations=Rotations.X,
            mesh=MeshReal((np.array([0, 0, 0]), np.array([0, 0, -0.43]))),
        )
    )
    bio_model.segments["SHANK"].add_marker(MarkerReal(name="KNEE", parent_name="SHANK", position=np.array([0, 0, 0])))
    bio_model.segments["SHANK"].add_marker(MarkerReal(name="KNEE_Z", parent_name="SHANK", position=np.array([0, 0, 1])))
    bio_model.segments["SHANK"].add_marker(
        MarkerReal(name="KNEE_XZ", parent_name="SHANK", position=np.array([1, 0, 1]))
    )

    # The foot segment
    bio_model.segments.append(
        SegmentReal(
            name="FOOT",
            parent_name="SHANK",
            segment_coordinate_system=SegmentCoordinateSystemReal.from_euler_and_translation(
                np.array([-np.pi / 2, 0, 0]), "xyz", np.array([0, 0, -0.43])
            ),
            rotations=Rotations.X,
            mesh=MeshReal((np.array([0, 0, 0]), np.array([0, 0, 0.25]))),
        )
    )
    bio_model.segments["FOOT"].add_marker(MarkerReal(name="ANKLE", parent_name="FOOT", position=np.array([0, 0, 0])))
    bio_model.segments["FOOT"].add_marker(MarkerReal(name="TOE", parent_name="FOOT", position=np.array([0, 0, 0.25])))
    bio_model.segments["FOOT"].add_marker(MarkerReal(name="ANKLE_Z", parent_name="FOOT", position=np.array([0, 0, 1])))
    bio_model.segments["FOOT"].add_marker(MarkerReal(name="ANKLE_YZ", parent_name="FOOT", position=np.array([0, 1, 1])))

    # Put the model together, print it and print it to a bioMod file
    bio_model.to_biomod(kinematic_model_file_path)

    model = Model(kinematic_model_file_path)
    assert model.nbQ() == 7
    assert model.nbSegment() == 8
    assert model.nbMarkers() == 25
    value = model.markers(np.zeros((model.nbQ(),)))[-3].to_array()
    np.testing.assert_almost_equal(value, [0, 0.25, -0.85], decimal=4)

    if remove_temporary:
        os.remove(kinematic_model_file_path)


def write_markers_to_c3d(save_path: str, model):
    q = np.zeros(model.nbQ())
    marker_names = tuple(name.to_string() for name in model.markerNames())
    marker_positions = np.array(tuple(m.to_array() for m in model.markers(q))).T[:, :, np.newaxis]

    c3d = ezc3d.c3d()

    # Fill it with random data
    c3d["parameters"]["POINT"]["RATE"]["value"] = [100]
    c3d["parameters"]["POINT"]["LABELS"]["value"] = marker_names
    c3d["data"]["points"] = marker_positions

    # Write the data
    c3d.write(save_path)


def test_model_creation_from_data(remove_temporary: bool = True):

    kinematic_model_file_path = "temporary.bioMod"
    c3d_file_path = "temporary.c3d"
    test_model_creation_from_static(remove_temporary=False)

    # Prepare a fake model and a fake static from the previous test
    model = Model(kinematic_model_file_path)
    write_markers_to_c3d(c3d_file_path, model)
    os.remove(kinematic_model_file_path)

    # Fill the kinematic chain model
    model = BiomechanicalModel()
    de_leva = DeLevaTable(total_mass=100, sex="female")

    model.segments["TRUNK"].append(
        Segment(
            name="TRUNK",
            translations=Translations.YZ,
            rotations=Rotations.X,
            inertia_parameters=de_leva["TRUNK"],
        )
    )
    model.segments["TRUNK"].add_marker(Marker("PELVIS"))

    model.segments.append(
        Segment(
            name="HEAD",
            parent_name="TRUNK",
            segment_coordinate_system=SegmentCoordinateSystem(
                "BOTTOM_HEAD",
                first_axis=Axis(name=Axis.Name.Z, start="BOTTOM_HEAD", end="HEAD_Z"),
                second_axis=Axis(name=Axis.Name.X, start="BOTTOM_HEAD", end="HEAD_XZ"),
                axis_to_keep=Axis.Name.Z,
            ),
            mesh=Mesh(("BOTTOM_HEAD", "TOP_HEAD", "HEAD_Z", "HEAD_XZ", "BOTTOM_HEAD")),
            inertia_parameters=de_leva["HEAD"],
        )
    )
    model.segments["HEAD"].add_marker(Marker("BOTTOM_HEAD"))
    model.segments["HEAD"].add_marker(Marker("TOP_HEAD"))
    model.segments["HEAD"].add_marker(Marker("HEAD_Z"))
    model.segments["HEAD"].add_marker(Marker("HEAD_XZ"))

    model.segments.append(
        Segment(
            name="UPPER_ARM",
            parent_name="TRUNK",
            rotations=Rotations.X,
            segment_coordinate_system=SegmentCoordinateSystem(
                origin="SHOULDER",
                first_axis=Axis(name=Axis.Name.X, start="SHOULDER", end="SHOULDER_X"),
                second_axis=Axis(name=Axis.Name.Y, start="SHOULDER", end="SHOULDER_XY"),
                axis_to_keep=Axis.Name.X,
            ),
            inertia_parameters=de_leva["UPPER_ARM"],
        )
    )
    model.segments["UPPER_ARM"].add_marker(Marker("SHOULDER"))
    model.segments["UPPER_ARM"].add_marker(Marker("SHOULDER_X"))
    model.segments["UPPER_ARM"].add_marker(Marker("SHOULDER_XY"))

    model.segments.append(
        Segment(
            name="LOWER_ARM",
            parent_name="UPPER_ARM",
            segment_coordinate_system=SegmentCoordinateSystem(
                origin="ELBOW",
                first_axis=Axis(name=Axis.Name.Y, start="ELBOW", end="ELBOW_Y"),
                second_axis=Axis(name=Axis.Name.X, start="ELBOW", end="ELBOW_XY"),
                axis_to_keep=Axis.Name.Y,
            ),
            inertia_parameters=de_leva["LOWER_ARM"],
        )
    )
    model.segments["LOWER_ARM"].add_marker(Marker("ELBOW"))
    model.segments["LOWER_ARM"].add_marker(Marker("ELBOW_Y"))
    model.segments["LOWER_ARM"].add_marker(Marker("ELBOW_XY"))

    model.segments.append(
        Segment(
            name="HAND",
            parent_name="LOWER_ARM",
            segment_coordinate_system=SegmentCoordinateSystem(
                origin="WRIST",
                first_axis=Axis(name=Axis.Name.Y, start="WRIST", end="HAND_Y"),
                second_axis=Axis(name=Axis.Name.Z, start="WRIST", end="HAND_YZ"),
                axis_to_keep=Axis.Name.Y,
            ),
            inertia_parameters=de_leva["HAND"],
        )
    )
    model.segments["HAND"].add_marker(Marker("WRIST"))
    model.segments["HAND"].add_marker(Marker("FINGER"))
    model.segments["HAND"].add_marker(Marker("HAND_Y"))
    model.segments["HAND"].add_marker(Marker("HAND_YZ"))

    model.segments.append(
        Segment(
            name="THIGH",
            parent_name="TRUNK",
            rotations=Rotations.X,
            segment_coordinate_system=SegmentCoordinateSystem(
                origin="THIGH_ORIGIN",
                first_axis=Axis(name=Axis.Name.X, start="THIGH_ORIGIN", end="THIGH_X"),
                second_axis=Axis(name=Axis.Name.Y, start="THIGH_ORIGIN", end="THIGH_Y"),
                axis_to_keep=Axis.Name.X,
            ),
            inertia_parameters=de_leva["THIGH"],
        )
    )
    model.segments["THIGH"].add_marker(Marker("THIGH_ORIGIN"))
    model.segments["THIGH"].add_marker(Marker("THIGH_X"))
    model.segments["THIGH"].add_marker(Marker("THIGH_Y"))

    model.segments.append(
        Segment(
            name="SHANK",
            parent_name="THIGH",
            rotations=Rotations.X,
            segment_coordinate_system=SegmentCoordinateSystem(
                origin="KNEE",
                first_axis=Axis(name=Axis.Name.Z, start="KNEE", end="KNEE_Z"),
                second_axis=Axis(name=Axis.Name.X, start="KNEE", end="KNEE_XZ"),
                axis_to_keep=Axis.Name.Z,
            ),
            inertia_parameters=de_leva["SHANK"],
        )
    )
    model.segments["SHANK"].add_marker(Marker("KNEE"))
    model.segments["SHANK"].add_marker(Marker("KNEE_Z"))
    model.segments["SHANK"].add_marker(Marker("KNEE_XZ"))

    model.segments["FOOT"] = Segment(
        name="FOOT",
        parent_name="SHANK",
        rotations=Rotations.X,
        segment_coordinate_system=SegmentCoordinateSystem(
            origin="ANKLE",
            first_axis=Axis(name=Axis.Name.Z, start="ANKLE", end="ANKLE_Z"),
            second_axis=Axis(name=Axis.Name.Y, start="ANKLE", end="ANKLE_YZ"),
            axis_to_keep=Axis.Name.Z,
        ),
        inertia_parameters=de_leva["FOOT"],
    )
    model.segments["FOOT"].add_marker(Marker("ANKLE"))
    model.segments["FOOT"].add_marker(Marker("TOE"))
    model.segments["FOOT"].add_marker(Marker("ANKLE_Z"))
    model.segments["FOOT"].add_marker(Marker("ANKLE_YZ"))

    # Put the model together
    model.personalize_model(C3dData(c3d_file_path))

    # print it to a bioMod file
    model.to_biomod(kinematic_model_file_path)

    model = Model(kinematic_model_file_path)
    assert model.nbQ() == 7
    assert model.nbSegment() == 8
    assert model.nbMarkers() == 25
    value = model.markers(np.zeros((model.nbQ(),)))[-3].to_array()
    np.testing.assert_almost_equal(value, [0, 0.25, -0.85], decimal=4)

    if remove_temporary:
        os.remove(kinematic_model_file_path)
        os.remove(c3d_file_path)


def test_complex_model(remove_temporary: bool = True):

    current_path_folder = os.path.dirname(os.path.realpath(__file__))
    mesh_path = f"{current_path_folder}/../examples/models/meshes/pendulum.STL"

    kinematic_model_file_path = "temporary_complex.bioMod"

    # Create a model holder
    bio_model = BiomechanicalModel()

    # The ground segment
    bio_model.segments.append(Segment(name="GROUND"))

    # The pendulum segment
    bio_model.segments.append(
        Segment(
            name="PENDULUM",
            translations=Translations.XYZ,
            rotations=Rotations.X,
            q_ranges=RangeOfMotion(range_type=Ranges.Q, min_bound=[-1, -1, -1, -np.pi], max_bound=[1, 1, 1, np.pi]),
            qdot_ranges=RangeOfMotion(
                range_type=Ranges.Qdot, min_bound=[-10, -10, -10, -np.pi * 10], max_bound=[10, 10, 10, np.pi * 10]
            ),
            mesh_file=MeshFile(
                mesh_file_name=mesh_path,
                mesh_color=np.array([0, 0, 1]),
                scaling_function=lambda m: np.array([1, 1, 10]),
                rotation_function=lambda m: np.array([np.pi / 2, 0, 0]),
                translation_function=lambda m: np.array([0.1, 0, 0]),
            ),
        )
    )
    # The pendulum segment contact point
    bio_model.segments["PENDULUM"].add_contact(
        Contact(
            name="PENDULUM_CONTACT",
            function=lambda m: np.array([0, 0, 0]),
            parent_name="PENDULUM",
            axis=Translations.XYZ,
        )
    )

    # The pendulum muscle group
    bio_model.muscle_groups["PENDULUM_MUSCLE_GROUP"] = MuscleGroup(
        name="PENDULUM_MUSCLE_GROUP", origin_parent_name="GROUND", insertion_parent_name="PENDULUM"
    )

    # The pendulum muscle
    bio_model.muscles["PENDULUM_MUSCLE"] = Muscle(
        "PENDULUM_MUSCLE",
        muscle_type=MuscleType.HILL_THELEN,
        state_type=MuscleStateType.DEGROOTE,
        muscle_group="PENDULUM_MUSCLE_GROUP",
        origin_position_function=lambda m: np.array([0, 0, 0]),
        insertion_position_function=lambda m: np.array([0, 0, 1]),
        optimal_length_function=lambda model, m: 0.1,
        maximal_force_function=lambda m: 100.0,
        tendon_slack_length_function=lambda model, m: 0.05,
        pennation_angle_function=lambda model, m: 0.05,
        maximal_excitation=1,
    )
    bio_model.via_points["PENDULUM_MUSCLE"] = ViaPoint(
        "PENDULUM_MUSCLE",
        position_function=lambda m: np.array([0, 0, 0.5]),
        parent_name="PENDULUM",
        muscle_name="PENDULUM_MUSCLE",
        muscle_group="PENDULUM_MUSCLE_GROUP",
    )

    # Put the model together
    bio_model.personalize_model({})

    # Print it to a bioMod file
    bio_model.to_biomod(kinematic_model_file_path)

    model = Model(kinematic_model_file_path)
    assert model.nbQ() == 4
    assert model.nbSegment() == 2
    assert model.nbMarkers() == 0
    assert model.nbMuscles() == 1
    assert model.nbMuscleGroups() == 1
    assert model.nbContacts() == 3

    if remove_temporary:
        os.remove(kinematic_model_file_path)
