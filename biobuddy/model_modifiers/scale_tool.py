from copy import deepcopy
from enum import Enum

import numpy as np

from ..components.real.biomechanical_model_real import BiomechanicalModelReal
from ..components.real.biomechanical_model_real_utils import inverse_kinematics
from ..components.real.rigidbody.segment_scaling import SegmentScaling
from ..components.real.rigidbody.segment_real import SegmentReal
from ..components.real.rigidbody.marker_real import MarkerReal
from ..components.real.rigidbody.contact_real import ContactReal
from ..components.real.rigidbody.inertial_measurement_unit_real import InertialMeasurementUnitReal
from ..components.real.rigidbody.inertia_parameters_real import InertiaParametersReal
from ..components.real.rigidbody.segment_coordinate_system_real import SegmentCoordinateSystemReal
from ..components.real.muscle.muscle_real import MuscleReal
from ..components.real.muscle.via_point_real import ViaPointReal
from ..utils.linear_algebra import RotoTransMatrix
from ..utils.named_list import NamedList
from ..utils.c3d_data import C3dData


class InertialCharacteristics(Enum):
    DE_LEVA = "de_leva"  # TODO


class ScaleTool:
    def __init__(self, original_model: BiomechanicalModelReal, personalize_mass_distribution: bool = True, max_marker_movement: float = 0.1):
        """
        Initialize the scale tool.

        Parameters
        ----------
        original_model
            The original model to scale
        personalize_mass_distribution
            If True, the mass distribution of the mass across segments will be personalized based on the marker positions. Otherwise, the mass distribution across segments will be the same as the original model.
        max_marker_movement
            The maximum acceptable marker movement in the static trial to consider it "static".
        """

        # Original attributes
        self.original_model = original_model
        self.original_model_biorbd = self.original_model.get_biorbd_model  # TODO: remove
        self.personalize_mass_distribution = personalize_mass_distribution
        self.max_marker_movement = max_marker_movement

        # Extended attributes to be filled
        self.scaled_model = BiomechanicalModelReal()
        self.scaled_model_biorbd = None  # This scaled model is defined later when the segment shape is defined
        self.mean_experimental_markers = None  # This field will be set when .scale is run

        self.header = ""
        self.original_mass = None
        self.scaling_segments = NamedList[SegmentScaling]()
        self.marker_weightings = {}
        self.warnings = ""

    def scale(
        self, file_path: str, first_frame: int, last_frame: int, mass: float, visualize_optimal_static_pose: bool = False
    ) -> BiomechanicalModelReal:
        """
        Scale the model using the configuration defined in the ScaleTool.

        Parameters
        ----------
        file_path
            The .c3d or .trc file of the static trial to use for the scaling
        first_frame
            The index of the first frame to use in the .c3d file.
        last_frame
            The index of the last frame to use in the .c3d file.
        mass
            The mass of the subject
        """

        # Check file format
        if file_path.endswith(".c3d"):
            # Load the c3d file
            c3d_data = C3dData(file_path, first_frame, last_frame)
            marker_names = c3d_data.marker_names
            marker_positions = c3d_data.all_marker_positions[:3, :, :]
        else:
            if file_path.endswith(".trc"):
                raise NotImplementedError(".trc files cannot be read yet.")
            else:
                raise RuntimeError("The file_path (static trial) must be a .c3d file in a static posture.")

        # Check the weights
        for marker in self.marker_weightings:
            if self.marker_weightings[marker] < 0:
                raise RuntimeError(f"The weight of marker {marker} is negative. It must be positive.")

        # Check the mass
        if mass <= 0:
            raise RuntimeError(f"The mass of the subject must be positive. The value given is {mass} kg.")

        # Check that a scaling configuration was set
        if len(self.scaling_segments) == 0:
            raise RuntimeError(
                "No scaling configuration was set. Please set a scaling configuration using ScaleTool.from_xml(filepath=filepath) or ScaleTool.from_biomod(filepath=filepath)."
            )

        self.check_that_makers_do_not_move(marker_positions, marker_names)
        self.check_segments()
        self.define_mean_experimental_markers(marker_positions, marker_names)

        self.scale_model_geometrically(marker_positions, marker_names, mass)
        self.scaled_model_biorbd = self.scaled_model.get_biorbd_model

        self.modify_muscle_parameters()
        self.place_model_in_static_pose(marker_positions, marker_names, visualize_optimal_static_pose)

        return self.scaled_model


    def check_that_makers_do_not_move(self, marker_positions, marker_names):
        """
        Check that the markers do not move too much in the static trial

        Parameters
        ----------
        marker_positions
            The position of the markers in the static trial (within the frame_range)
        marker_names
            The names of the marker labels in the c3d static file
        """

        if self.max_marker_movement is None:
            return

        else:
            for marker in self.marker_weightings:
                if marker not in marker_names:
                    raise RuntimeError(f"The marker {marker} is not in the c3d file.")
                marker_index = marker_names.index(marker)
                this_marker_position = marker_positions[:, marker_index, :]
                min_position = np.nanmin(this_marker_position, axis=1)
                max_position = np.nanmax(this_marker_position, axis=1)
                if np.linalg.norm(max_position - min_position) > self.max_marker_movement:
                    raise RuntimeError(
                        f"The marker {marker} moves of approximately {np.linalg.norm(max_position - min_position)} m during the static trial, which is above the maximal limit of {self.max_marker_movement} m."
                    )
            return

    def check_segments(self):

        # Check that all segments that bear mass are scaled.
        for segment_name in self.original_model.segments.keys():
            inertia = deepcopy(self.original_model.segments[segment_name].inertia_parameters)
            if inertia is not None and inertia.mass > 0.1 and segment_name not in self.scaling_segments.keys():
                raise RuntimeError(
                    f"The segment {segment_name} has a positive mass of {self.original_model.segments[segment_name].inertia_parameters.mass}, but is not defined in the scaling configuration."
                )

        # Check that all scaled segments exist in the original model.
        for segment_name in self.scaling_segments.keys():
            if segment_name not in self.original_model.segments.keys():
                raise RuntimeError(
                    f"The segment {segment_name} has a scaling configuration, but does not exist in the original model."
                )

    def define_mean_experimental_markers(self, marker_positions, marker_names):
        model_marker_names = [m.to_string() for m in self.original_model_biorbd.markerNames()]
        self.mean_experimental_markers = np.zeros((3, len(model_marker_names)))
        for i_marker, name in enumerate(model_marker_names):
            marker_index = marker_names.index(name)
            this_marker_position = marker_positions[:, marker_index, :]
            self.mean_experimental_markers[:, i_marker] = np.nanmean(this_marker_position, axis=1)

    def get_scaling_factors_and_masses(
        self,
        marker_positions: np.ndarray,
        marker_names: list[str],
        mass: float,
        original_mass: float,
    ) -> tuple[dict[str, "ScaleFactor"], dict[str, np.ndarray]]:

        scaling_factors = {}
        segment_masses = {}
        total_scaled_mass = 0
        for segment_name in self.scaling_segments.keys():
            # Compute the scale factors
            scaling_factors[segment_name] = self.scaling_segments[segment_name].compute_scaling_factors(
                marker_positions, marker_names, self.original_model_biorbd
            )
            # Get each segment's scaled mass
            if self.personalize_mass_distribution:
                segment_masses[segment_name] = (
                    deepcopy(self.original_model.segments[segment_name].inertia_parameters.mass)
                    * scaling_factors[segment_name].to_vector()
                )
            else:
                segment_masses[segment_name] = (
                    deepcopy(self.original_model.segments[segment_name].inertia_parameters.mass) * mass / original_mass
                )
            total_scaled_mass += segment_masses[segment_name]

        # Renormalize segment's mass to make sure the total mass is the mass of the subject
        for segment_name in self.scaling_segments.keys():
            segment_masses[segment_name] *= mass / total_scaled_mass

        return scaling_factors, segment_masses

    def scale_model_geometrically(self, marker_positions: np.ndarray, marker_names: list[str], mass: float):

        original_mass = self.original_model_biorbd.mass()

        scaling_factors, segment_masses = self.get_scaling_factors_and_masses(
            marker_positions, marker_names, mass, original_mass
        )

        self.scaled_model.header = deepcopy(self.original_model.header) + f"\nModel scaled using Biobuddy.\n"
        self.scaled_model.gravity = deepcopy(self.original_model.gravity)

        for segment_name in self.original_model.segments.keys():

            # Check if the segments has a ghost parent
            if (
                self.original_model.segments[segment_name].name + "_parent_offset"
                in self.original_model.segments.keys()
            ):
                offset_parent = self.original_model.segments[segment_name + "_parent_offset"].parent_name
                if offset_parent in self.scaling_segments.keys():
                    # Apply scaling to the position of the offset parent segment instead of the current segment
                    offset_parent_scale_factor = scaling_factors[offset_parent].to_vector()
                    scs_scaled = SegmentCoordinateSystemReal(
                        scs=self.scale_rt(
                            deepcopy(
                                self.original_model.segments[
                                    segment_name + "_parent_offset"
                                ].segment_coordinate_system.scs[:, :, 0]
                            ),
                            offset_parent_scale_factor,
                        ),
                        is_scs_local=True,
                    )
                    self.scaled_model.segments[segment_name + "_parent_offset"].segment_coordinate_system = scs_scaled
                    parent_scale_factor = np.ones((4, 1))
            else:
                # Apply scaling to the current segment
                if self.original_model.segments[segment_name].parent_name in self.scaling_segments.keys():
                    parent_scale_factor = scaling_factors[
                        self.original_model.segments[segment_name].parent_name
                    ].to_vector()
                else:
                    parent_scale_factor = np.ones((4, 1))

            # Scale segments
            if segment_name in self.scaling_segments.keys():
                this_segment_scale_factor = scaling_factors[segment_name].to_vector()
                self.scaled_model.segments.append(
                    self.scale_segment(
                        deepcopy(self.original_model.segments[segment_name]),
                        parent_scale_factor,
                        this_segment_scale_factor,
                        segment_masses[segment_name],
                    )
                )
            else:
                self.scaled_model.segments[segment_name] = deepcopy(self.original_model.segments[segment_name])

            for marker in deepcopy(self.original_model.segments[segment_name].markers):
                self.scaled_model.segments[segment_name].remove_marker(marker.name)
                self.scaled_model.segments[segment_name].add_marker(
                    self.scale_marker(marker, this_segment_scale_factor)
                )

            for contact in deepcopy(self.original_model.segments[segment_name].contacts):
                self.scaled_model.segments[segment_name].remove_contact(contact.name)
                self.scaled_model.segments[segment_name].add_contact(
                    self.scale_contact(contact, this_segment_scale_factor)
                )

            for imu in deepcopy(self.original_model.segments[segment_name].imus):
                self.scaled_model.segments[segment_name].remove_imu(imu.name)
                self.scaled_model.segments[segment_name].add_imu(self.scale_imu(imu, this_segment_scale_factor))

        # Set muscle groups
        self.scaled_model.muscle_groups = deepcopy(self.original_model.muscle_groups)

        # Scale muscles
        for muscle_name in self.original_model.muscles.keys():

            muscle_group_name = deepcopy(self.original_model.muscles[muscle_name].muscle_group)
            origin_parent_name = deepcopy(self.original_model.muscle_groups[muscle_group_name].origin_parent_name)
            insertion_parent_name = deepcopy(self.original_model.muscle_groups[muscle_group_name].insertion_parent_name)
            origin_scale_factor = scaling_factors[origin_parent_name].to_vector()
            insertion_scale_factor = scaling_factors[insertion_parent_name].to_vector()

            if (
                origin_parent_name not in self.scaling_segments.keys()
                and insertion_parent_name not in self.scaling_segments.keys()
            ):
                # If the muscle is not attached to a segment that is scaled, do not scale the muscle
                self.scaled_model.muscles.append(deepcopy(self.original_model.muscles[muscle_name]))
            else:
                self.scaled_model.muscles.append(
                    self.scale_muscle(
                        deepcopy(self.original_model.muscles[muscle_name]), origin_scale_factor, insertion_scale_factor
                    )
                )

        # Scale via points
        for via_point_name in self.original_model.via_points.keys():

            parent_name = deepcopy(self.original_model.via_points[via_point_name].parent_name)
            parent_scale_factor = scaling_factors[parent_name].to_vector()

            if parent_name not in self.scaling_segments.keys():
                # If the via point is not attached to a segment that is scaled, do not scale the via point
                self.scaled_model.via_points.append(deepcopy(self.original_model.via_points[via_point_name]))
            else:
                self.scaled_model.via_points.append(
                    self.scale_via_point(deepcopy(self.original_model.via_points[via_point_name]), parent_scale_factor)
                )

        self.scaled_model.warnings = deepcopy(self.original_model.warnings)

        return

    @staticmethod
    def scale_rt(rt: np.ndarray, scale_factor: np.ndarray) -> np.ndarray:
        rt_matrix = deepcopy(rt)
        rt_matrix[:3, 3] *= scale_factor[:3].reshape(
            3,
        )
        return rt_matrix

    def scale_segment(
        self,
        original_segment: SegmentReal,
        parent_scale_factor: np.ndarray,
        scale_factor: np.ndarray,
        segment_mass: float,
    ) -> SegmentReal:
        """
        Only geometrical scaling is implemented.
        TODO: Implement scaling based on De Leva table.
        """

        if original_segment.segment_coordinate_system.is_in_global:
            raise NotImplementedError(
                "The segment_coordinate_system is not in the parent reference frame. This is not implemented yet."
            )

        segment_coordinate_system = SegmentCoordinateSystemReal(
            scs=self.scale_rt(original_segment.segment_coordinate_system.scs[:, :, 0], parent_scale_factor),
            is_scs_local=True,
        )

        original_radii_of_gyration = np.array(
            [
                np.sqrt(inertia / original_segment.inertia_parameters.mass)
                for inertia in original_segment.inertia_parameters.inertia
            ]
        )
        scaled_inertia = segment_mass * (original_radii_of_gyration * scale_factor) ** 2

        inertia_parameters = InertiaParametersReal(
            mass=segment_mass,
            center_of_mass=original_segment.inertia_parameters.center_of_mass * scale_factor,
            inertia=scaled_inertia,
        )

        mesh_file = deepcopy(original_segment.mesh_file)
        mesh_file.mesh_scale *= scale_factor
        mesh_file.mesh_translation *= scale_factor

        scaled_segment = deepcopy(original_segment)
        scaled_segment.segment_coordinate_system = segment_coordinate_system
        scaled_segment.inertia_parameters = inertia_parameters
        scaled_segment.mesh_file = mesh_file

        return scaled_segment

    def scale_marker(self, original_marker: MarkerReal, scale_factor: np.ndarray) -> MarkerReal:
        scaled_marker = deepcopy(original_marker)
        scaled_marker.position *= scale_factor
        return scaled_marker

    def scale_contact(self, original_contact: ContactReal, scale_factor: np.ndarray) -> ContactReal:
        scaled_contact = deepcopy(original_contact)
        scaled_contact.position *= scale_factor
        return scaled_contact

    def scale_imu(
        self, original_imu: InertialMeasurementUnitReal, scale_factor: np.ndarray
    ) -> InertialMeasurementUnitReal:
        scaled_imu = deepcopy(original_imu)
        scaled_imu.scs = self.scale_rt(original_imu.scs[:, :, 0], scale_factor)
        return scaled_imu

    def scale_muscle(
        self, original_muscle: MuscleReal, origin_scale_factor: np.ndarray, insertion_scale_factor: np.ndarray
    ) -> MuscleReal:
        scaled_muscle = deepcopy(original_muscle)
        scaled_muscle.origin_position *= origin_scale_factor
        scaled_muscle.insertion_position *= insertion_scale_factor
        scaled_muscle.optimal_length = (None,)  # Will be set later
        scaled_muscle.tendon_slack_length = (None,)  # Will be set later
        return scaled_muscle

    def scale_via_point(self, original_via_point: ViaPointReal, parent_scale_factor: np.ndarray) -> ViaPointReal:
        scaled_via_point = deepcopy(original_via_point)
        scaled_via_point.position *= parent_scale_factor
        return scaled_via_point

    def find_static_pose(self, marker_positions: np.ndarray, marker_names: list[str], visualize_optimal_static_pose: bool) -> np.ndarray:

        nb_q = self.scaled_model_biorbd.nbQ()
        optimal_q = inverse_kinematics(
                self.scaled_model_biorbd,
                marker_positions=marker_positions,
                marker_names=marker_names,
                q_regularization_weight=0.001,
                q_target=np.zeros((nb_q, )))

        if visualize_optimal_static_pose:
            # Show the animation for debugging
            try:
                import pyorerun
                from pyomeca import Markers
            except ImportError:
                raise ImportError("You must install pyorerun and pyomeca to visualize the model")

            t = np.linspace(0, 1, marker_positions.shape[2])
            viz = pyorerun.PhaseRerun(t)

            # Biorbd model translated from .osim
            model_biorbd = self.original_model.get_biorbd_model
            viz_biomod_model = pyorerun.BiorbdModel.from_biorbd_object(model_biorbd)
            viz_biomod_model.options.transparent_mesh = False
            viz_biomod_model.options.show_gravity = True
            viz.add_animated_model(viz_biomod_model, optimal_q)

            biorbd_marker_names = [m.to_string() for m in model_biorbd.markerNames()]
            marker_indices = [marker_names.index(m) for m in biorbd_marker_names]
            pyomarkers = Markers(data=marker_positions[:, marker_indices, :], channels=biorbd_marker_names)
            viz.add_xp_markers(name=marker_names, markers=pyomarkers)
            # tracked_markers=pyomarkers
            viz.rerun_by_frame("Model output")

        if any(np.std(optimal_q, axis=1) > 20 * np.pi / 180):
            raise RuntimeError(
                "The inverse kinematics shows more than 20° variance over the frame range specified."
                "Please see the animation provided to verify that the subject does not move during the static trial."
                "If not, please make sure the model and subject are not positioned close to singularities (gimbal lock)."
            )

        return np.median(optimal_q, axis=1)


    def make_static_pose_the_zero(self, q_static: np.ndarray):
        for i_segment, segment_name in enumerate(self.scaled_model.segments.keys()):
            segment_jcs = self.scaled_model_biorbd.globalJCS(q_static, i_segment, True).to_array()
            self.scaled_model.segments[segment_name].segment_coordinate_system = SegmentCoordinateSystemReal(
                scs=segment_jcs,
                parent_scs=None,
                is_scs_local=(
                    segment_name == "base"
                ),  # joint coordinate system is now expressed in the global except for the base because it does not have a parent
            )

    def replace_markers_on_segments(self, q_static: np.ndarray, marker_positions: np.ndarray, marker_names: list[str]):
        for i_segment, segment_name in enumerate(self.scaled_model.segments.keys()):
            for marker in self.scaled_model.segments[segment_name].markers:
                marker_name = marker.name
                marker_index = marker_names.index(marker_name)
                this_marker_position = np.nanmean(marker_positions[:, marker_index], axis=1)
                segment_jcs = self.scaled_model_biorbd.globalJCS(q_static, i_segment, True).to_array()
                rt_matrix = RotoTransMatrix()
                rt_matrix.from_rt_matrix(segment_jcs)
                marker.position = rt_matrix.inverse @ np.hstack((this_marker_position, 1))

    def place_model_in_static_pose(self, marker_positions: np.ndarray, marker_names: list[str], visualize_optimal_static_pose: bool):
        q_static = self.find_static_pose(marker_positions, marker_names, visualize_optimal_static_pose)
        self.make_static_pose_the_zero(q_static)
        self.replace_markers_on_segments(q_static, marker_positions, marker_names)

    def modify_muscle_parameters(self):
        """
        Modify the optimal length, tendon slack length and pennation angle of the muscles.
        """
        muscle_names = [m.to_string() for m in self.original_model_biorbd.muscleNames()]
        q_zeros = np.zeros((self.original_model_biorbd.nbQ(),))
        for muscle_name in self.original_model.muscles.keys():
            muscle_idx = muscle_names.index(muscle_name)
            original_muscle_length = self.original_model_biorbd.muscle(muscle_idx).length(
                self.original_model_biorbd, q_zeros
            )
            scaled_muscle_length = self.scaled_model_biorbd.muscle(muscle_idx).length(self.scaled_model_biorbd, q_zeros)
            if self.original_model.muscles[muscle_name].optimal_length is None:
                print("sss")
            self.scaled_model.muscles[muscle_name].optimal_length = (
                deepcopy(self.original_model.muscles[muscle_name].optimal_length)
                * scaled_muscle_length
                / original_muscle_length
            )
            self.scaled_model.muscles[muscle_name].tendon_slack_length = (
                deepcopy(self.original_model.muscles[muscle_name].tendon_slack_length)
                * scaled_muscle_length
                / original_muscle_length
            )

    @staticmethod
    def from_biomod(
        filepath: str,
    ):
        """
        Create a biomechanical model from a biorbd model
        """
        from ..model_parser.biorbd import BiomodConfigurationParser

        return BiomodConfigurationParser(filepath=filepath)

    def from_xml(
            self,
            filepath: str,
    ):
        """
        Read an xml file from OpenSim and extract the scaling configuration.

        Parameters
        ----------
        filepath: str
            The path to the xml file to read from
        """
        from ..model_parser.opensim import OsimConfigurationParser

        configuration = OsimConfigurationParser(filepath=filepath, original_model=self.original_model)
        return configuration.scale_tool
