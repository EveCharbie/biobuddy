import os
import shutil
from xml.etree import ElementTree

import numpy as np
from lxml import etree

from biobuddy.utils import find,OrthoMatrix
from biobuddy.segment_real import SegmentReal


class OsimReader:
    def __init__(self, osim_path: str, output_model: "BiomechanicalModel", print_warnings: bool = True):
        self.osim_path = osim_path
        self.osim_model = etree.parse(self.osim_path)
        self.model = ElementTree.parse(self.osim_path)
        self.root = self.model.getroot()[0]
        self.print_warnings = print_warnings
        if self.get_file_version() < 40000:
            raise RuntimeError(
                f".osim file version must be superior or equal to '40000' and you have: {self.get_file_version()}."
                "To convert the osim file to the newest version please open and save your file in"
                "Opensim 4.0 or later."
            )

        self.gravity = np.array([0.0, 0.0, -9.81])
        self.ground_elt, self.default_elt, self.credit, self.publications = None, None, None, None
        self.bodyset_elt, self.jointset_elt, self.forceset_elt, self.markerset_elm = None, None, None, None
        self.controllerset_elt, self.constraintset_elt, self.contact_geometryset_elt = None, None, None
        self.componentset_elt, self.probeset_elt = None, None
        self.length_units, self.force_units = "meters", "newtons"

        for element in self.root:
            if element.tag == "gravity":
                gravity = [float(i) for i in element.text.split(' ')]
                self.gravity = np.array(gravity)
            elif element.tag == "Ground":
                self.ground_elt = element
            elif element.tag == "defaults":
                self.default_elt = element
            elif element.tag == "BodySet":
                self.bodyset_elt = element
            elif element.tag == "JointSet":
                self.jointset_elt = element
            elif element.tag == "ControllerSet":
                self.controllerset_elt = element
            elif element.tag == "ConstraintSet":
                self.constraintset_elt = element
            elif element.tag == "ForceSet":
                self.forceset_elt = element
            elif element.tag == "MarkerSet":
                self.markerset_elt = element
            elif element.tag == "ContactGeometrySet":
                self.contact_geometryset_elt = element
            elif element.tag == "ComponentSet":
                self.componentset_elt = element
            elif element.tag == "ProbeSet":
                self.probeset_elt = element
            elif element.tag == "credits":
                self.credit = element.text
            elif element.tag == "publications":
                self.publications = element.text
            elif element.tag == "length_units":
                self.length_units = element.text
                if self.length_units != "meters":
                    raise RuntimeError("Lengths units must be in meters.")
            elif element.tag == "force_units":
                self.force_units = element.text
                if self.force_units != "N":
                    raise RuntimeError("Force units must be in newtons.")
            else:
                raise RuntimeError(
                    f"Element {element.tag} not recognize. Please verify your xml file or send an issue"
                    f" in the github repository."
                )

        self.bodies = []
        self.forces = []
        self.joints = []
        self.markers = []
        self.constraint_set = []
        self.controller_set = []
        self.prob_set = []
        self.component_set = []
        self.geometry_set = []
        self.warnings = []

        # TODO: @Pariterre: How should I have done this ?
        self.output_model = output_model

    @staticmethod
    def _is_element_empty(element):
        if element:
            if not element[0].text:
                return True
            else:
                return False
        else:
            return True

    def get_body_set(self, body_set=None):
        bodies = []
        body_set = body_set if body_set else self.bodyset_elt[0]
        if self._is_element_empty(body_set):
            return None
        else:
            for element in body_set:
                name, mass, inertia, center_of_mass  = Body().get_body_attrib(element).return_segment_attrib()
                self.output_model.segments[name] = SegmentReal(
                    name=name,
                    mass=mass,
                    inertia=inertia,
                    center_of_mass=center_of_mass,
                    # translations=Translations.YZ,
                    # rotations=Rotations.X,
                    # mesh=MeshReal(((0, 0, 0), (0, 0, 0.53))),
                )
            return

    def get_body_mesh_list(self, body_set=None) -> list[str]:
        """returns the list of vtp files included in the model"""
        body_mesh_list = []
        body_set = body_set if body_set else self.bodyset_elt[0]
        if self._is_element_empty(body_set):
            return None
        else:
            for element in body_set:
                body_mesh_list.extend(Body().get_body_attrib(element).mesh)
            return body_mesh_list

    def get_marker_set(self, markers_to_ignore: list[str]):
        markers = []
        if self._is_element_empty(self.markerset_elt):
            return None
        else:
            original_marker_names = []
            for element in self.markerset_elt[0]:
                marker = Marker().get_marker_attrib(element)
                original_marker_names += [marker.name]
                if marker.name not in markers_to_ignore:
                    markers.append(marker)
            # for marker_to_ignore in markers_to_ignore:
            #     if marker_to_ignore not in original_marker_names:
            #         raise RuntimeError(
            #             f"The marker {marker_to_ignore} cannot be ignored as it is not present in the original osim model."
            #         )
            return markers

    def get_force_set(self, ignore_muscle_applied_tag=False, muscles_to_ignore=None):
        forces = []
        wrap = []
        original_muscle_names = []
        if self._is_element_empty(self.forceset_elt):
            return None
        else:
            for element in self.forceset_elt[0]:
                if "Muscle" in element.tag:
                    original_muscle_names += [(element.attrib["name"]).split("/")[-1]]
                    current_muscle = Muscle().get_muscle_attributes(element, ignore_muscle_applied_tag, muscles_to_ignore)
                    # TODO: insersion = current_muscle.return_muscle_attrib()
                    if current_muscle is not None:
                        forces.append(current_muscle)
                        if forces[-1].wrap:
                            wrap.append(forces[-1].name)
                elif "Force" in element.tag or "Actuator" in element.tag:
                    self.warnings.append(
                        f"Some {element.tag} were present in the original file force set. "
                        "Only muscles are supported so they will be ignored."
                    )
            if len(wrap) != 0:
                self.warnings.append(
                    f"Some wrapping objects were present on the muscles :{wrap} in the original file force set.\n"
                    "Only via point are supported in biomod so they will be ignored."
                )

            for muscle_to_ignore in muscles_to_ignore:
                if muscle_to_ignore not in original_muscle_names:
                    raise RuntimeError(
                        f"The muscle {muscle_to_ignore} cannot be ignored as it is not present in the original osim model."
                    )

            return forces

    def get_joint_set(self, ignore_fixed_dof_tag=False, ignore_clamped_dof_tag=False):
        joints = []
        if self._is_element_empty(self.forceset_elt):
            return None
        else:
            for element in self.jointset_elt[0]:
                joints.append(Joint().get_joint_attrib(element, ignore_fixed_dof_tag, ignore_clamped_dof_tag))
                if joints[-1].function:
                    self.warnings.append(
                        f"Some functions were present for the {joints[-1].name} joint. "
                        "This feature is not implemented in biorbd yet so it will be ignored."
                    )
            # joints = self._reorder_joints(joints)
            return joints

    @staticmethod
    def add_markers_to_bodies(bodies, markers):
        for b, body in enumerate(bodies):
            # TODO: Do not add a try here. If the you can know in advance the error, test it with a if. If you actually need a try, catch a specific error (`except ERRORNAME:` instead of `except:`)
            try:
                for marker in markers:
                    if body.socket_frame == marker.parent:
                        bodies[b].markers.append(marker)
            except:
                pass
        return bodies

    @staticmethod
    def _reorder_joints(joints: list):
        # TODO: This function is not actually called. Is it necessary?
        ordered_joints = [joints[0]]
        joints.pop(0)
        while len(joints) != 0:
            for o, ord_joint in enumerate(ordered_joints):
                idx = []
                for j, joint in enumerate(joints):
                    if joint.parent == ord_joint.child:
                        ordered_joints = ordered_joints + [joint]
                        idx.append(j)
                    elif ord_joint.parent == joint.child:
                        ordered_joints = [joint] + ordered_joints
                        idx.append(j)
                if len(idx) != 0:
                    joints.pop(idx[0])
                elif len(idx) > 1:
                    raise RuntimeError("Two segment can't have the same parent in a biomod.")
        return ordered_joints

    def get_controller_set(self):
        if self._is_element_empty(self.controllerset_elt):
            self.controller_set = None
        else:
            self.warnings.append(
                "Some controllers were present in the original file. "
                "This feature is not implemented in biorbd yet so it will be ignored."
            )

    def get_constraint_set(self):
        if self._is_element_empty(self.constraintset_elt):
            self.constraintset_elt = None
        else:
            self.warnings.append(
                "Some constraints were present in the original file. "
                "This feature is not implemented in biorbd yet so it will be ignored."
            )

    def get_contact_geometry_set(self):
        if self._is_element_empty(self.contact_geometryset_elt):
            self.contact_geometryset_elt = None
        else:
            self.warnings.append(
                "Some contact geometry were present in the original file. "
                "This feature is not implemented in biorbd yet so it will be ignored."
            )

    def get_component_set(self):
        if self._is_element_empty(self.componentset_elt):
            self.componentset_elt = None
        else:
            self.warnings.append(
                "Some additional components were present in the original file. "
                "This feature is not implemented in biorbd yet so it will be ignored."
            )

    def get_probe_set(self):
        if self._is_element_empty(self.probeset_elt):
            self.probeset_elt = None
        else:
            self.warnings.append(
                "Some probes were present in the original file. "
                "This feature is not implemented in biorbd yet so it will be ignored."
            )

    def get_warnings(self):
        self.get_probe_set()
        self.get_component_set()
        self.get_contact_geometry_set()
        self.get_constraint_set()
        return self.warnings

    def get_file_version(self):
        return int(self.model.getroot().attrib["Version"])

    def get_infos(self):
        if self.publications:
            self.infos["Publication"] = self.publications
        if self.credit:
            self.infos["Credit"] = self.credit
        if self.force_units:
            self.infos["Force units"] = self.force_units
        if self.length_units:
            self.infos["Length units"] = self.length_units

    def read(self):

        self.ground = self.get_body_set(body_set=[self.ground_elt])
        self.forces = self.get_force_set(ignore_muscle_applied_tag=False, muscles_to_ignore=None)
        self.joints = self.get_joint_set(ignore_fixed_dof_tag, ignore_clamped_dof_tag)
        self.bodies = self.get_body_set()
        self.markers = self.get_marker_set(markers_to_ignore=markers_to_ignore)
        self.infos, self.warnings = self.infos, self.get_warnings()
        self.ground = self.add_markers_to_bodies(self.ground, self.markers)

        self.bodies = self.add_markers_to_bodies(self.bodies, self.markers)
        self.muscle_type = muscle_type if muscle_type else "hilldegroote"
        self.state_type = state_type
        self.mesh_dir = "Geometry" if mesh_dir is None else mesh_dir

        self.new_mesh_dir = self.mesh_dir + "_cleaned"
        self.vtp_polygons_to_triangles = vtp_polygons_to_triangles
        self.vtp_files = self.get_body_mesh_list()

        if isinstance(self.muscle_type, MuscleType):
            self.muscle_type = self.muscle_type.value
        if isinstance(self.state_type, MuscleStateType):
            self.state_type = self.state_type.value

        if isinstance(muscle_type, str):
            if self.muscle_type not in [e.value for e in MuscleType]:
                raise RuntimeError(f"Muscle of type {self.muscle_type} is not a biorbd muscle.")

        if isinstance(muscle_type, str):
            if self.state_type not in [e.value for e in MuscleStateType]:
                raise RuntimeError(f"Muscle state type {self.state_type} is not a biorbd muscle state type.")
        if self.state_type == "default":
            self.state_type = None


    def convert_file(self):

        if self.vtp_polygons_to_triangles:
            self.convert_vtp_to_triangles()

        # segment
        self.writer.write("\n// SEGMENT DEFINITION\n")
        # Handle the ground frame as a segment
        if self.ground:
            body = self.ground[0]
            dof = Joint()
            dof.child_offset_trans, dof.child_offset_rot = [0] * 3, [0] * 3
            self.writer.write_dof(
                body,
                dof,
                self.new_mesh_dir if self.vtp_polygons_to_triangles else self.mesh_dir,
                skip_virtual=True,
                parent="base",
            )
            self.writer.write(f"\n\t// Markers\n")
            for marker in body.markers:
                self.writer.write_marker(marker)
        for dof in self.joints:
            for body in self.bodies:
                if body.socket_frame == dof.child_body:
                    self.writer.write_dof(
                        body,
                        dof,
                        self.new_mesh_dir if self.vtp_polygons_to_triangles else self.mesh_dir,
                    )
                    self.writer.write(f"\n\t// Markers\n")
                    for marker in body.markers:
                        self.writer.write_marker(marker)

        muscle_groups = []
        for muscle in self.forces:
            group = muscle.group
            if group not in muscle_groups:
                muscle_groups.append(group)

        self.writer.write("\n// MUSCLE DEFINIION\n")
        muscles = self.forces
        while len(muscles) != 0:
            for muscle_group in muscle_groups:
                idx = []
                self.writer.write_muscle_group(muscle_group)
                for m, muscle in enumerate(muscles):
                    if muscle.group == muscle_group:
                        if not muscle.applied:
                            self.writer.write("\n/*")
                        self.writer.write_muscle(muscle, self.muscle_type, self.state_type)
                        for via_point in muscle.via_point:
                            self.writer.write_via_point(via_point)
                        if not muscle.applied:
                            self.writer.write("*/\n")
                        idx.append(m)
                count = 0
                for i in idx:
                    muscles.pop(i - count)
                    count += 1
            muscle_groups.pop(0)

        if self.print_warnings:
            self.writer.write("\n/*-------------- WARNINGS---------------\n")
            for warning in self.warnings:
                self.writer.write("\n" + warning)
            self.writer.write("*/\n")
        self.writer.biomod_file.close()
        print(f"\nYour file {self.osim_path} has been converted into {self.biomod_path}.")

    def convert_vtp_to_triangles(self):
        """
        Convert vtp mesh to triangles mesh
        """
        mesh_dir_relative = os.path.join(self.biomod_path[: self.biomod_path.rindex("/")], self.mesh_dir)
        new_mesh_dir_relative = os.path.join(self.biomod_path[: self.biomod_path.rindex("/")], self.new_mesh_dir)

        if not os.path.exists(new_mesh_dir_relative):
            os.makedirs(new_mesh_dir_relative)

        log_file_failed = []
        print("Cleaning vtp file into triangles: ")

        # select only files in such that filename.endswith('.vtp') or filename in self.vtp_files
        files = [
            filename
            for filename in os.listdir(mesh_dir_relative)
            if filename.endswith(".vtp") and filename in self.vtp_files
        ]

        for filename in files:
            complete_path = os.path.join(mesh_dir_relative, filename)

            with open(complete_path, "r") as f:
                print(complete_path)
                try:
                    mesh = read_vtp_file(complete_path)
                    if mesh["polygons"].shape[1] == 3:  # it means it doesn't need to be converted into triangles
                        shutil.copy(complete_path, new_mesh_dir_relative)
                    else:
                        poly, nodes, normals = transform_polygon_to_triangles(
                            mesh["polygons"], mesh["nodes"], mesh["normals"]
                        )
                        new_mesh = dict(polygons=poly, nodes=nodes, normals=normals)

                        write_vtp_file(new_mesh, new_mesh_dir_relative, filename)

                except:
                    print(f"Error with {filename}")
                    log_file_failed.append(filename)
                    # if failed we just copy the file in the new folder
                    shutil.copy(complete_path, new_mesh_dir_relative)

        if len(log_file_failed) > 0:
            print("Files failed to clean:")
            print(log_file_failed)


class Body:
    def __init__(self):
        self.name = None
        self.mass = None
        self.inertia = None
        self.mass_center = None
        self.wrap = None
        self.socket_frame = None
        self.markers = []
        self.mesh = []
        self.mesh_color = []
        self.mesh_scale_factor = []
        self.mesh_offset = []
        self.virtual_body = []

    def get_body_attrib(self, element):
        self.name = (element.attrib["name"]).split("/")[-1]
        self.mass = find(element, "mass")
        self.inertia = find(element, "inertia")
        self.mass_center = find(element, "mass_center")
        geometry = element.find("FrameGeometry")
        self.socket_frame = self.name
        if geometry:
            self.socket_frame = geometry.find("socket_frame").text.split("/")[-1]
            if self.socket_frame == "..":
                self.socket_frame = self.name

        if element.find("WrapObjectSet") is not None:
            self.wrap = True if len(element.find("WrapObjectSet").text) != 0 else False

        if element.find("attached_geometry") is not None:
            mesh_list = element.find("attached_geometry").findall("Mesh")
            mesh_list = extend_mesh_list_with_extra_components(mesh_list, element)

            for mesh in mesh_list:
                self.mesh.append(mesh[0].find("mesh_file").text)
                self.virtual_body.append(mesh[0].attrib["name"])
                mesh_scale_factor = mesh[0].find("scale_factors")
                self.mesh_scale_factor.append(mesh_scale_factor.text if mesh_scale_factor is not None else None)
                if mesh[0].find("Appearance") is not None:
                    mesh_color = mesh[0].find("Appearance").find("color")
                    self.mesh_color.append(mesh_color.text if mesh_color is not None else None)
                self.mesh_offset.append(mesh[1])
        return self

    def return_segment_attrib(self):
        name = self.name
        mass = 1e-8 if not self.mass else float(self.mass.text)
        inertia = np.eye(3) if not self.inertia else np.array([float(i) for i in self.inertia.text.split(" ")])
        center_of_mass = np.zeros(3) if not self.mass_center else np.array([float(i) for i in self.mass_center.text.split(" ")])
        # Add meshes
        return name, mass, inertia, center_of_mass


def extend_mesh_list_with_extra_components(mesh_list, element) -> list[tuple[str, OrthoMatrix]]:
    """Convert mesh_list from list[str] to list[tuple(str, OrthoMatrix)] to include offset in some meshes"""
    mesh_list_and_offset = [(mesh, OrthoMatrix()) for mesh in mesh_list]

    if element.find("components"):
        frames = element.find("components").findall("PhysicalOffsetFrame")
        for frame in frames:
            if frame.find("attached_geometry") is not None:
                translation = frame.find("translation").text
                translation_tuple = tuple([float(t) for t in translation.split(" ")])
                tuple_mesh_and_offset = (
                    frame.find("attached_geometry").find("Mesh"),
                    OrthoMatrix(translation=translation_tuple),
                )
                mesh_list_and_offset.append(tuple_mesh_and_offset)

    return mesh_list_and_offset


class Joint:
    def __init__(self):
        self.parent = None
        self.child = None
        self.type = None
        self.name = None
        self.coordinates = []
        self.parent_offset_trans = []
        self.parent_offset_rot = []
        self.child_offset_trans = []
        self.child_offset_rot = []
        self.child_body = None
        self.parent_body = None
        self.spatial_transform = []
        self.implemented_joint = [""]
        self.function = False

    def get_joint_attrib(self, element, ignore_fixed, ignore_clamped):
        self.type = element.tag
        if self.type not in [e.value for e in JointType]:
            raise RuntimeError(
                f"Joint type {self.type} is not implemented yet."
                f"Allowed joint type are: {[e.value for e in JointType]}"
            )
        self.name = (element.attrib["name"]).split("/")[-1]
        self.parent = find(element, "socket_parent_frame").split("/")[-1]
        self.child = find(element, "socket_child_frame").split("/")[-1]

        if element.find("coordinates") is not None:
            for coordinate in element.find("coordinates").findall("Coordinate"):
                self.coordinates.append(Coordinate().get_coordinate_attrib(coordinate, ignore_fixed, ignore_clamped))

        if element.find("SpatialTransform") is not None:
            for i, transform in enumerate(element.find("SpatialTransform").findall("TransformAxis")):
                spat_transform = SpatialTransform().get_transform_attrib(transform)
                if i < 3:
                    spat_transform.type = "rotation"
                else:
                    spat_transform.type = "translation"
                for coordinate in self.coordinates:
                    if coordinate.name == spat_transform.coordinate_name:
                        spat_transform.coordinate = coordinate
                self.function = spat_transform.function
                self.spatial_transform.append(spat_transform)

        for frame in element.find("frames").findall("PhysicalOffsetFrame"):
            if self.parent == frame.attrib["name"]:
                self.parent_body = frame.find("socket_parent").text.split("/")[-1]
                self.parent_offset_rot = frame.find("orientation").text
                self.parent_offset_trans = frame.find("translation").text
            elif self.child == frame.attrib["name"]:
                self.child_body = frame.find("socket_parent").text.split("/")[-1]
                offset_rot = frame.find("orientation").text
                offset_trans = frame.find("translation").text
                offset_trans = [float(i) for i in offset_trans.split(" ")]
                offset_rot = [-float(i) for i in offset_rot.split(" ")]
                self.child_offset_trans, self.child_offset_rot = self._convert_offset_child(offset_rot, offset_trans)
        return self

    @staticmethod
    def _convert_offset_child(offset_child_rot, offset_child_trans):
        R = compute_matrix_rotation(offset_child_rot).T
        new_translation = -np.dot(R.T, offset_child_trans)
        new_rotation = -rot2eul(R)
        new_rotation_str = ""
        new_translation_str = ""
        for i in range(3):
            if i == 0:
                pass
            else:
                new_rotation_str += " "
                new_translation_str += " "
            new_rotation_str += str(new_rotation[i])
            new_translation_str += str(new_translation[i])
        return new_translation, new_rotation


class Coordinate:
    def __init__(self):
        self.name = None
        self.type = None
        self.default_value = None
        self.range = []
        self.clamped = True
        self.locked = False

    def get_coordinate_attrib(self, element, ignore_fixed=False, ignore_clamped=False):
        self.name = (element.attrib["name"]).split("/")[-1]
        self.default_value = find(element, "default_value")
        self.range = find(element, "range")
        if not ignore_clamped:
            clamped = find(element, "clamped")
            self.clamped = clamped == "true" if clamped else False

        if not ignore_fixed:
            locked = find(element, "locked")
            self.locked = locked == "true" if locked else False
        return self


class SpatialTransform:
    def __init__(self):
        self.name = None
        self.type = None
        self.coordinate_name = None
        self.coordinate = []
        self.axis = None
        self.function = False

    def get_transform_attrib(self, element):
        self.name = (element.attrib["name"]).split("/")[-1]
        self.coordinate_name = find(element, "coordinates")
        self.axis = find(element, "axis")
        for elt in element[0]:
            if "Function" in elt.tag and len(elt.text) != 0:
                self.function = True
        return self

class Muscle:
    def __init__(self):
        self.name = None
        self.via_point = []
        self.type = None
        self.origin = None
        self.insersion = None
        self.optimal_length = None
        self.maximal_force = None
        self.tendon_slack_length = None
        self.pennation_angle = None
        self.applied = True
        self.pcsa = None
        self.maximal_velocity = None
        self.wrap = False
        self.group = None
        self.state_type = None


    def get_muscle_attrib(self, element, ignore_applied, muscles_to_ignore):
        name = (element.attrib["name"]).split("/")[-1]
        if name in muscles_to_ignore:
            return None
        self.name = name
        self.maximal_force = find(element, "max_isometric_force")
        self.optimal_length = find(element, "optimal_fiber_length")
        self.tendon_slack_length = find(element, "tendon_slack_length")
        self.pennation_angle = find(element, "pennation_angle_at_optimal")
        self.maximal_velocity = find(element, "max_contraction_velocity")

        if element.find("appliesForce") is not None and not ignore_applied:
            self.applied = element.find("appliesForce").text == "true"

        for path_point_elt in element.find("GeometryPath").find("PathPointSet")[0].findall("PathPoint"):
            via_point = PathPoint().get_path_point_attrib(path_point_elt)
            via_point.muscle = self.name
            self.via_point.append(via_point)
        self.group = [self.via_point[0].body, self.via_point[-1].body]
        for i in range(len(self.via_point)):
            self.via_point[i].muscle_group = f"{self.group[0]}_to_{self.group[1]}"

        if element.find("GeometryPath").find("PathWrapSet") is not None:
            try:
                wrap = element.find("GeometryPath").find("PathWrapSet")[0].text
            except:
                wrap = 0
            n_wrap = 0 if not wrap else len(wrap)
            self.wrap = True if n_wrap != 0 else False

        self.insersion = self.via_point[-1].position
        self.origin = self.via_point[0].position
        self.via_point = self.via_point[1:-1]

        return self

    def return_muscle_attrib(self):
        insersion = None if not self.insersion else float(self.insersion)
        # TODO: via points
        return insersion


class PathPoint:
    def __init__(self):
        self.name = None
        self.muscle = None
        self.body = None
        self.muscle_group = None
        self.position = None

    def get_path_point_attrib(self, element):
        self.name = element.attrib["name"]
        self.body = find(element, "socket_parent_frame").split("/")[-1]
        self.position = find(element, "location")
        return self


class Marker:
    def __init__(self):
        self.name = None
        self.parent = None
        self.position = None
        self.fixed = True

    def get_marker_attrib(self, element):
        self.name = (element.attrib["name"]).split("/")[-1]
        self.position = find(element, "location")
        self.parent = find(element, "socket_parent_frame").split("/")[-1]
        fixed = find(element, "fixed")
        self.fixed = fixed == "true" if fixed else None
        return self