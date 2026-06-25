from copy import deepcopy
import numpy as np

from ....utils.named_list import NamedList
from ..force.muscle_real import MuscleReal
from ..force.via_point_real import ViaPointReal


class MuscleGroupReal:
    def __init__(
        self,
        name: str,
        origin_parent_name: str,
        insertion_parent_name: str,
    ):
        """
        Parameters
        ----------
        name
            The name of the new muscle group
        origin_parent_name
            The name of the parent segment for this muscle group
        insertion_parent_name
            The name of the insertion segment for this muscle group
        """
        # Sanity checks
        if origin_parent_name == insertion_parent_name and origin_parent_name != "":
            raise ValueError("The origin and insertion parent names cannot be the same.")

        self.name = name
        self.origin_parent_name = origin_parent_name
        self.insertion_parent_name = insertion_parent_name
        self.muscles = NamedList[MuscleReal]()

    def add_muscle(self, muscle: MuscleReal) -> None:
        """
        Add a muscle to the model

        Parameters
        ----------
        muscle
            The muscle to add
        """
        if muscle.muscle_group is not None and muscle.muscle_group != self.name:
            raise ValueError(
                "The muscle's muscle_group should be the same as the 'key'. Alternatively, muscle.muscle_group can be left undefined"
            )

        muscle.muscle_group = self.name
        self.muscles._append(muscle)

    def remove_muscle(self, muscle_name: str) -> None:
        """
        Remove a muscle from the model

        Parameters
        ----------
        muscle_name
            The name of the muscle to remove
        """
        self.muscles._remove(muscle_name)

    def merge_muscles(self, muscle_names: str) -> None:
        """
        Merge muscles together

        Parameters
        ----------
        muscle_names
            The name of the muscles to merge together
        """
        new_muscle_name = ""
        origin_position = []
        insertion_position = []
        optimal_length = []
        maximal_force = []
        tendon_slack_length = []
        pennation_angle = []
        maximal_velocity = []
        maximal_excitation = []
        via_points = {}

        for muscle_name in muscle_names:
            if self.muscles[muscle_name].muscle_group != self.name:
                raise ValueError(f"Muscle {muscle_name} does not belong to muscle group {self.name}")

            if len(new_muscle_name) == 0:
                # First muscle
                new_muscle_name = deepcopy(muscle_name)
                muscle_type = deepcopy(self.muscles[muscle_name].muscle_type)
                state_type = deepcopy(self.muscles[muscle_name].state_type)
                for via_point in self.muscles[muscle_name].via_points:
                    if via_point.condition is not None:
                        raise NotImplementedError(
                            f"Conditional via points not implemented for muscle, got a conditional via point in {via_point.name} in the muscle {muscle_name}"
                        )
                    if via_point.movement is not None:
                        raise NotImplementedError(
                            f"Moving via points not implemented for muscle, got a movement via point in {via_point.name} in the muscle {muscle_name}"
                        )
                    via_points[via_point.name] = {
                        "position": [via_point.position],
                        "parent": via_point.parent_name,
                    }
            else:
                # Other muscles
                new_muscle_name += f"_and_{deepcopy(muscle_name)}"
                if self.muscles[muscle_name].muscle_type != muscle_type:
                    raise ValueError(
                        f"The muscle type of the muscles is not all the same, "
                        f"got {muscle_type} and {self.muscles[muscle_name].muscle_type}"
                    )
                if self.muscles[muscle_name].state_type != state_type:
                    raise ValueError(
                        f"The muscle state type of the muscles is not all the same, "
                        f"got {state_type} and {self.muscles[muscle_name].state_type}"
                    )
                if self.muscles[muscle_name].nb_via_points != len(via_points.keys()):
                    raise ValueError(
                        f"The are not the same number of via points for the muscle {muscle_name} ({self.muscles[muscle_name].nb_via_points}) and"
                        f"for the muscle {muscle_names[0]} ({len(via_points.keys())})."
                    )
                for via_point in self.muscles[muscle_name].via_points:
                    if via_point.condition is not None:
                        raise NotImplementedError(
                            f"Conditional via points not implemented for muscle, got a conditional via point in {via_point.name} in the muscle {muscle_name}"
                        )
                    if via_point.movement is not None:
                        raise NotImplementedError(
                            f"Moving via points not implemented for muscle, got a movement via point in {via_point.name} in the muscle {muscle_name}."
                        )
                    # Find the right via point
                    candidates = []
                    distance = []
                    for via_name in via_points.keys():
                        if via_points[via_name]["parent"] == via_point.parent_name:
                            candidates += [via_name]
                            distance += [
                                np.linalg.norm(via_points[via_name]["position"][0][:3] - via_point.position[:3])
                            ]
                    if len(candidates) == 0:
                        raise ValueError(
                            f"The via_point {via_point.name} from muscle {muscle_name} could not be "
                            f"matched with any via points of muscle {muscle_names[0]} with via points {via_points.keys()}"
                        )
                    via_point_index = np.argmin(np.array(distance))
                    via_points[candidates[via_point_index]]["position"] += [via_point.position]

            origin_position += [self.muscles[muscle_name].origin_position.position]
            insertion_position += [self.muscles[muscle_name].insertion_position.position]
            if self.muscles[muscle_name].origin_position.position is not None:
                optimal_length += [self.muscles[muscle_name].optimal_length]
            if self.muscles[muscle_name].maximal_force is not None:
                maximal_force += [self.muscles[muscle_name].maximal_force]
            if self.muscles[muscle_name].tendon_slack_length is not None:
                tendon_slack_length += [self.muscles[muscle_name].tendon_slack_length]
            if self.muscles[muscle_name].pennation_angle is not None:
                pennation_angle += [self.muscles[muscle_name].pennation_angle]
            if self.muscles[muscle_name].maximal_velocity is not None:
                maximal_velocity += [self.muscles[muscle_name].maximal_velocity]
            if self.muscles[muscle_name].maximal_excitation is not None:
                maximal_excitation += [self.muscles[muscle_name].maximal_excitation]

        # Add the new merged muscle
        self.add_muscle(
            MuscleReal(
                name=new_muscle_name,
                muscle_type=muscle_type,
                state_type=state_type,
                muscle_group=self.name,
                origin_position=ViaPointReal(
                    name=f"origin_{new_muscle_name}",
                    parent_name=self.origin_parent_name,
                    position=np.mean(np.array(origin_position), axis=0),
                ),
                insertion_position=ViaPointReal(
                    name=f"insertion_{new_muscle_name}",
                    parent_name=self.insertion_parent_name,
                    position=np.mean(np.array(insertion_position), axis=0),
                ),
                optimal_length=np.mean(np.array(optimal_length), axis=0),
                maximal_force=np.sum(np.array(maximal_force)),
                tendon_slack_length=np.mean(np.array(tendon_slack_length), axis=0),
                pennation_angle=np.mean(np.array(pennation_angle), axis=0),
                maximal_velocity=np.mean(np.array(maximal_velocity), axis=0),
                maximal_excitation=np.mean(np.array(maximal_excitation), axis=0),
            )
        )
        for name, via in via_points.items():
            self.muscles[new_muscle_name].add_via_point(
                ViaPointReal(
                    name=name,
                    parent_name=via["parent"],
                    position=np.mean(np.array(via["position"]), axis=0),
                )
            )

        # Remove the original muscles
        for muscle_name in muscle_names:
            self.remove_muscle(muscle_name)

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str):
        self._name = value

    @property
    def origin_parent_name(self) -> str:
        return self._origin_parent_name

    @origin_parent_name.setter
    def origin_parent_name(self, value: str):
        self._origin_parent_name = value

    @property
    def insertion_parent_name(self) -> str:
        return self._insertion_parent_name

    @insertion_parent_name.setter
    def insertion_parent_name(self, value: str):
        self._insertion_parent_name = value

    @property
    def nb_muscles(self):
        return len(self.muscles)

    @property
    def muscle_names(self):
        return [m.name for m in self.muscles]

    def to_biomod(self):
        # Define the print function, so it automatically formats things in the file properly
        out_string = f"musclegroup\t{self.name}\n"
        out_string += f"\tOriginParent\t{self.origin_parent_name}\n"
        out_string += f"\tInsertionParent\t{self.insertion_parent_name}\n"
        out_string += "endmusclegroup\n"

        out_string += "\n // ------ MUSCLES ------\n"
        for muscle in self.muscles:
            out_string += muscle.to_biomod()
        return out_string

    def to_urdf(self):
        raise NotImplementedError("Muscle groups are not implemented yet for URDF export")
