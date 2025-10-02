from lxml import etree

from .utils import find_in_tree, find_sub_elements_in_tree, match_tag
from ...components.real.muscle.wrapping_object_real import WrappingObjectReal, WrappingEllipsoid, PathWrapMethod


class PathWrap:
    def __init__(
            self,
            name: str,
            wrap_object_name: str,
            method: PathWrapMethod,
            range_min: str,
            range_max: str,
    ):
        self.name = name
        self.wrap_object_name = wrap_object_name
        self.method = method
        self.range_min = range_min
        self.range_max = range_max

    @staticmethod
    def from_element(element: etree.ElementTree) -> "PathWrap":
        name = element.attrib["name"]
        wrap_object_name = find_in_tree(element, "wrap_object")
        method = PathWrapMethod(find_in_tree(element, "method"))
        range_min = find_in_tree(element, "range").split(" ")[0],
        range_max = find_in_tree(element, "range").split(" ")[1],
        return PathWrap(
            name=name,
            wrap_object_name=wrap_object_name,
            method=method,
            range_min=range_min,
            range_max=range_max,
        )

def wrapping_object_set_from_element(element: etree.ElementTree) -> tuple[WrappingObjectReal, str]:
    WrapObjectSet
    if wrapping_type == "Ellipsoid":
        return (
            WrappingEllipsoid(
                dof_name=find_in_tree(element, "socket_coordinate").split("/")[-1],
                joint_name=joint_name,
                range_min=find_in_tree(element, "range").split(" ")[0],
                range_max=find_in_tree(element, "range").split(" ")[1],
            ),
            "",
        )

