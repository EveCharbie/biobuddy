from lxml import etree

from ..utils_xml import find_in_tree, match_tag


class SpatialTransform:
    def __init__(self, name: str, type: str, coordinate_name: str, coordinate: list, axis: str, function: bool):
        self.name = name
        self.type = type
        self.coordinate_name = coordinate_name
        self.coordinate = coordinate
        self.axis = axis
        self.function = function

    @staticmethod
    def from_element(element: etree.ElementTree, parent_name: str) -> "SpatialTransform":
        function = False
        for elt in element:
            if match_tag(elt, "Function") and len(elt.text) != 0:
                function = True
            elif match_tag(elt, "MultiplierFunction") and len(elt.text) != 0:
                function = True
            elif match_tag(elt, "SimmSpline") and len(elt.text) != 0:
                function = True

        if function:
            dof_name = None
            coordinate = None
        else:
            coordinate_name = find_in_tree(element, "coordinates")
            dof_name = f"{parent_name}_{coordinate_name}"
            coordinate = find_in_tree(element, "coordinate")

        return SpatialTransform(
            name=(element.attrib["name"]).split("/")[-1],
            type=find_in_tree(element, "type"),
            coordinate_name=dof_name,
            coordinate=coordinate,
            axis=find_in_tree(element, "axis"),
            function=function,
        )
