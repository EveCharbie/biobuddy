from typing import Callable

from ..rigidbody.marker import Marker
from ...real.biomechanical_model_real import BiomechanicalModelReal
from ...real.rigidbody.axis_real import AxisReal
from ...real.rigidbody.segment_coordinate_system_real import SegmentCoordinateSystemReal
from ....utils.protocols import Data


class Axis:
    class Name(AxisReal.Name):
        """
        A copy of AxisReal.Name
        """

        pass

    def __init__(self, name: AxisReal.Name, start: Callable | str, end: Callable | str):
        """
        Parameters
        ----------
        name
            The AxisName of the Axis
        start
            The function (f(m) -> np.ndarray, where m is a dict of markers) that defines the starting point of the axis.
            If a str is provided, the position of the corresponding marker is used
        end
            The function (f(m) -> np.ndarray, where m is a dict of markers) that defines the end point of the axis.
            If a str is provided, the position of the corresponding marker is used
        """
        self.name = name
        name_start = start if isinstance(start, str) else "start"
        name_end = end if isinstance(end, str) else "end"
        self.start = Marker(name=name_start, function=start)
        self.end = Marker(name=name_end, function=end)

    def to_axis(
        self, data: Data, kinematic_chain: BiomechanicalModelReal, parent_scs: SegmentCoordinateSystemReal = None
    ) -> AxisReal:
        """
        Compute the axis from actual data
        Parameters
        ----------
        data
            The actual data
        kinematic_chain
            The model as it is constructed at that particular time. It is useful if some values must be obtained from
            previously computed values
        parent_scs
            The transformation from global to local
        """

        start = self.start.to_marker(data, kinematic_chain, parent_scs)
        end = self.end.to_marker(data, kinematic_chain, parent_scs)
        return AxisReal(self.name, start, end)
