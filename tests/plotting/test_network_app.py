import pytest

import flixopt as fx

from ..conftest import (
    flow_system_long,
    flow_system_segments_of_flows_2,
    simple_flow_system,
)


@pytest.fixture(params=[simple_flow_system, flow_system_segments_of_flows_2, flow_system_long])
def flow_system(request):
    fs = request.getfixturevalue(request.param.__name__)
    if isinstance(fs, fx.FlowSystem):
        return fs
    else:
        return fs[0]


def test_network_app(flow_system):
    """Test that flow model constraints are correctly generated."""
    flow_system.start_network_app()
    flow_system.stop_network_app()
