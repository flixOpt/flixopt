import pytest

from flixopt.effects import detect_cycles


def test_empty_graph():
    """Test that an empty graph has no cycles."""
    assert detect_cycles({}) == []


def test_single_node():
    """Test that a graph with a single node and no edges has no cycles."""
    assert detect_cycles({'A': []}) == []


def test_self_loop():
    """Test that a graph with a self-loop has a cycle."""
    cycles = detect_cycles({'A': ['A']})
    assert len(cycles) == 1
    assert cycles[0] == ['A', 'A']


def test_simple_cycle():
    """Test that a simple cycle is detected."""
    graph = {'A': ['B'], 'B': ['C'], 'C': ['A']}
    cycles = detect_cycles(graph)
    assert len(cycles) == 1
    assert cycles[0] == ['A', 'B', 'C', 'A'] or cycles[0] == ['B', 'C', 'A', 'B'] or cycles[0] == ['C', 'A', 'B', 'C']


def test_no_cycles():
    """Test that a directed acyclic graph has no cycles."""
    graph = {'A': ['B', 'C'], 'B': ['D', 'E'], 'C': ['F'], 'D': [], 'E': [], 'F': []}
    assert detect_cycles(graph) == []


def test_multiple_cycles():
    """Test that a graph with multiple cycles is detected."""
    graph = {'A': ['B', 'D'], 'B': ['C'], 'C': ['A'], 'D': ['E'], 'E': ['D']}
    cycles = detect_cycles(graph)
    assert len(cycles) == 2

    # Check that both cycles are detected (order might vary)
    cycle_strings = [','.join(cycle) for cycle in cycles]
    assert (
        any('A,B,C,A' in s for s in cycle_strings)
        or any('B,C,A,B' in s for s in cycle_strings)
        or any('C,A,B,C' in s for s in cycle_strings)
    )
    assert any('D,E,D' in s for s in cycle_strings) or any('E,D,E' in s for s in cycle_strings)


def test_hidden_cycle():
    """Test that a cycle hidden deep in the graph is detected."""
    graph = {
        'A': ['B', 'C'],
        'B': ['D'],
        'C': ['E'],
        'D': ['F'],
        'E': ['G'],
        'F': ['H'],
        'G': ['I'],
        'H': ['J'],
        'I': ['K'],
        'J': ['L'],
        'K': ['M'],
        'L': ['N'],
        'M': ['N'],
        'N': ['O'],
        'O': ['P'],
        'P': ['Q'],
        'Q': ['O'],  # Hidden cycle O->P->Q->O
    }
    cycles = detect_cycles(graph)
    assert len(cycles) == 1

    # Check that the O-P-Q cycle is detected
    cycle = cycles[0]
    assert 'O' in cycle and 'P' in cycle and 'Q' in cycle

    # Check that they appear in the correct order
    o_index = cycle.index('O')
    p_index = cycle.index('P')
    q_index = cycle.index('Q')

    # Check the cycle order is correct (allowing for different starting points)
    cycle_len = len(cycle)
    assert (
        (p_index == (o_index + 1) % cycle_len and q_index == (p_index + 1) % cycle_len)
        or (q_index == (o_index + 1) % cycle_len and p_index == (q_index + 1) % cycle_len)
        or (o_index == (p_index + 1) % cycle_len and q_index == (o_index + 1) % cycle_len)
    )


def test_disconnected_graph():
    """Test with a disconnected graph."""
    graph = {'A': ['B'], 'B': ['C'], 'C': [], 'D': ['E'], 'E': ['F'], 'F': []}
    assert detect_cycles(graph) == []


def test_disconnected_graph_with_cycle():
    """Test with a disconnected graph containing a cycle in one component."""
    graph = {
        'A': ['B'],
        'B': ['C'],
        'C': [],
        'D': ['E'],
        'E': ['F'],
        'F': ['D'],  # Cycle in D->E->F->D
    }
    cycles = detect_cycles(graph)
    assert len(cycles) == 1

    # Check that the D-E-F cycle is detected
    cycle = cycles[0]
    assert 'D' in cycle and 'E' in cycle and 'F' in cycle

    # Check if they appear in the correct order
    d_index = cycle.index('D')
    e_index = cycle.index('E')
    f_index = cycle.index('F')

    # Check the cycle order is correct (allowing for different starting points)
    cycle_len = len(cycle)
    assert (
        (e_index == (d_index + 1) % cycle_len and f_index == (e_index + 1) % cycle_len)
        or (f_index == (d_index + 1) % cycle_len and e_index == (f_index + 1) % cycle_len)
        or (d_index == (e_index + 1) % cycle_len and f_index == (d_index + 1) % cycle_len)
    )


def test_complex_dag():
    """Test with a complex directed acyclic graph."""
    graph = {
        'A': ['B', 'C', 'D'],
        'B': ['E', 'F'],
        'C': ['E', 'G'],
        'D': ['G', 'H'],
        'E': ['I', 'J'],
        'F': ['J', 'K'],
        'G': ['K', 'L'],
        'H': ['L', 'M'],
        'I': ['N'],
        'J': ['N', 'O'],
        'K': ['O', 'P'],
        'L': ['P', 'Q'],
        'M': ['Q'],
        'N': ['R'],
        'O': ['R', 'S'],
        'P': ['S'],
        'Q': ['S'],
        'R': [],
        'S': [],
    }
    assert detect_cycles(graph) == []


def test_missing_node_in_connections():
    """Test behavior when a node referenced in edges doesn't have its own key."""
    graph = {
        'A': ['B', 'C'],
        'B': ['D'],
        # C and D don't have their own entries
    }
    assert detect_cycles(graph) == []


def test_non_string_keys():
    """Test with non-string keys to ensure the algorithm is generic."""
    graph = {1: [2, 3], 2: [4], 3: [4], 4: []}
    assert detect_cycles(graph) == []

    graph_with_cycle = {1: [2], 2: [3], 3: [1]}
    cycles = detect_cycles(graph_with_cycle)
    assert len(cycles) == 1
    assert cycles[0] == [1, 2, 3, 1] or cycles[0] == [2, 3, 1, 2] or cycles[0] == [3, 1, 2, 3]


def test_complex_network_with_many_nodes():
    """Test with a large network to check performance and correctness."""
    graph = {}
    # Create a large DAG
    for i in range(100):
        # Connect each node to the next few nodes
        graph[i] = [j for j in range(i + 1, min(i + 5, 100))]

    # No cycles in this arrangement
    assert detect_cycles(graph) == []

    # Add a single back edge to create a cycle
    graph[99] = [0]  # This creates a cycle
    cycles = detect_cycles(graph)
    assert len(cycles) >= 1
    # The cycle might include many nodes, but must contain both 0 and 99
    any_cycle_has_both = any(0 in cycle and 99 in cycle for cycle in cycles)
    assert any_cycle_has_both


if __name__ == '__main__':
    pytest.main(['-v'])
