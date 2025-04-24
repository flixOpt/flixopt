import pytest

from flixopt.effects import has_cycles

def test_empty_graph():
    """Test that an empty graph has no cycles."""
    assert not has_cycles({})


def test_single_node():
    """Test that a graph with a single node and no edges has no cycles."""
    assert not has_cycles({"A": []})


def test_self_loop():
    """Test that a graph with a self-loop has a cycle."""
    assert has_cycles({"A": ["A"]})


def test_simple_cycle():
    """Test that a simple cycle is detected."""
    graph = {
        "A": ["B"],
        "B": ["C"],
        "C": ["A"]
    }
    assert has_cycles(graph)


def test_no_cycles():
    """Test that a directed acyclic graph has no cycles."""
    graph = {
        "A": ["B", "C"],
        "B": ["D", "E"],
        "C": ["F"],
        "D": [],
        "E": [],
        "F": []
    }
    assert not has_cycles(graph)


def test_multiple_cycles():
    """Test that a graph with multiple cycles is detected."""
    graph = {
        "A": ["B", "D"],
        "B": ["C"],
        "C": ["A"],
        "D": ["E"],
        "E": ["D"]
    }
    assert has_cycles(graph)


def test_hidden_cycle():
    """Test that a cycle hidden deep in the graph is detected."""
    graph = {
        "A": ["B", "C"],
        "B": ["D"],
        "C": ["E"],
        "D": ["F"],
        "E": ["G"],
        "F": ["H"],
        "G": ["I"],
        "H": ["J"],
        "I": ["K"],
        "J": ["L"],
        "K": ["M"],
        "L": ["N"],
        "M": ["N"],
        "N": ["O"],
        "O": ["P"],
        "P": ["Q"],
        "Q": ["O"]  # Hidden cycle O->P->Q->O
    }
    assert has_cycles(graph)


def test_disconnected_graph():
    """Test with a disconnected graph."""
    graph = {
        "A": ["B"],
        "B": ["C"],
        "C": [],
        "D": ["E"],
        "E": ["F"],
        "F": []
    }
    assert not has_cycles(graph)


def test_disconnected_graph_with_cycle():
    """Test with a disconnected graph containing a cycle in one component."""
    graph = {
        "A": ["B"],
        "B": ["C"],
        "C": [],
        "D": ["E"],
        "E": ["F"],
        "F": ["D"]  # Cycle in D->E->F->D
    }
    assert has_cycles(graph)


def test_complex_dag():
    """Test with a complex directed acyclic graph."""
    graph = {
        "A": ["B", "C", "D"],
        "B": ["E", "F"],
        "C": ["E", "G"],
        "D": ["G", "H"],
        "E": ["I", "J"],
        "F": ["J", "K"],
        "G": ["K", "L"],
        "H": ["L", "M"],
        "I": ["N"],
        "J": ["N", "O"],
        "K": ["O", "P"],
        "L": ["P", "Q"],
        "M": ["Q"],
        "N": ["R"],
        "O": ["R", "S"],
        "P": ["S"],
        "Q": ["S"],
        "R": [],
        "S": []
    }
    assert not has_cycles(graph)


def test_missing_node_in_connections():
    """Test behavior when a node referenced in edges doesn't have its own key."""
    graph = {
        "A": ["B", "C"],
        "B": ["D"]
        # C and D don't have their own entries
    }
    assert not has_cycles(graph)


def test_non_string_keys():
    """Test with non-string keys to ensure the algorithm is generic."""
    graph = {
        1: [2, 3],
        2: [4],
        3: [4],
        4: []
    }
    assert not has_cycles(graph)

    graph_with_cycle = {
        1: [2],
        2: [3],
        3: [1]
    }
    assert has_cycles(graph_with_cycle)


def test_complex_network_with_many_nodes():
    """Test with a large network to check performance and correctness."""
    graph = {}
    # Create a large DAG
    for i in range(100):
        # Connect each node to the next few nodes
        graph[i] = [j for j in range( i +1, min( i +5, 100))]

    # No cycles in this arrangement
    assert not has_cycles(graph)

    # Add a single back edge to create a cycle
    graph[99] = [0]  # This creates a cycle
    assert has_cycles(graph)


if __name__ == "__main__":
    pytest.main(["-v"])