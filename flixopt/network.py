from dash import Dash, html, dcc, Input, Output
import dash_cytoscape as cyto
import networkx
import socket
import logging

from .flow_system import FlowSystem
from .elements import Bus, Flow, Component
from .components import Sink, Source, SourceAndSink, Storage, LinearConverter

logger = logging.getLogger('flixopt')


def flow_graph(flow_system: FlowSystem) -> networkx.DiGraph:
    nodes = list(flow_system.components.values()) + list(flow_system.buses.values())
    edges = list(flow_system.flows.values())

    def get_color(element):
        if isinstance(element, Flow):
            raise TypeError('Flow graph shape not yet implemented')
        if isinstance(element, Bus):
            return '#7F8C8D'
        if isinstance(element, (Sink, Source, SourceAndSink)):
            return '#F1C40F'
        if isinstance(element, Storage):
            return '#2980B9'
        if isinstance(element, LinearConverter):
            return '#D35400'
        return '#27AE60'

    def get_shape(element):

        if isinstance(element, Bus):
            return 'ellipse'
        if isinstance(element, (Source)):
            return 'custom-source'
        if isinstance(element, (Sink, SourceAndSink)):
            return 'custom-sink'
        return 'rectangle'

    graph = networkx.DiGraph()  # Directed Graph using networkx

    for node in nodes:
        graph.add_node(
            node.label_full,
            color=get_color(node),
            shape=get_shape(node),
            #type
            parameters=node.__str__(),
        )

    for edge in edges:
        graph.add_edge(
            u_of_edge=edge.bus if edge.is_input_in_component else edge.component,
            v_of_edge=edge.component if edge.is_input_in_component else edge.bus,
            label=edge.label_full,
            parameters=edge.__str__().replace(')', '\n)'),
        )

    return graph



def make_cytoscape_elements(graph: networkx.DiGraph):
    nodes = [{'data': {'id': node,
                       'label': node,
                       'type' : graph.nodes[node].get('type',{}),
                       'color': graph.nodes[node]['color'],
                       'shape': graph.nodes[node]['shape'],
                       'parameters': {}}} #graph.nodes[node].get('parameters', {})}}
                       for node in graph.nodes()]
    edges = [{'data': {'source': u, 'target': v}} for u, v in graph.edges()]
    return nodes + edges


def shownetwork(graph: networkx.DiGraph):
    app = Dash(__name__, suppress_callback_exceptions=True)

    elements = make_cytoscape_elements(graph)
    cyto.load_extra_layouts()

    textcolor = 'white'

    app.layout = html.Div([
        dcc.Dropdown(
            id='layout-dropdown',
            options=[
                {'label': 'Klay (horizontal)', 'value': 'klay'},
                {'label': 'Dagre (vertical)', 'value': 'dagre'},
                {'label': 'Breadthfirst', 'value': 'breadthfirst'},
                {'label': 'Cose (force-directed)', 'value': 'cose'},
                {'label': 'Grid', 'value': 'grid'},
                {'label': 'Circle', 'value': 'circle'},
            ],
            value='klay',
            clearable=False,
            style={'width': '300px', 'margin': '10px'}
        ),

        cyto.Cytoscape(
            id='cytoscape',
            layout={'name': 'grid'},
            style={'width': '100%', 'height': '500px'},
            elements=elements,
            stylesheet=cytoscape_stylesheet[
                {
                    'selector': 'node',
                    'style': {
                        'content': 'data(label)',
                        'background-color': 'data(color)',
                        'font-size': 10,
                        'color': 'white',
                        'text-valign': 'center',
                        'text-halign': 'center',
                        'width': '90px',
                        'height': '70px',
                        'shape': 'data(shape)',
                        'text-outline-color': 'black',
                        'text-outline-width': 0.5,
                    }
                },
                {
                    'selector': '[shape = "custom-source"]',
                    'style': {
                        'shape': 'polygon',
                        'shape-polygon-points': '-0.5 0.5, 0.5 0.5, 1 -0.5, -1 -0.5',
                    }
                },
                {
                    'selector': '[shape = "custom-sink"]',
                    'style': {
                        'shape': 'polygon',
                        'shape-polygon-points': '-0.5 -0.5, 0.5 -0.5, 1 0.5, -1 0.5',
                    }
                },
                {
                    'selector': 'edge',
                    'style': {
                        'curve-style': 'straight',
                        'width': 2,
                        'line-color': 'gray',
                        'target-arrow-color': 'gray',
                        'target-arrow-shape': 'triangle',
                        'arrow-scale': 2,
                    }
                }
            ]
        ),

        html.Div(id='node-data', style={
            'white-space': 'pre-line', 'color': 'white',
            'background-color': '#2D3033', 'padding': '10px'
        }),

        html.Button("Export as Image", id="btn-image", n_clicks=0)
    ])

    # Show node data on click
    @app.callback(
        Output('node-data', 'children'),
        Input('cytoscape', 'tapNodeData')
    )
    def display_node_data(data):
        if data:
            parameters = data.get('parameters', {})
            components = [html.H4(f"Node {data['id']} Parameters:", style={'color': textcolor})]
            for k, v in parameters.items():
                components.append(html.P(f"{k}: {v}", style={'color': textcolor}))
            return html.Div(components)
        return html.P("Click on a node to see its parameters.", style={'color': textcolor})

    # Allow changing layout dynamically
    @app.callback(
        Output('cytoscape', 'layout'),
        Input('layout-dropdown', 'value')
    )
    def update_layout(selected_layout):
        return {'name': selected_layout}

    # Export graph as image
    app.clientside_callback(
        """
        function(n_clicks) {
            if (n_clicks > 0) {
                var cy = window.cy;
                if (cy) {
                    var png64 = cy.png({scale: 3, full: true});
                    var a = document.createElement('a');
                    a.href = png64;
                    a.download = 'Oemof_model.png';
                    a.click();
                }
            }
            return 'Export as Image';
        }
        """,
        Output('btn-image', 'children'),
        Input('btn-image', 'n_clicks')
    )

    # Find a free port
    def is_port_in_use(port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0

    def find_free_port(start_port=8050, end_port=8100):
        for port in range(start_port, end_port):
            if not is_port_in_use(port):
                return port
        raise Exception("No free port found")

    # Run app
    port = find_free_port(8050, 8100)
    print(f'Starting Network on port {port}')# logger.info(f'Starting Network on port {port}')
    app.run(debug=True, port=port)

    return app