from __future__ import annotations

import logging
import socket
import threading
from typing import TYPE_CHECKING, Any

try:
    import dash_cytoscape as cyto
    import dash_daq as daq
    import networkx as nx
    from dash import Dash, Input, Output, State, callback_context, dcc, html, no_update
    from werkzeug.serving import make_server

    DASH_CYTOSCAPE_AVAILABLE = True
    VISUALIZATION_ERROR = None
except ImportError as e:
    DASH_CYTOSCAPE_AVAILABLE = False
    VISUALIZATION_ERROR = str(e)

from .components import LinearConverter, Sink, Source, SourceAndSink, Storage
from .config import SUCCESS_LEVEL
from .elements import Bus

if TYPE_CHECKING:
    from .flow_system import FlowSystem

logger = logging.getLogger('flixopt')


# Configuration class for better organization
class VisualizationConfig:
    """Configuration constants for the visualization"""

    DEFAULT_COLORS = {
        'Bus': '#7F8C8D',
        'Source': '#F1C40F',
        'Sink': '#F1C40F',
        'Storage': '#2980B9',
        'Converter': '#D35400',
        'Other': '#27AE60',
    }

    COLOR_PRESETS = {
        'Default': DEFAULT_COLORS,
        'Vibrant': {
            'Bus': '#FF6B6B',
            'Source': '#4ECDC4',
            'Sink': '#45B7D1',
            'Storage': '#96CEB4',
            'Converter': '#FFEAA7',
            'Other': '#DDA0DD',
        },
        'Dark': {
            'Bus': '#2C3E50',
            'Source': '#34495E',
            'Sink': '#7F8C8D',
            'Storage': '#95A5A6',
            'Converter': '#BDC3C7',
            'Other': '#ECF0F1',
        },
        'Pastel': {
            'Bus': '#FFB3BA',
            'Source': '#BAFFC9',
            'Sink': '#BAE1FF',
            'Storage': '#FFFFBA',
            'Converter': '#FFDFBA',
            'Other': '#E0BBE4',
        },
    }

    DEFAULT_STYLESHEET = [
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
            },
        },
        {
            'selector': '[shape = "custom-source"]',
            'style': {
                'shape': 'polygon',
                'shape-polygon-points': '-0.5 0.5, 0.5 0.5, 1 -0.5, -1 -0.5',
            },
        },
        {
            'selector': '[shape = "custom-sink"]',
            'style': {
                'shape': 'polygon',
                'shape-polygon-points': '-0.5 -0.5, 0.5 -0.5, 1 0.5, -1 0.5',
            },
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
            },
        },
    ]


def flow_graph(flow_system: FlowSystem) -> nx.DiGraph:
    """Convert FlowSystem to NetworkX graph - simplified and more robust"""
    if not DASH_CYTOSCAPE_AVAILABLE:
        raise ImportError(
            'Network visualization requires optional dependencies. '
            'Install with: pip install flixopt[viz] or '
            'pip install dash dash-cytoscape networkx werkzeug. '
            f'Original error: {VISUALIZATION_ERROR}'
        )

    nodes = list(flow_system.components.values()) + list(flow_system.buses.values())
    edges = list(flow_system.flows.values())

    def get_element_type(element):
        """Determine element type for coloring"""
        if isinstance(element, Bus):
            return 'Bus'
        elif isinstance(element, Source):
            return 'Source'
        elif isinstance(element, (Sink, SourceAndSink)):
            return 'Sink'
        elif isinstance(element, Storage):
            return 'Storage'
        elif isinstance(element, LinearConverter):
            return 'Converter'
        else:
            return 'Other'

    def get_shape(element):
        """Determine node shape"""
        if isinstance(element, Bus):
            return 'ellipse'
        elif isinstance(element, Source):
            return 'custom-source'
        elif isinstance(element, (Sink, SourceAndSink)):
            return 'custom-sink'
        else:
            return 'rectangle'

    graph = nx.DiGraph()

    # Add nodes with attributes
    for node in nodes:
        graph.add_node(
            node.id,
            color=VisualizationConfig.DEFAULT_COLORS[get_element_type(node)],
            shape=get_shape(node),
            element_type=get_element_type(node),
            parameters=str(node),
        )

    # Add edges
    for edge in edges:
        try:
            graph.add_edge(
                u_of_edge=edge.bus if edge.is_input_in_component else edge.component,
                v_of_edge=edge.component if edge.is_input_in_component else edge.bus,
                label=edge.id,
                parameters=edge.__str__().replace(')', '\n)'),
            )
        except Exception as e:
            logger.error(f'Failed to add edge {edge}: {e}')

    return graph


def make_cytoscape_elements(graph: nx.DiGraph) -> list[dict[str, Any]]:
    """Convert NetworkX graph to Cytoscape elements"""
    elements = []

    # Add nodes
    for node_id in graph.nodes():
        node_data = graph.nodes[node_id]
        elements.append(
            {
                'data': {
                    'id': node_id,
                    'label': node_id,
                    'color': node_data.get('color', '#7F8C8D'),
                    'shape': node_data.get('shape', 'rectangle'),
                    'element_type': node_data.get('element_type', 'Other'),
                    'parameters': node_data.get('parameters', ''),
                }
            }
        )

    # Add edges
    for u, v in graph.edges():
        edge_data = graph.edges[u, v]
        elements.append(
            {
                'data': {
                    'source': u,
                    'target': v,
                    'id': f'{u}-{v}',
                    'label': edge_data.get('label', ''),
                    'parameters': edge_data.get('parameters', ''),
                }
            }
        )

    return elements


def create_color_picker_input(label: str, input_id: str, default_color: str):
    """Create a compact color picker with DAQ ColorPicker"""
    return html.Div(
        [
            html.Label(
                label, style={'color': 'white', 'font-size': '12px', 'margin-bottom': '5px', 'display': 'block'}
            ),
            daq.ColorPicker(
                id=input_id,
                label='',
                value={'hex': default_color},
                size=200,
                theme={'dark': True},
                style={'margin-bottom': '10px'},
            ),
        ]
    )


def create_style_section(title: str, children: list):
    """Create a collapsible section for organizing controls"""
    return html.Div(
        [
            html.H4(
                title,
                style={
                    'color': 'white',
                    'margin-bottom': '10px',
                    'border-bottom': '2px solid #3498DB',
                    'padding-bottom': '5px',
                },
            ),
            html.Div(children, style={'margin-bottom': '20px'}),
        ]
    )


def create_sidebar():
    """Create the main sidebar with improved organization"""
    return html.Div(
        [
            html.Div(
                [
                    html.H3(
                        'Style Controls',
                        style={
                            'color': 'white',
                            'margin-bottom': '20px',
                            'text-align': 'center',
                            'border-bottom': '3px solid #9B59B6',
                            'padding-bottom': '10px',
                        },
                    ),
                    # Layout Section
                    create_style_section(
                        'Layout',
                        [
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
                                style={'width': '100%'},
                            ),
                        ],
                    ),
                    # Color Scheme Section
                    create_style_section(
                        'Color Scheme',
                        [
                            dcc.Dropdown(
                                id='color-scheme-dropdown',
                                options=[{'label': k, 'value': k} for k in VisualizationConfig.COLOR_PRESETS.keys()],
                                value='Default',
                                style={'width': '100%', 'margin-bottom': '10px'},
                            ),
                        ],
                    ),
                    # Color Pickers Section
                    create_style_section(
                        'Custom Colors',
                        [
                            create_color_picker_input('Bus', 'bus-color-picker', '#7F8C8D'),
                            create_color_picker_input('Source', 'source-color-picker', '#F1C40F'),
                            create_color_picker_input('Sink', 'sink-color-picker', '#F1C40F'),
                            create_color_picker_input('Storage', 'storage-color-picker', '#2980B9'),
                            create_color_picker_input('Converter', 'converter-color-picker', '#D35400'),
                            create_color_picker_input('Edge', 'edge-color-picker', '#808080'),
                        ],
                    ),
                    # Node Settings
                    create_style_section(
                        'Node Settings',
                        [
                            html.Label('Size', style={'color': 'white', 'font-size': '12px'}),
                            dcc.Slider(
                                id='node-size-slider',
                                min=50,
                                max=150,
                                step=10,
                                value=90,
                                marks={
                                    i: {'label': str(i), 'style': {'color': 'white', 'font-size': '10px'}}
                                    for i in range(50, 151, 25)
                                },
                                tooltip={'placement': 'bottom', 'always_visible': True},
                            ),
                            html.Br(),
                            html.Label('Font Size', style={'color': 'white', 'font-size': '12px'}),
                            dcc.Slider(
                                id='font-size-slider',
                                min=8,
                                max=20,
                                step=1,
                                value=10,
                                marks={
                                    i: {'label': str(i), 'style': {'color': 'white', 'font-size': '10px'}}
                                    for i in range(8, 21, 2)
                                },
                                tooltip={'placement': 'bottom', 'always_visible': True},
                            ),
                        ],
                    ),
                    # Reset Button
                    html.Div(
                        [
                            html.Button(
                                'Reset to Defaults',
                                id='reset-btn',
                                n_clicks=0,
                                style={
                                    'width': '100%',
                                    'background-color': '#E74C3C',
                                    'color': 'white',
                                    'border': 'none',
                                    'padding': '10px',
                                    'border-radius': '5px',
                                    'cursor': 'pointer',
                                    'margin-top': '20px',
                                },
                            ),
                        ]
                    ),
                ],
                id='sidebar-content',
                style={
                    'width': '280px',
                    'height': '100vh',
                    'background-color': '#2C3E50',
                    'padding': '20px',
                    'position': 'fixed',
                    'left': '0',
                    'top': '0',
                    'overflow-y': 'auto',
                    'border-right': '3px solid #34495E',
                    'box-shadow': '2px 0 5px rgba(0,0,0,0.1)',
                    'z-index': '999',
                    'transform': 'translateX(-100%)',
                    'transition': 'transform 0.3s ease',
                },
            )
        ]
    )


def shownetwork(graph: nx.DiGraph):
    """Main function to create and run the network visualization"""
    if not DASH_CYTOSCAPE_AVAILABLE:
        raise ImportError(f'Required packages not available: {VISUALIZATION_ERROR}')

    app = Dash(__name__, suppress_callback_exceptions=True)

    # Load extra layouts
    cyto.load_extra_layouts()

    # Create initial elements
    elements = make_cytoscape_elements(graph)

    # App Layout
    app.layout = html.Div(
        [
            # Toggle button
            html.Button(
                '☰',
                id='toggle-sidebar',
                n_clicks=0,
                style={
                    'position': 'fixed',
                    'top': '20px',
                    'left': '20px',
                    'z-index': '1000',
                    'background-color': '#3498DB',
                    'color': 'white',
                    'border': 'none',
                    'padding': '10px 15px',
                    'border-radius': '5px',
                    'cursor': 'pointer',
                    'font-size': '18px',
                    'box-shadow': '0 2px 5px rgba(0,0,0,0.3)',
                },
            ),
            # Data storage
            dcc.Store(id='elements-store', data=elements),
            # Sidebar
            create_sidebar(),
            # Main content
            html.Div(
                [
                    # Header
                    html.Div(
                        [
                            html.H2(
                                'Network Visualization', style={'color': 'white', 'margin': '0', 'text-align': 'center'}
                            ),
                            html.Button(
                                'Export PNG',
                                id='export-btn',
                                n_clicks=0,
                                style={
                                    'position': 'absolute',
                                    'right': '20px',
                                    'top': '15px',
                                    'background-color': '#27AE60',
                                    'color': 'white',
                                    'border': 'none',
                                    'padding': '10px 20px',
                                    'border-radius': '5px',
                                    'cursor': 'pointer',
                                },
                            ),
                        ],
                        style={
                            'background-color': '#34495E',
                            'padding': '15px 20px',
                            'position': 'relative',
                            'border-bottom': '2px solid #3498DB',
                        },
                    ),
                    # Cytoscape graph
                    cyto.Cytoscape(
                        id='cytoscape',
                        layout={'name': 'klay'},
                        style={'width': '100%', 'height': '70vh'},
                        elements=elements,
                        stylesheet=VisualizationConfig.DEFAULT_STYLESHEET,
                    ),
                    # Info panel
                    html.Div(
                        [
                            html.H4(
                                'Element Information',
                                style={
                                    'color': 'white',
                                    'margin': '0 0 10px 0',
                                    'border-bottom': '2px solid #3498DB',
                                    'padding-bottom': '5px',
                                },
                            ),
                            html.Div(
                                id='info-panel',
                                children=[
                                    html.P(
                                        'Click on a node or edge to see details.',
                                        style={'color': '#95A5A6', 'font-style': 'italic'},
                                    )
                                ],
                            ),
                        ],
                        style={
                            'background-color': '#2C3E50',
                            'padding': '15px',
                            'height': '25vh',
                            'overflow-y': 'auto',
                            'border-top': '2px solid #34495E',
                        },
                    ),
                ],
                id='main-content',
                style={
                    'margin-left': '0',
                    'background-color': '#1A252F',
                    'min-height': '100vh',
                    'transition': 'margin-left 0.3s ease',
                },
            ),
        ]
    )

    # Callbacks
    @app.callback(
        [Output('sidebar-content', 'style'), Output('main-content', 'style')], [Input('toggle-sidebar', 'n_clicks')]
    )
    def toggle_sidebar(n_clicks):
        is_open = (n_clicks or 0) % 2 == 1
        sidebar_transform = 'translateX(0)' if is_open else 'translateX(-100%)'
        main_margin = '280px' if is_open else '0'

        sidebar_style = {
            'width': '280px',
            'height': '100vh',
            'background-color': '#2C3E50',
            'padding': '20px',
            'position': 'fixed',
            'left': '0',
            'top': '0',
            'overflow-y': 'auto',
            'border-right': '3px solid #34495E',
            'box-shadow': '2px 0 5px rgba(0,0,0,0.1)',
            'z-index': '999',
            'transform': sidebar_transform,
            'transition': 'transform 0.3s ease',
        }

        main_style = {
            'margin-left': main_margin,
            'background-color': '#1A252F',
            'min-height': '100vh',
            'transition': 'margin-left 0.3s ease',
        }

        return sidebar_style, main_style

    # Combined callback to handle both color scheme changes and reset
    @app.callback(
        [
            Output('bus-color-picker', 'value'),
            Output('source-color-picker', 'value'),
            Output('sink-color-picker', 'value'),
            Output('storage-color-picker', 'value'),
            Output('converter-color-picker', 'value'),
        ],
        [Input('color-scheme-dropdown', 'value'), Input('reset-btn', 'n_clicks')],
    )
    def update_color_pickers(color_scheme, reset_clicks):
        """Update color pickers when color scheme changes or reset is clicked"""
        ctx = callback_context

        # Determine which input triggered the callback
        if ctx.triggered:
            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
            if trigger_id == 'reset-btn' and reset_clicks and reset_clicks > 0:
                # Reset was clicked, use default colors
                colors = VisualizationConfig.DEFAULT_COLORS
            else:
                # Color scheme changed
                colors = VisualizationConfig.COLOR_PRESETS.get(color_scheme, VisualizationConfig.DEFAULT_COLORS)
        else:
            # Initial load
            colors = VisualizationConfig.COLOR_PRESETS.get(color_scheme, VisualizationConfig.DEFAULT_COLORS)

        return (
            {'hex': colors['Bus']},
            {'hex': colors['Source']},
            {'hex': colors['Sink']},
            {'hex': colors['Storage']},
            {'hex': colors['Converter']},
        )

    # Updated main visualization callback - simplified logic
    @app.callback(
        [Output('cytoscape', 'elements'), Output('cytoscape', 'stylesheet')],
        [
            Input('bus-color-picker', 'value'),
            Input('source-color-picker', 'value'),
            Input('sink-color-picker', 'value'),
            Input('storage-color-picker', 'value'),
            Input('converter-color-picker', 'value'),
            Input('edge-color-picker', 'value'),
            Input('node-size-slider', 'value'),
            Input('font-size-slider', 'value'),
        ],
        [State('elements-store', 'data')],
    )
    def update_visualization(
        bus_color,
        source_color,
        sink_color,
        storage_color,
        converter_color,
        edge_color,
        node_size,
        font_size,
        stored_elements,
    ):
        """Update visualization based on current color picker values"""
        if not stored_elements:
            return no_update, no_update

        # Use colors from pickers (which are now synced with scheme selection)
        default_colors = VisualizationConfig.DEFAULT_COLORS
        colors = {
            'Bus': bus_color.get('hex') if bus_color else default_colors['Bus'],
            'Source': source_color.get('hex') if source_color else default_colors['Source'],
            'Sink': sink_color.get('hex') if sink_color else default_colors['Sink'],
            'Storage': storage_color.get('hex') if storage_color else default_colors['Storage'],
            'Converter': converter_color.get('hex') if converter_color else default_colors['Converter'],
            'Other': default_colors['Other'],
        }

        # Update element colors
        updated_elements = []
        for element in stored_elements:
            if 'data' in element and 'element_type' in element['data']:
                element_copy = element.copy()
                element_copy['data'] = element['data'].copy()
                element_type = element_copy['data']['element_type']
                if element_type in colors:
                    element_copy['data']['color'] = colors[element_type]
                updated_elements.append(element_copy)
            else:
                updated_elements.append(element)

        # Create stylesheet
        edge_color_hex = edge_color.get('hex') if edge_color else 'gray'
        stylesheet = [
            {
                'selector': 'node',
                'style': {
                    'content': 'data(label)',
                    'background-color': 'data(color)',
                    'font-size': font_size or 10,
                    'color': 'white',
                    'text-valign': 'center',
                    'text-halign': 'center',
                    'width': f'{node_size or 90}px',
                    'height': f'{int((node_size or 90) * 0.8)}px',
                    'shape': 'data(shape)',
                    'text-outline-color': 'black',
                    'text-outline-width': 0.5,
                },
            },
            {
                'selector': '[shape = "custom-source"]',
                'style': {
                    'shape': 'polygon',
                    'shape-polygon-points': '-0.5 0.5, 0.5 0.5, 1 -0.5, -1 -0.5',
                },
            },
            {
                'selector': '[shape = "custom-sink"]',
                'style': {
                    'shape': 'polygon',
                    'shape-polygon-points': '-0.5 -0.5, 0.5 -0.5, 1 0.5, -1 0.5',
                },
            },
            {
                'selector': 'edge',
                'style': {
                    'curve-style': 'straight',
                    'width': 2,
                    'line-color': edge_color_hex,
                    'target-arrow-color': edge_color_hex,
                    'target-arrow-shape': 'triangle',
                    'arrow-scale': 2,
                },
            },
        ]

        return updated_elements, stylesheet

    @app.callback(
        Output('info-panel', 'children'), [Input('cytoscape', 'tapNodeData'), Input('cytoscape', 'tapEdgeData')]
    )
    def display_element_info(node_data, edge_data):
        ctx = callback_context
        if not ctx.triggered:
            return [
                html.P('Click on a node or edge to see details.', style={'color': '#95A5A6', 'font-style': 'italic'})
            ]

        # Determine what was clicked
        if ctx.triggered[0]['prop_id'] == 'cytoscape.tapNodeData' and node_data:
            return [
                html.H5(
                    f'Node: {node_data.get("label", "Unknown")}', style={'color': 'white', 'margin-bottom': '10px'}
                ),
                html.P(f'Type: {node_data.get("element_type", "Unknown")}', style={'color': '#BDC3C7'}),
                html.Pre(
                    node_data.get('parameters', 'No parameters'),
                    style={'color': '#BDC3C7', 'font-size': '11px', 'white-space': 'pre-wrap'},
                ),
            ]
        elif ctx.triggered[0]['prop_id'] == 'cytoscape.tapEdgeData' and edge_data:
            return [
                html.H5(
                    f'Edge: {edge_data.get("label", "Unknown")}', style={'color': 'white', 'margin-bottom': '10px'}
                ),
                html.P(f'{edge_data.get("source", "")} → {edge_data.get("target", "")}', style={'color': '#E67E22'}),
                html.Pre(
                    edge_data.get('parameters', 'No parameters'),
                    style={'color': '#BDC3C7', 'font-size': '11px', 'white-space': 'pre-wrap'},
                ),
            ]

        return [html.P('Click on a node or edge to see details.', style={'color': '#95A5A6', 'font-style': 'italic'})]

    @app.callback(Output('cytoscape', 'layout'), Input('layout-dropdown', 'value'))
    def update_layout(selected_layout):
        return {'name': selected_layout}

    # Reset callback for non-color-picker controls
    @app.callback(
        [
            Output('color-scheme-dropdown', 'value'),
            Output('edge-color-picker', 'value'),
            Output('node-size-slider', 'value'),
            Output('font-size-slider', 'value'),
            Output('layout-dropdown', 'value'),
        ],
        [Input('reset-btn', 'n_clicks')],
    )
    def reset_controls(n_clicks):
        """Reset all controls to defaults (color pickers handled separately)"""
        if n_clicks and n_clicks > 0:
            return (
                'Default',  # color scheme (will trigger color picker updates)
                {'hex': '#808080'},  # edge color
                90,  # node size
                10,  # font size
                'klay',  # layout
            )
        return no_update

    # Export functionality
    app.clientside_callback(
        """
        function(n_clicks) {
            if (n_clicks > 0 && window.cy) {
                var png64 = window.cy.png({scale: 3, full: true});
                var a = document.createElement('a');
                a.href = png64;
                a.download = 'network_visualization.png';
                a.click();
            }
            return 'Export PNG';
        }
        """,
        Output('export-btn', 'children'),
        Input('export-btn', 'n_clicks'),
    )

    # Start server
    def find_free_port(start_port=8050, end_port=8100):
        for port in range(start_port, end_port):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(('localhost', port)) != 0:
                    return port
        raise Exception('No free port found')

    port = find_free_port()
    server = make_server('127.0.0.1', port, app.server)

    # Start server in background thread
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    logger.log(SUCCESS_LEVEL, f'Network visualization started on http://127.0.0.1:{port}/')

    # Store server reference for cleanup
    app.server_instance = server
    app.port = port

    return app
