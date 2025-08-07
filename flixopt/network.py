from dash import Dash, html, dcc, Input, Output, State, callback_context
import dash_cytoscape as cyto
import networkx
import socket
import logging
import json
import threading
from werkzeug.serving import make_server

from .flow_system import FlowSystem
from .elements import Bus, Flow, Component
from .components import Sink, Source, SourceAndSink, Storage, LinearConverter

logger = logging.getLogger('flixopt')

# Default stylesheet (can be reset to this)
default_cytoscape_stylesheet = [
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

# Color presets for different node types
color_presets = {
    'Default': {'Bus': '#7F8C8D', 'Source': '#F1C40F', 'Sink': '#F1C40F', 'Storage': '#2980B9', 'Converter': '#D35400',
                'Other': '#27AE60'},
    'Vibrant': {'Bus': '#FF6B6B', 'Source': '#4ECDC4', 'Sink': '#45B7D1', 'Storage': '#96CEB4', 'Converter': '#FFEAA7',
                'Other': '#DDA0DD'},
    'Dark': {'Bus': '#2C3E50', 'Source': '#34495E', 'Sink': '#7F8C8D', 'Storage': '#95A5A6', 'Converter': '#BDC3C7',
             'Other': '#ECF0F1'},
    'Pastel': {'Bus': '#FFB3BA', 'Source': '#BAFFC9', 'Sink': '#BAE1FF', 'Storage': '#FFFFBA', 'Converter': '#FFDFBA',
               'Other': '#E0BBE4'}
}


def flow_graph(flow_system: FlowSystem) -> networkx.DiGraph:
    nodes = list(flow_system.components.values()) + list(flow_system.buses.values())
    edges = list(flow_system.flows.values())

    def get_color(element, color_scheme='Default'):
        colors = color_presets[color_scheme]
        if isinstance(element, Flow):
            raise TypeError('Flow graph shape not yet implemented')
        if isinstance(element, Bus):
            return colors['Bus']
        if isinstance(element, (Sink, Source, SourceAndSink)):
            return colors['Source'] if isinstance(element, Source) else colors['Sink']
        if isinstance(element, Storage):
            return colors['Storage']
        if isinstance(element, LinearConverter):
            return colors['Converter']
        return colors['Other']

    def get_shape(element):
        if isinstance(element, Bus):
            return 'ellipse'
        if isinstance(element, (Source)):
            return 'custom-source'
        if isinstance(element, (Sink, SourceAndSink)):
            return 'custom-sink'
        return 'rectangle'

    graph = networkx.DiGraph()

    for node in nodes:
        graph.add_node(
            node.label_full,
            color=get_color(node),
            shape=get_shape(node),
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
                       'color': graph.nodes[node]['color'],
                       'shape': graph.nodes[node]['shape'],
                       'parameters': graph.nodes[node].get('parameters', {})}}
             for node in graph.nodes()]
    edges = [{'data': {'source': u, 'target': v}} for u, v in graph.edges()]
    return nodes + edges


def create_style_section(title, children):
    """Create a collapsible section for organizing controls"""
    return html.Div([
        html.H4(title, style={
            'color': 'white',
            'margin-bottom': '10px',
            'border-bottom': '2px solid #3498DB',
            'padding-bottom': '5px'
        }),
        html.Div(children, style={'margin-bottom': '20px'})
    ])


def create_collapsible_sidebar():
    """Create a collapsible sidebar with toggle functionality"""
    return html.Div([
        # Sidebar content
        html.Div([
            html.H3("Style Controls", style={
                'color': 'white',
                'margin-bottom': '20px',
                'text-align': 'center',
                'border-bottom': '3px solid #9B59B6',
                'padding-bottom': '10px'
            }),

            # Layout Controls
            create_style_section("Layout", [
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
                    style={'width': '100%'}
                )
            ]),

            # Color Scheme Section
            create_style_section("Color Scheme", [
                dcc.Dropdown(
                    id='color-scheme-dropdown',
                    options=[{'label': k, 'value': k} for k in color_presets.keys()],
                    value='Default',
                    style={'width': '100%', 'margin-bottom': '10px'}
                )
            ]),

            # Custom Colors Section
            create_style_section("Custom Colors", [
                html.Div([
                    html.Label("Bus", style={'color': 'white', 'font-size': '12px'}),
                    dcc.Input(id='bus-color-input', type='text', value='#7F8C8D',
                              style={'width': '100%', 'margin-bottom': '8px'})
                ]),
                html.Div([
                    html.Label("Source", style={'color': 'white', 'font-size': '12px'}),
                    dcc.Input(id='source-color-input', type='text', value='#F1C40F',
                              style={'width': '100%', 'margin-bottom': '8px'})
                ]),
                html.Div([
                    html.Label("Sink", style={'color': 'white', 'font-size': '12px'}),
                    dcc.Input(id='sink-color-input', type='text', value='#F1C40F',
                              style={'width': '100%', 'margin-bottom': '8px'})
                ]),
                html.Div([
                    html.Label("Storage", style={'color': 'white', 'font-size': '12px'}),
                    dcc.Input(id='storage-color-input', type='text', value='#2980B9',
                              style={'width': '100%', 'margin-bottom': '8px'})
                ]),
                html.Div([
                    html.Label("Converter", style={'color': 'white', 'font-size': '12px'}),
                    dcc.Input(id='converter-color-input', type='text', value='#D35400',
                              style={'width': '100%', 'margin-bottom': '8px'})
                ]),
                html.Div([
                    html.Label("Edge", style={'color': 'white', 'font-size': '12px'}),
                    dcc.Input(id='edge-color-input', type='text', value='gray',
                              style={'width': '100%', 'margin-bottom': '8px'})
                ])
            ]),

            # Node Styling Section
            create_style_section("Node Styling", [
                html.Div([
                    html.Label("Node Size", style={'color': 'white', 'font-size': '12px'}),
                    dcc.Slider(
                        id='node-size-slider',
                        min=50, max=150, step=10, value=90,
                        marks={i: {'label': str(i), 'style': {'color': 'white', 'font-size': '10px'}}
                               for i in range(50, 151, 25)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], style={'margin-bottom': '15px'}),
                html.Div([
                    html.Label("Font Size", style={'color': 'white', 'font-size': '12px'}),
                    dcc.Slider(
                        id='font-size-slider',
                        min=8, max=20, step=1, value=10,
                        marks={i: {'label': str(i), 'style': {'color': 'white', 'font-size': '10px'}}
                               for i in range(8, 21, 2)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], style={'margin-bottom': '15px'})
            ]),

            # Text Styling Section
            create_style_section("Text Styling", [
                html.Div([
                    html.Label("Text Color", style={'color': 'white', 'font-size': '12px'}),
                    dcc.Input(id='text-color-input', type='text', value='white',
                              style={'width': '100%', 'margin-bottom': '8px'})
                ]),
                html.Div([
                    html.Label("Text Outline", style={'color': 'white', 'font-size': '12px'}),
                    dcc.Input(id='text-outline-input', type='text', value='black',
                              style={'width': '100%', 'margin-bottom': '8px'})
                ]),
                html.Div([
                    html.Label("Text Position", style={'color': 'white', 'font-size': '12px'}),
                    dcc.Dropdown(
                        id='text-valign-dropdown',
                        options=[
                            {'label': 'Top', 'value': 'top'},
                            {'label': 'Center', 'value': 'center'},
                            {'label': 'Bottom', 'value': 'bottom'}
                        ],
                        value='center',
                        style={'width': '100%', 'margin-bottom': '8px'}
                    )
                ]),
                html.Div([
                    html.Label("Text Alignment", style={'color': 'white', 'font-size': '12px'}),
                    dcc.Dropdown(
                        id='text-halign-dropdown',
                        options=[
                            {'label': 'Left', 'value': 'left'},
                            {'label': 'Center', 'value': 'center'},
                            {'label': 'Right', 'value': 'right'}
                        ],
                        value='center',
                        style={'width': '100%', 'margin-bottom': '8px'}
                    )
                ])
            ]),

            # Edge Styling Section
            create_style_section("Edge Styling", [
                html.Div([
                    html.Label("Edge Width", style={'color': 'white', 'font-size': '12px'}),
                    dcc.Slider(
                        id='edge-width-slider',
                        min=1, max=10, step=1, value=2,
                        marks={i: {'label': str(i), 'style': {'color': 'white', 'font-size': '10px'}}
                               for i in range(1, 11)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], style={'margin-bottom': '15px'}),
                html.Div([
                    html.Label("Edge Curve", style={'color': 'white', 'font-size': '12px'}),
                    dcc.Dropdown(
                        id='edge-curve-dropdown',
                        options=[
                            {'label': 'Straight', 'value': 'straight'},
                            {'label': 'Bezier', 'value': 'bezier'},
                            {'label': 'Unbundled Bezier', 'value': 'unbundled-bezier'},
                            {'label': 'Segments', 'value': 'segments'}
                        ],
                        value='straight',
                        style={'width': '100%', 'margin-bottom': '8px'}
                    )
                ]),
                html.Div([
                    html.Label("Arrow Style", style={'color': 'white', 'font-size': '12px'}),
                    dcc.Dropdown(
                        id='arrow-style-dropdown',
                        options=[
                            {'label': 'Triangle', 'value': 'triangle'},
                            {'label': 'Triangle (Tee)', 'value': 'triangle-tee'},
                            {'label': 'Circle', 'value': 'circle'},
                            {'label': 'Square', 'value': 'square'},
                            {'label': 'Diamond', 'value': 'diamond'},
                            {'label': 'None', 'value': 'none'}
                        ],
                        value='triangle',
                        style={'width': '100%', 'margin-bottom': '8px'}
                    )
                ])
            ]),

            # Advanced Section
            create_style_section("Advanced", [
                html.Div([
                    html.Label("Custom Stylesheet (JSON)", style={'color': 'white', 'font-size': '12px'}),
                    dcc.Textarea(
                        id='custom-stylesheet-textarea',
                        placeholder='Enter custom Cytoscape stylesheet as JSON...',
                        style={'width': '100%', 'height': '120px', 'background-color': '#34495E',
                               'color': 'white', 'font-size': '11px', 'margin-bottom': '10px'},
                        value=json.dumps(default_cytoscape_stylesheet, indent=2)
                    )
                ]),
                html.Div([
                    html.Button("Apply Custom", id="apply-custom-btn", n_clicks=0,
                                style={'width': '48%', 'margin-right': '4%', 'background-color': '#3498DB',
                                       'color': 'white', 'border': 'none', 'padding': '8px', 'border-radius': '3px'}),
                    html.Button("Reset Default", id="reset-style-btn", n_clicks=0,
                                style={'width': '48%', 'background-color': '#E74C3C',
                                       'color': 'white', 'border': 'none', 'padding': '8px', 'border-radius': '3px'})
                ])
            ])
        ], id='sidebar-content', style={
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
            'transform': 'translateX(-100%)',  # Initially hidden
            'transition': 'transform 0.3s ease',
            'z-index': '999'
        })
    ])


def shownetwork(graph: networkx.DiGraph):
    app = Dash(__name__, suppress_callback_exceptions=True)

    elements = make_cytoscape_elements(graph)
    cyto.load_extra_layouts()

    textcolor = 'white'

    app.layout = html.Div([
        # Toggle button
        html.Button("☰", id="toggle-sidebar", n_clicks=0,
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
                        'box-shadow': '0 2px 5px rgba(0,0,0,0.3)'
                    }),

        # Hidden div to store elements data
        html.Div(id='elements-store', style={'display': 'none'}),

        # Sidebar
        create_collapsible_sidebar(),

        # Main content area
        html.Div([
            # Top toolbar
            html.Div([
                html.H2("Network Visualization", style={
                    'color': 'white',
                    'margin': '0',
                    'text-align': 'center'
                }),
                html.Button("Export as Image", id="btn-image", n_clicks=0,
                            style={'position': 'absolute', 'right': '20px', 'top': '15px',
                                   'background-color': '#27AE60', 'color': 'white', 'border': 'none',
                                   'padding': '10px 20px', 'border-radius': '5px', 'cursor': 'pointer'})
            ], style={
                'background-color': '#34495E',
                'padding': '15px 20px',
                'margin-bottom': '0',
                'position': 'relative',
                'border-bottom': '2px solid #3498DB'
            }),

            # Main cytoscape component
            cyto.Cytoscape(
                id='cytoscape',
                layout={'name': 'klay'},
                style={'width': '100%', 'height': '70vh'},
                elements=elements,
                stylesheet=default_cytoscape_stylesheet,
            ),

            # Bottom panel for node information
            html.Div([
                html.H4("Node Information", style={
                    'color': 'white',
                    'margin': '0 0 10px 0',
                    'border-bottom': '2px solid #3498DB',
                    'padding-bottom': '5px'
                }),
                html.Div(id='node-data', children=[
                    html.P("Click on a node to see its parameters.", style={'color': 'white', 'margin': '0'})
                ])
            ], style={
                'background-color': '#2C3E50',
                'padding': '15px',
                'height': '25vh',
                'overflow-y': 'auto',
                'border-top': '2px solid #34495E'
            })
        ], id='main-content', style={
            'margin-left': '0',  # Initially no margin
            'background-color': '#1A252F',
            'min-height': '100vh',
            'transition': 'margin-left 0.3s ease'
        })
    ])

    # Toggle sidebar visibility
    @app.callback(
        [Output('sidebar-content', 'style'),
         Output('main-content', 'style')],
        [Input('toggle-sidebar', 'n_clicks')]
    )
    def toggle_sidebar(n_clicks):
        if n_clicks % 2 == 1:  # Sidebar is open
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
                'transform': 'translateX(0)',
                'transition': 'transform 0.3s ease',
                'z-index': '999'
            }
            main_style = {
                'margin-left': '280px',
                'background-color': '#1A252F',
                'min-height': '100vh',
                'transition': 'margin-left 0.3s ease'
            }
        else:  # Sidebar is closed
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
                'transform': 'translateX(-100%)',
                'transition': 'transform 0.3s ease',
                'z-index': '999'
            }
            main_style = {
                'margin-left': '0',
                'background-color': '#1A252F',
                'min-height': '100vh',
                'transition': 'margin-left 0.3s ease'
            }

        return sidebar_style, main_style

    # Reset all controls to defaults
    @app.callback(
        [Output('color-scheme-dropdown', 'value'),
         Output('bus-color-input', 'value'),
         Output('source-color-input', 'value'),
         Output('sink-color-input', 'value'),
         Output('storage-color-input', 'value'),
         Output('converter-color-input', 'value'),
         Output('edge-color-input', 'value'),
         Output('text-color-input', 'value'),
         Output('text-outline-input', 'value'),
         Output('text-valign-dropdown', 'value'),
         Output('text-halign-dropdown', 'value'),
         Output('node-size-slider', 'value'),
         Output('font-size-slider', 'value'),
         Output('edge-width-slider', 'value'),
         Output('edge-curve-dropdown', 'value'),
         Output('arrow-style-dropdown', 'value'),
         Output('layout-dropdown', 'value'),
         Output('custom-stylesheet-textarea', 'value')],
        [Input('reset-style-btn', 'n_clicks')]
    )
    def reset_all_controls(reset_clicks):
        if reset_clicks and reset_clicks > 0:
            return (
                'Default',  # color-scheme-dropdown
                '#7F8C8D',  # bus-color-input
                '#F1C40F',  # source-color-input
                '#F1C40F',  # sink-color-input
                '#2980B9',  # storage-color-input
                '#D35400',  # converter-color-input
                'gray',  # edge-color-input
                'white',  # text-color-input
                'black',  # text-outline-input
                'center',  # text-valign-dropdown
                'center',  # text-halign-dropdown
                90,  # node-size-slider
                10,  # font-size-slider
                2,  # edge-width-slider
                'straight',  # edge-curve-dropdown
                'triangle',  # arrow-style-dropdown
                'klay',  # layout-dropdown
                json.dumps(default_cytoscape_stylesheet, indent=2)  # custom-stylesheet-textarea
            )
        # Return current values if no reset
        return (
            'Default', '#7F8C8D', '#F1C40F', '#F1C40F', '#2980B9', '#D35400', 'gray',
            'white', 'black', 'center', 'center', 90, 10, 2, 'straight', 'triangle', 'klay',
            json.dumps(default_cytoscape_stylesheet, indent=2)
        )

    # Update elements and stylesheet based on controls
    @app.callback(
        [Output('cytoscape', 'elements'),
         Output('cytoscape', 'stylesheet')],
        [Input('color-scheme-dropdown', 'value'),
         Input('bus-color-input', 'value'),
         Input('source-color-input', 'value'),
         Input('sink-color-input', 'value'),
         Input('storage-color-input', 'value'),
         Input('converter-color-input', 'value'),
         Input('edge-color-input', 'value'),
         Input('text-color-input', 'value'),
         Input('text-outline-input', 'value'),
         Input('text-valign-dropdown', 'value'),
         Input('text-halign-dropdown', 'value'),
         Input('node-size-slider', 'value'),
         Input('font-size-slider', 'value'),
         Input('edge-width-slider', 'value'),
         Input('edge-curve-dropdown', 'value'),
         Input('arrow-style-dropdown', 'value'),
         Input('apply-custom-btn', 'n_clicks')],
        [State('custom-stylesheet-textarea', 'value')]
    )
    def update_elements_and_stylesheet(color_scheme, bus_color, source_color, sink_color, storage_color,
                                       converter_color, edge_color, text_color, text_outline,
                                       text_valign, text_halign, node_size, font_size, edge_width,
                                       edge_curve, arrow_style, apply_clicks, custom_style):
        ctx = callback_context

        if ctx.triggered:
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]

            # Apply custom stylesheet
            if button_id == 'apply-custom-btn':
                try:
                    return elements, json.loads(custom_style)
                except json.JSONDecodeError:
                    return elements, default_cytoscape_stylesheet

        # Determine which colors to use
        use_custom_colors = any([
            bus_color and bus_color != '#7F8C8D',
            source_color and source_color != '#F1C40F',
            sink_color and sink_color != '#F1C40F',
            storage_color and storage_color != '#2980B9',
            converter_color and converter_color != '#D35400',
            edge_color and edge_color != 'gray'
        ])

        if use_custom_colors:
            colors = {
                'Bus': bus_color or '#7F8C8D',
                'Source': source_color or '#F1C40F',
                'Sink': sink_color or '#F1C40F',
                'Storage': storage_color or '#2980B9',
                'Converter': converter_color or '#D35400',
                'Other': converter_color or '#27AE60'
            }
        else:
            colors = color_presets.get(color_scheme, color_presets['Default'])

        # Create updated elements with new colors
        updated_elements = []
        for element in elements:
            if 'data' in element:
                element_copy = element.copy()
                element_copy['data'] = element['data'].copy()

                # Update node colors
                if 'color' in element_copy['data']:
                    node_id = element_copy['data']['id']
                    shape = element_copy['data'].get('shape', 'rectangle')

                    if shape == 'ellipse':
                        node_type = 'Bus'
                    elif shape == 'custom-source':
                        node_type = 'Source'
                    elif shape == 'custom-sink':
                        node_type = 'Sink'
                    elif 'storage' in node_id.lower():
                        node_type = 'Storage'
                    elif 'converter' in node_id.lower():
                        node_type = 'Converter'
                    else:
                        node_type = 'Other'

                    if node_type in colors:
                        element_copy['data']['color'] = colors[node_type]

                updated_elements.append(element_copy)
            else:
                updated_elements.append(element)

        # Create updated stylesheet
        stylesheet = [
            {
                'selector': 'node',
                'style': {
                    'content': 'data(label)',
                    'background-color': 'data(color)',
                    'font-size': font_size or 10,
                    'color': text_color or 'white',
                    'text-valign': text_valign or 'center',
                    'text-halign': text_halign or 'center',
                    'width': f'{node_size or 90}px',
                    'height': f'{(node_size or 90) * 0.8}px',
                    'shape': 'data(shape)',
                    'text-outline-color': text_outline or 'black',
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
                    'curve-style': edge_curve or 'straight',
                    'width': edge_width or 2,
                    'line-color': edge_color or 'gray',
                    'target-arrow-color': edge_color or 'gray',
                    'target-arrow-shape': arrow_style or 'triangle',
                    'arrow-scale': 2,
                }
            }
        ]

        return updated_elements, stylesheet

    # Show node data on click
    @app.callback(
        Output('node-data', 'children'),
        Input('cytoscape', 'tapNodeData')
    )
    def display_node_data(data):
        if data:
            parameters = data.get('parameters', {})
            if isinstance(parameters, dict) and parameters:
                components = [html.H5(f"Node: {data['id']}", style={'color': 'white', 'margin-bottom': '10px'})]
                for k, v in parameters.items():
                    components.append(html.P(f"{k}: {v}", style={'color': '#BDC3C7', 'margin': '5px 0'}))
                return components
            else:
                return [html.P(f"Node: {data['id']}", style={'color': 'white'}),
                        html.P(str(parameters), style={'color': '#BDC3C7'})]
        return [html.P("Click on a node to see its parameters.", style={'color': '#95A5A6'})]

    # Update layout when dropdown changes
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
                    a.download = 'network_visualization.png';
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
        raise Exception('No free port found')

    port = find_free_port(8050, 8100)
    server = make_server('127.0.0.1', port, app.server)

    # Start server in background thread
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    print(f'Network visualization started on port {port}')
    print(f'Access it at: http://127.0.0.1:{port}/')
    print('The app is running in the background. You can continue using the console.')

    # Store the actual server instance for shutdown
    app.server_instance = server
    app.port = port

    return app
