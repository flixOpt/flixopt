from dash import Dash, html, dcc, Input, Output, State, callback_context
import dash_cytoscape as cyto
import networkx
import socket
import logging
import json

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


def create_style_controls():
    """Create UI controls for modifying the stylesheet"""
    return html.Div([
        html.H3("Style Controls", style={'color': 'white', 'margin-bottom': '10px'}),

        # Color scheme selector
        html.Div([
            html.Label("Color Scheme:", style={'color': 'white', 'margin-right': '10px'}),
            dcc.Dropdown(
                id='color-scheme-dropdown',
                options=[{'label': k, 'value': k} for k in color_presets.keys()],
                value='Default',
                style={'width': '200px', 'display': 'inline-block'}
            ),
            # Arrow styling
            html.Div([
                html.Label("Arrow Style:", style={'color': 'white', 'margin-right': '10px'}),
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
                    style={'width': '200px', 'display': 'inline-block'}
                )
            ], style={'margin-bottom': '15px'}),

            # Individual color pickers for each node type
            html.Div([
                html.H4("Custom Colors:", style={'color': 'white', 'margin-bottom': '10px'}),
                html.Div([
                    html.Label("Bus Color:", style={'color': 'white', 'margin-right': '10px'}),
                    dcc.Input(id='bus-color-input', type='text', value='#7F8C8D',
                              style={'width': '100px', 'margin-right': '20px'}),
                    html.Label("Source Color:", style={'color': 'white', 'margin-right': '10px'}),
                    dcc.Input(id='source-color-input', type='text', value='#F1C40F',
                              style={'width': '100px', 'margin-right': '20px'}),
                ], style={'margin-bottom': '10px'}),
                html.Div([
                    html.Label("Sink Color:", style={'color': 'white', 'margin-right': '10px'}),
                    dcc.Input(id='sink-color-input', type='text', value='#F1C40F',
                              style={'width': '100px', 'margin-right': '20px'}),
                    html.Label("Storage Color:", style={'color': 'white', 'margin-right': '10px'}),
                    dcc.Input(id='storage-color-input', type='text', value='#2980B9',
                              style={'width': '100px', 'margin-right': '20px'}),
                ], style={'margin-bottom': '10px'}),
                html.Div([
                    html.Label("Converter Color:", style={'color': 'white', 'margin-right': '10px'}),
                    dcc.Input(id='converter-color-input', type='text', value='#D35400',
                              style={'width': '100px', 'margin-right': '20px'}),
                    html.Label("Edge Color:", style={'color': 'white', 'margin-right': '10px'}),
                    dcc.Input(id='edge-color-input', type='text', value='gray',
                              style={'width': '100px', 'margin-right': '20px'}),
                ], style={'margin-bottom': '15px'}),
            ]),

            # Text styling controls
            html.Div([
                html.H4("Text Styling:", style={'color': 'white', 'margin-bottom': '10px'}),
                html.Div([
                    html.Label("Text Color:", style={'color': 'white', 'margin-right': '10px'}),
                    dcc.Input(id='text-color-input', type='text', value='white',
                              style={'width': '100px', 'margin-right': '20px'}),
                    html.Label("Text Outline:", style={'color': 'white', 'margin-right': '10px'}),
                    dcc.Input(id='text-outline-input', type='text', value='black',
                              style={'width': '100px', 'margin-right': '20px'}),
                ], style={'margin-bottom': '10px'}),
                html.Div([
                    html.Label("Text Position:", style={'color': 'white', 'margin-right': '10px'}),
                    dcc.Dropdown(
                        id='text-valign-dropdown',
                        options=[
                            {'label': 'Top', 'value': 'top'},
                            {'label': 'Center', 'value': 'center'},
                            {'label': 'Bottom', 'value': 'bottom'}
                        ],
                        value='center',
                        style={'width': '120px', 'margin-right': '20px', 'display': 'inline-block'}
                    ),
                    html.Label("Text Alignment:", style={'color': 'white', 'margin-right': '10px'}),
                    dcc.Dropdown(
                        id='text-halign-dropdown',
                        options=[
                            {'label': 'Left', 'value': 'left'},
                            {'label': 'Center', 'value': 'center'},
                            {'label': 'Right', 'value': 'right'}
                        ],
                        value='center',
                        style={'width': '120px', 'display': 'inline-block'}
                    ),
                ], style={'margin-bottom': '15px'}),
            ]),
            html.Div([
                html.Label("Node Size:", style={'color': 'white', 'margin-right': '10px'}),
                dcc.Slider(
                    id='node-size-slider',
                    min=50,
                    max=150,
                    step=10,
                    value=90,
                    marks={i: str(i) for i in range(50, 151, 25)},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], style={'margin-bottom': '15px'}),

            # Font size controls
            html.Div([
                html.Label("Font Size:", style={'color': 'white', 'margin-right': '10px'}),
                dcc.Slider(
                    id='font-size-slider',
                    min=8,
                    max=20,
                    step=1,
                    value=10,
                    marks={i: str(i) for i in range(8, 21, 2)},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], style={'margin-bottom': '15px'}),

            # Edge styling controls
            html.Div([
                html.H4("Edge Styling:", style={'color': 'white', 'margin-bottom': '10px'}),
                html.Div([
                    html.Label("Edge Width:", style={'color': 'white', 'margin-right': '10px'}),
                    dcc.Slider(
                        id='edge-width-slider',
                        min=1,
                        max=10,
                        step=1,
                        value=2,
                        marks={i: str(i) for i in range(1, 11)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], style={'margin-bottom': '15px'}),

                # Edge curve style
                html.Div([
                    html.Label("Edge Curve Style:", style={'color': 'white', 'margin-right': '10px'}),
                    dcc.Dropdown(
                        id='edge-curve-dropdown',
                        options=[
                            {'label': 'Straight', 'value': 'straight'},
                            {'label': 'Bezier', 'value': 'bezier'},
                            {'label': 'Unbundled Bezier', 'value': 'unbundled-bezier'},
                            {'label': 'Segments', 'value': 'segments'}
                        ],
                        value='straight',
                        style={'width': '200px', 'display': 'inline-block'}
                    )
                ], style={'margin-bottom': '15px'}),

                # Save/Load custom styles
                html.Div([
                    html.H4("Style Presets:", style={'color': 'white', 'margin-bottom': '10px'}),
                    html.Div([
                        dcc.Input(id='save-style-name', type='text', placeholder='Enter style name...',
                                  style={'width': '200px', 'margin-right': '10px'}),
                        html.Button("Save Style", id="save-style-btn", n_clicks=0,
                                    style={'margin-right': '10px', 'background-color': '#27AE60', 'color': 'white',
                                           'border': 'none', 'padding': '5px 10px'}),
                        dcc.Dropdown(
                            id='load-style-dropdown',
                            options=[],
                            placeholder="Load saved style...",
                            style={'width': '200px', 'display': 'inline-block'}
                        )
                    ], style={'margin-bottom': '15px'}),
                ]),
                html.Div([
                    html.Label("Custom Stylesheet (JSON):", style={'color': 'white', 'margin-bottom': '5px'}),
                    dcc.Textarea(
                        id='custom-stylesheet-textarea',
                        placeholder='Enter custom Cytoscape stylesheet as JSON...',
                        style={'width': '100%', 'height': '150px', 'background-color': '#34495E', 'color': 'white'},
                        value=json.dumps(default_cytoscape_stylesheet, indent=2)
                    )
                ], style={'margin-bottom': '15px'}),

                # Control buttons
                html.Div([
                    html.Button("Apply Custom Style", id="apply-custom-btn", n_clicks=0,
                                style={'margin-right': '10px', 'background-color': '#3498DB', 'color': 'white',
                                       'border': 'none', 'padding': '5px 10px'}),
                    html.Button("Reset to Default", id="reset-style-btn", n_clicks=0,
                                style={'background-color': '#E74C3C', 'color': 'white', 'border': 'none',
                                       'padding': '5px 10px'})
                ])
            ], style={'background-color': '#2D3033', 'padding': '15px', 'margin': '10px', 'border-radius': '5px'})
        ])])


def shownetwork(graph: networkx.DiGraph):
    app = Dash(__name__, suppress_callback_exceptions=True)

    elements = make_cytoscape_elements(graph)
    cyto.load_extra_layouts()

    textcolor = 'white'

    app.layout = html.Div([
        # Main controls row
        html.Div([
            html.Div([
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
            ], style={'width': '30%', 'display': 'inline-block'}),

            html.Div([
                html.Button("Toggle Style Panel", id="toggle-style-btn", n_clicks=0,
                            style={'margin': '10px', 'background-color': '#9B59B6', 'color': 'white', 'border': 'none',
                                   'padding': '10px 20px'})
            ], style={'width': '70%', 'display': 'inline-block', 'text-align': 'right'})
        ]),

        # Collapsible style panel
        html.Div(id='style-panel', children=[create_style_controls()], style={'display': 'none'}),

        # Main cytoscape component
        cyto.Cytoscape(
            id='cytoscape',
            layout={'name': 'grid'},
            style={'width': '100%', 'height': '500px'},
            elements=elements,
            stylesheet=default_cytoscape_stylesheet,
        ),

        # Node data display
        html.Div(id='node-data', style={
            'white-space': 'pre-line', 'color': 'white',
            'background-color': '#2D3033', 'padding': '10px'
        }),

        # Export button
        html.Button("Export as Image", id="btn-image", n_clicks=0,
                    style={'margin': '10px', 'background-color': '#27AE60', 'color': 'white', 'border': 'none',
                           'padding': '10px 20px'})
    ])

    # Toggle style panel visibility
    @app.callback(
        Output('style-panel', 'style'),
        Input('toggle-style-btn', 'n_clicks')
    )
    def toggle_style_panel(n_clicks):
        if n_clicks % 2 == 1:
            return {'display': 'block'}
        return {'display': 'none'}

    # Update stylesheet based on controls
    @app.callback(
        Output('cytoscape', 'stylesheet'),
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
         Input('apply-custom-btn', 'n_clicks'),
         Input('reset-style-btn', 'n_clicks'),
         Input('load-style-dropdown', 'value')],
        [State('custom-stylesheet-textarea', 'value')]
    )
    def update_stylesheet(color_scheme, bus_color, source_color, sink_color, storage_color,
                          converter_color, edge_color, text_color, text_outline,
                          text_valign, text_halign, node_size, font_size, edge_width,
                          edge_curve, arrow_style, apply_clicks, reset_clicks,
                          load_style, custom_style):
        ctx = callback_context

        if ctx.triggered:
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]

            # Reset to default
            if button_id == 'reset-style-btn':
                return default_cytoscape_stylesheet

            # Apply custom stylesheet
            if button_id == 'apply-custom-btn':
                try:
                    return json.loads(custom_style)
                except json.JSONDecodeError:
                    return default_cytoscape_stylesheet

        # Update stylesheet based on controls
        colors = color_presets[color_scheme]

        # Update elements with new colors (only for preset color schemes)
        if color_scheme != 'Custom':
            for element in elements:
                if 'color' in element['data']:
                    # Get the actual node type from the graph or determine it from the element
                    node_id = element['data']['id']
                    # Since we don't have direct access to the node type, we'll determine it from the shape
                    shape = element['data'].get('shape', 'rectangle')

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
                        element['data']['color'] = colors[node_type]

            # Create updated stylesheet using individual color inputs
            updated_stylesheet = [
                {
                    'selector': 'node',
                    'style': {
                        'content': 'data(label)',
                        'background-color': 'data(color)',
                        'font-size': font_size,
                        'color': text_color,
                        'text-valign': text_valign,
                        'text-halign': text_halign,
                        'width': f'{node_size}px',
                        'height': f'{node_size * 0.8}px',
                        'shape': 'data(shape)',
                        'text-outline-color': text_outline,
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
                        'curve-style': edge_curve,
                        'width': edge_width,
                        'line-color': edge_color,
                        'target-arrow-color': edge_color,
                        'target-arrow-shape': arrow_style,
                        'arrow-scale': 2,
                    }
                }
            ]

        # Update node colors based on individual inputs if not using preset
        elif color_scheme == 'Custom' or any([bus_color, source_color, sink_color, storage_color, converter_color]):
            custom_colors = {
                'Bus': bus_color or '#7F8C8D',
                'Source': source_color or '#F1C40F',
                'Sink': sink_color or '#F1C40F',
                'Storage': storage_color or '#2980B9',
                'Converter': converter_color or '#D35400',
                'Other': converter_color or '#27AE60'
            }

            for element in elements:
                if 'color' in element['data']:
                    # Determine node type from shape and id
                    node_id = element['data']['id']
                    shape = element['data'].get('shape', 'rectangle')

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

                    if node_type in custom_colors:
                        element['data']['color'] = custom_colors[node_type]

        return updated_stylesheet


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
    print(f'Starting Network on port {port}')
    app.run(debug=True, port=port)

    return app