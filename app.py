import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import math
import logging
import json
from io import StringIO
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)

def validate_numeric(value, default=0.0, allow_zero=True):
    """
    Validate and convert numeric values with enhanced validation.
    
    Args:
        value: Value to validate
        default: Default value if invalid
        allow_zero: Whether to allow zero values
    Returns:
        float: Validated numeric value
    """
    try:
        result = float(value)
        if pd.isna(result):
            return default
        if not allow_zero and result == 0:
            return default
        return result
    except (ValueError, TypeError):
        return default

def unmerge_cell_copy_top_value_to_df(uploaded_file, sheet_name):
    try:
        df = pd.read_excel(uploaded_file, sheet_name=sheet_name, skiprows=3)
        return df
    except Exception as e:
        st.error(f"Error reading Excel file: {str(e)}")
        return None

def clean_data(df):
    # Map Excel columns to expected column names
    column_mapping = {
        'X': 'X',
        'Y': 'Y',
        'Rim Level or Ground Level at the center of the manhole cover.': 'Z = Ground Level Elevation',
        'Manhole Bottom Level or Depth (chose one only)': 'Depth of Manhole to GL',
        'Type of Utility (Choose or write yourself)': 'Type of Utility',
        'Pipe Invert Level': 'Depth of the Utilities from GL',
        'Diameter of the Utilities (inch)': 'Diameter of the Utilities (inch)',
        'Exit Azimuth of Utility (0-360)': 'Exit Azimuth of Utility',
        'Material of the Utility': 'Material of the Utility',
        'MANHOLE NUMBER': 'exoTag',
        'PIPE NUMBER': 'pipeTag'
    }
    
    # Rename columns
    df = df.rename(columns=column_mapping)
    
    # Remove rows where both X and Y are 0 or missing
    df = df[~((df['X'].fillna(0) == 0) & (df['Y'].fillna(0) == 0))]
    
    # Filter out rows without pipe numbers (handling both numeric and string values)
    if 'pipeTag' in df.columns:
        # Convert pipeTag to string and handle NaN values
        df['pipeTag'] = df['pipeTag'].astype(str)
        df = df[df['pipeTag'].notna() & (df['pipeTag'] != 'nan') & (df['pipeTag'].str.strip() != '')]
    else:
        st.error("PIPE NUMBER column not found in the Excel file. Please check the data format.")
        return None
    
    # Convert numeric columns with validation
    numeric_columns = {
        'X': {'allow_zero': False},
        'Y': {'allow_zero': False},
        'Z = Ground Level Elevation': {'allow_zero': True},
        'Depth of Manhole to GL': {'allow_zero': True},
        'Depth of the Utilities from GL': {'allow_zero': True},
        'Diameter of the Utilities (inch)': {'allow_zero': False, 'default': 1.0},
        'Exit Azimuth of Utility': {'allow_zero': True}
    }
    
    for col, params in numeric_columns.items():
        if col in df.columns:
            df[col] = df[col].apply(lambda x: validate_numeric(
                x, 
                default=params.get('default', 0.0),
                allow_zero=params.get('allow_zero', True)
            ))
    
    # Ensure non-numeric columns are strings and remove empty rows
    string_columns = ['Type of Utility', 'Material of the Utility', 'exoTag']
    for col in string_columns:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown').astype(str)
            df = df[df[col].str.strip() != '']
    
    # Remove rows with invalid coordinates
    df = df[df['X'].notna() & df['Y'].notna() & (df['X'] != 0) & (df['Y'] != 0)]
    
    return df

def get_color_for_utility_type(utility_type):
    color_map = {
        'Sewer': '#8B4513',      # Brown
        'Storm': '#228B22',      # Green
        'Water': '#4169E1',      # Blue
        'Gas': '#FF0000',        # Red
        'Electric': '#FFD700',   # Yellow
        'Communication': '#FFA500',  # Orange
        'Unknown': '#808080',    # Gray
        'Other': '#800080'       # Purple
    }
    return color_map.get(utility_type, '#808080')

def calculate_pipe_endpoints(manholes, pipes, pipe_length):
    pipe_endpoints = []
    for pipe in pipes:
        start_point = [pipe['start_point'][0], pipe['start_point'][1], pipe['start_point'][2]]
        end_point = [
            start_point[0] + pipe['direction'][0] * pipe_length,
            start_point[1] + pipe['direction'][1] * pipe_length,
            start_point[2]
        ]
        pipe_endpoints.append({
            'start': start_point,
            'end': end_point,
            'type': pipe['type'],
            'diameter': pipe['diameter']
        })
    return pipe_endpoints

def create_3d_visualization(manholes, pipes, pipe_length, show_manholes=True, show_pipes=True, selected_utilities=None):
    """
    Create 3D visualization of the pipe network.
    
    Args:
        manholes: Dictionary of manhole data
        pipes: List of pipe data
        pipe_length: Length of pipes (in meters)
        show_manholes: Boolean to show/hide manholes
        show_pipes: Boolean to show/hide pipes
        selected_utilities: List of utility types to display
    """
    fig = go.Figure()

    # Add manholes
    if show_manholes:
        manhole_x = []
        manhole_y = []
        manhole_z = []
        manhole_text = []
        
        for manhole_id, data in manholes.items():
            manhole_x.append(data['x'])
            manhole_y.append(data['y'])
            manhole_z.append(data['z'])
            manhole_text.append(f"Manhole {manhole_id}<br>Depth: {data['depth']:.2f}m")

        fig.add_trace(go.Scatter3d(
            x=manhole_x,
            y=manhole_y,
            z=manhole_z,
            mode='markers',
            marker=dict(
                size=8,
                color='black',
                symbol='square'
            ),
            text=manhole_text,
            hoverinfo='text',
            name='Manholes'
        ))

    # Add pipes and connections
    if show_pipes:
        # Group pipes by type
        pipe_groups = defaultdict(list)
        
        for pipe in pipes:
            if selected_utilities is None or pipe['type'] in selected_utilities:
                if 'is_connection' in pipe:
                    # For connection pipes, use the pre-calculated endpoints
                    pipe_data = {
                        'start': pipe['start_point'],
                        'end': pipe['end_point'],
                        'type': pipe['type'],
                        'diameter': pipe['diameter'],
                        'is_connection': True,
                        'pipe_number': pipe['pipe_number']
                    }
                else:
                    # For regular pipes, calculate endpoints based on direction and length
                    start_point = pipe['start_point']
                    end_point = (
                        start_point[0] + pipe['direction'][0] * pipe_length,
                        start_point[1] + pipe['direction'][1] * pipe_length,
                        start_point[2]
                    )
                    pipe_data = {
                        'start': start_point,
                        'end': end_point,
                        'type': pipe['type'],
                        'diameter': pipe['diameter'],
                        'is_connection': False,
                        'pipe_number': pipe['pipe_number']
                    }
                pipe_groups[pipe['type']].append(pipe_data)

        # Add pipes by type
        for pipe_type, pipes_of_type in pipe_groups.items():
            for pipe in pipes_of_type:
                try:
                    min_width = 2
                    max_width = 10
                    if pd.isna(pipe['diameter']) or pipe['diameter'] <= 0:
                        line_width = min_width
                    else:
                        line_width = min(max_width, max(min_width, pipe['diameter'] * 50))
                except:
                    line_width = min_width

                line_style = 'dot' if pipe.get('is_connection', False) else 'solid'
                
                hover_text = (
                    f"{pipe_type}<br>"
                    f"{'Connection ' if pipe.get('is_connection', False) else ''}"
                    f"Pipe: {pipe['pipe_number']}<br>"
                    f"Diameter: {pipe['diameter']*1000:.1f}mm"
                )

                fig.add_trace(go.Scatter3d(
                    x=[pipe['start'][0], pipe['end'][0]],
                    y=[pipe['start'][1], pipe['end'][1]],
                    z=[pipe['start'][2], pipe['end'][2]],
                    mode='lines',
                    line=dict(
                        color=get_color_for_utility_type(pipe_type),
                        width=line_width,
                        dash=line_style
                    ),
                    name=f"{pipe_type} {'(Connection)' if pipe.get('is_connection', False) else ''}",
                    hovertext=hover_text,
                    hoverinfo='text'
                ))

    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        showlegend=True,
        title='3D Pipe Network Visualization',
        height=800,
        legend=dict(
            groupclick="toggleitem"
        )
    )

    return fig

def create_elevation_profile(manholes, pipes):
    fig = go.Figure()
    
    # Add manholes to elevation profile
    manhole_x = []
    manhole_y = []
    manhole_labels = []
    for manhole_id, data in manholes.items():
        manhole_x.append(data['x'])
        manhole_y.append(data['z'])
        manhole_labels.append(f"MH {manhole_id}")
        
        # Add manhole depth line
        fig.add_trace(go.Scatter(
            x=[data['x'], data['x']],
            y=[data['z'], data['z'] - data['depth']],
            mode='lines',
            line=dict(color='gray', dash='dash'),
            name=f'MH {manhole_id} Depth',
            showlegend=False
        ))

    # Add manhole points
    fig.add_trace(go.Scatter(
        x=manhole_x,
        y=manhole_y,
        mode='markers+text',
        marker=dict(size=10, color='black'),
        text=manhole_labels,
        textposition='top center',
        name='Manholes'
    ))

    fig.update_layout(
        title='Elevation Profile',
        xaxis_title='Distance (m)',
        yaxis_title='Elevation (m)',
        showlegend=True
    )
    
    return fig

def create_network_analysis(pipes):
    # Analyze pipe types and diameters
    pipe_types = defaultdict(int)
    pipe_diameters = defaultdict(list)
    
    for pipe in pipes:
        pipe_type = pipe['type']
        pipe_types[pipe_type] += 1
        pipe_diameters[pipe_type].append(pipe['diameter'])

    # Create pie chart for pipe types
    pipe_type_fig = go.Figure(data=[go.Pie(
        labels=list(pipe_types.keys()),
        values=list(pipe_types.values()),
        marker=dict(colors=[get_color_for_utility_type(t) for t in pipe_types.keys()])
    )])
    pipe_type_fig.update_layout(title='Distribution of Pipe Types')

    # Create box plot for pipe diameters by type
    diameter_data = []
    for pipe_type, diameters in pipe_diameters.items():
        diameter_data.append(go.Box(
            y=[d * 1000 for d in diameters],  # Convert to mm
            name=pipe_type,
            marker_color=get_color_for_utility_type(pipe_type)
        ))
    
    diameter_fig = go.Figure(data=diameter_data)
    diameter_fig.update_layout(
        title='Pipe Diameter Distribution by Type',
        yaxis_title='Diameter (mm)',
        showlegend=True
    )

    return pipe_type_fig, diameter_fig

def calculate_network_metrics(manholes, pipes):
    metrics = {
        'total_manholes': len(manholes),
        'total_pipes': len(pipes),
        'avg_manhole_depth': np.mean([m['depth'] for m in manholes.values()]),
        'avg_pipe_diameter': np.mean([p['diameter'] for p in pipes]) * 1000,  # mm
        'network_extent': {
            'x_min': min(m['x'] for m in manholes.values()),
            'x_max': max(m['x'] for m in manholes.values()),
            'y_min': min(m['y'] for m in manholes.values()),
            'y_max': max(m['y'] for m in manholes.values()),
        }
    }
    return metrics

def calculate_pipe_connections(manholes, pipes, params):
    """Calculate pipe connections based on connection parameters."""
    connected_pipes = []
    processed_pairs = set()  # To avoid duplicate connections

    # For each pipe endpoint
    for pipe1 in pipes:
        start1 = pipe1['start_point']
        end1 = (
            start1[0] + pipe1['direction'][0] * params['pipe_length'],
            start1[1] + pipe1['direction'][1] * params['pipe_length'],
            start1[2]
        )

        # Check for connections with other pipes
        for pipe2 in pipes:
            if pipe1 == pipe2:
                continue

            # Create a unique identifier for this pipe pair
            pair_id = tuple(sorted([pipe1['pipe_number'], pipe2['pipe_number']]))
            if pair_id in processed_pairs:
                continue

            start2 = pipe2['start_point']
            end2 = (
                start2[0] + pipe2['direction'][0] * params['pipe_length'],
                start2[1] + pipe2['direction'][1] * params['pipe_length'],
                start2[2]
            )

            # Calculate distances between pipe endpoints
            distances = [
                (math.sqrt((end1[0] - start2[0])**2 + (end1[1] - start2[1])**2), end1, start2),
                (math.sqrt((start1[0] - end2[0])**2 + (start1[1] - end2[1])**2), start1, end2),
                (math.sqrt((end1[0] - end2[0])**2 + (end1[1] - end2[1])**2), end1, end2),
                (math.sqrt((start1[0] - start2[0])**2 + (start1[1] - start2[1])**2), start1, start2)
            ]

            min_distance, point1, point2 = min(distances, key=lambda x: x[0])

            # Check if pipes should be connected based on type and distance
            max_distance = (params['pipe_to_pipe_diff_material_max_distance'] 
                          if pipe1['type'] != pipe2['type'] 
                          else params['pipe_to_pipe_max_distance'])
            
            tolerance = (params['pipe_to_pipe_diff_tolerance'] 
                       if pipe1['type'] != pipe2['type'] 
                       else params['pipe_to_pipe_tolerance'])

            if min_distance <= max_distance:
                # Calculate azimuth between points
                dx = point2[0] - point1[0]
                dy = point2[1] - point1[1]
                azimuth = math.degrees(math.atan2(dx, dy)) % 360

                # Check if the connection angle is within tolerance
                connected_pipes.append({
                    'start_point': point1,
                    'end_point': point2,
                    'type': pipe1['type'],
                    'diameter': min(pipe1['diameter'], pipe2['diameter']),
                    'is_connection': True,
                    'pipe_number': f"CONN_{pipe1['pipe_number']}_{pipe2['pipe_number']}"
                })
                
                processed_pairs.add(pair_id)

    return connected_pipes

def create_3d_visualization(manholes, pipes, pipe_length, show_manholes=True, show_pipes=True, selected_utilities=None):
    """
    Create 3D visualization of the pipe network.
    
    Args:
        manholes: Dictionary of manhole data
        pipes: List of pipe data
        pipe_length: Integer/Float value for pipe length in meters
        show_manholes: Boolean to show/hide manholes
        show_pipes: Boolean to show/hide pipes
        selected_utilities: List of utility types to display
    """
    fig = go.Figure()

    # Add manholes
    if show_manholes and manholes:
        manhole_x = []
        manhole_y = []
        manhole_z = []
        manhole_text = []
        
        for manhole_id, data in manholes.items():
            manhole_x.append(data['x'])
            manhole_y.append(data['y'])
            manhole_z.append(data['z'])
            manhole_text.append(f"Manhole {manhole_id}<br>Depth: {data['depth']:.2f}m")

        fig.add_trace(go.Scatter3d(
            x=manhole_x,
            y=manhole_y,
            z=manhole_z,
            mode='markers',
            marker=dict(
                size=8,
                color='black',
                symbol='square'
            ),
            text=manhole_text,
            hoverinfo='text',
            name='Manholes'
        ))

    # Add pipes
    if show_pipes and pipes:
        # Group pipes by type
        pipe_groups = defaultdict(list)
        
        for pipe in pipes:
            if selected_utilities is None or pipe['type'] in selected_utilities:
                start_point = pipe['start_point']
                
                if 'is_connection' in pipe and pipe['is_connection']:
                    # For connection pipes, use the pre-calculated endpoints
                    end_point = pipe['end_point']
                else:
                    # For regular pipes, calculate endpoints based on direction and length
                    direction = pipe['direction']
                    end_point = (
                        start_point[0] + direction[0] * float(pipe_length),
                        start_point[1] + direction[1] * float(pipe_length),
                        start_point[2]
                    )

                pipe_data = {
                    'start': start_point,
                    'end': end_point,
                    'type': pipe['type'],
                    'diameter': pipe['diameter'],
                    'is_connection': pipe.get('is_connection', False),
                    'pipe_number': pipe['pipe_number']
                }
                pipe_groups[pipe['type']].append(pipe_data)

        # Add pipes by type
        for pipe_type, pipes_of_type in pipe_groups.items():
            for pipe in pipes_of_type:
                try:
                    min_width = 2
                    max_width = 10
                    if pd.isna(pipe['diameter']) or pipe['diameter'] <= 0:
                        line_width = min_width
                    else:
                        line_width = min(max_width, max(min_width, pipe['diameter'] * 50))
                except:
                    line_width = min_width

                line_style = 'dot' if pipe['is_connection'] else 'solid'
                
                hover_text = (
                    f"{pipe_type}<br>"
                    f"{'Connection ' if pipe['is_connection'] else ''}"
                    f"Pipe: {pipe['pipe_number']}<br>"
                    f"Diameter: {pipe['diameter']*1000:.1f}mm"
                )

                fig.add_trace(go.Scatter3d(
                    x=[pipe['start'][0], pipe['end'][0]],
                    y=[pipe['start'][1], pipe['end'][1]],
                    z=[pipe['start'][2], pipe['end'][2]],
                    mode='lines',
                    line=dict(
                        color=get_color_for_utility_type(pipe_type),
                        width=line_width,
                        dash=line_style
                    ),
                    name=f"{pipe_type} {'(Connection)' if pipe['is_connection'] else ''}",
                    hovertext=hover_text,
                    hoverinfo='text'
                ))

    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        showlegend=True,
        title='3D Pipe Network Visualization',
        height=800,
        legend=dict(
            groupclick="toggleitem"
        )
    )

    return fig

# At the top of your file, let's make sure params are properly defined
def process_data(df, params):
    # First, ensure params has all required fields
    pipe_params = {
        'pipe_length': params.get('pipe_length', 10),
        'pipe_to_pipe_max_distance': params.get('pipe_to_pipe_max_distance', 100),
        'pipe_to_pipe_tolerance': params.get('pipe_to_pipe_tolerance', 20),
        'pipe_to_pipe_diff_material_max_distance': params.get('pipe_to_pipe_diff_material_max_distance', 50),
        'pipe_to_pipe_diff_tolerance': params.get('pipe_to_pipe_diff_tolerance', 20),
        'pipe_to_manhole_max_distance': params.get('pipe_to_manhole_max_distance', 25),
        'pipe_to_manhole_tolerance': params.get('pipe_to_manhole_tolerance', 10)
    }

    cleaned_data = clean_data(df)
    if cleaned_data is None:
        return None, None
        
    manholes = {}
    pipes = []
    valid_manholes = {}  # Store manholes with valid pipe data

    # First pass: Process manholes and identify valid ones
    for manhole_id, manhole_group in cleaned_data.groupby('exoTag'):
        first_row = manhole_group.iloc[0]
        
        # Skip invalid coordinates
        if first_row['X'] == 0 and first_row['Y'] == 0:
            continue
            
        x, y = first_row['X'], first_row['Y']
        z_ground_level = first_row['Z = Ground Level Elevation']
        depth_of_manhole = first_row['Depth of Manhole to GL']

        # Check if manhole has any valid pipe data
        has_valid_pipe = False
        for _, row in manhole_group.iterrows():
            if (not pd.isna(row['Depth of the Utilities from GL']) and 
                not pd.isna(row['Diameter of the Utilities (inch)']) and 
                not pd.isna(row['Exit Azimuth of Utility']) and 
                not pd.isna(row['Material of the Utility'])):
                has_valid_pipe = True
                break

        if has_valid_pipe:
            valid_manholes[str(manhole_id)] = {
                'x': float(x),
                'y': float(y),
                'z': float(z_ground_level),
                'depth': float(depth_of_manhole)
            }

        manholes[str(manhole_id)] = {
            'x': float(x),
            'y': float(y),
            'z': float(z_ground_level),
            'depth': float(depth_of_manhole)
        }

        # Create directional pipes for visualization
        for _, row in manhole_group.iterrows():
            try:
                # Skip rows with missing pipe data
                if (pd.isna(row['Depth of the Utilities from GL']) or 
                    pd.isna(row['Diameter of the Utilities (inch)']) or 
                    pd.isna(row['Exit Azimuth of Utility']) or 
                    pd.isna(row['Material of the Utility'])):
                    continue

                # Create directional pipe
                pipe_data = create_directional_pipe(row, x, y, z_ground_level, str(manhole_id))
                if pipe_data:
                    pipes.append(pipe_data)

            except Exception as e:
                st.warning(f"Skipping invalid pipe data: {str(e)}")
                continue

    # Second pass: Create connecting pipes between valid manholes
    processed_pairs = set()
    for manhole1_id, manhole1_data in valid_manholes.items():
        manhole1_pipes = [p for p in pipes if p['manhole_id'] == manhole1_id]
        
        for manhole2_id, manhole2_data in valid_manholes.items():
            if manhole1_id >= manhole2_id:  # Avoid duplicate connections
                continue

            pair_id = tuple(sorted([manhole1_id, manhole2_id]))
            if pair_id in processed_pairs:
                continue

            # Check if manholes have compatible utility types
            utility_types1 = set(p['type'] for p in manhole1_pipes)
            utility_types2 = set(p['type'] for p in [p for p in pipes if p['manhole_id'] == manhole2_id])
            common_utilities = utility_types1.intersection(utility_types2)

            if common_utilities:
                distance = math.sqrt(
                    (manhole1_data['x'] - manhole2_data['x'])**2 +
                    (manhole1_data['y'] - manhole2_data['y'])**2
                )

                if distance <= pipe_params['pipe_to_manhole_max_distance']:
                    # Create connecting pipe for each common utility type
                    for utility_type in common_utilities:
                        pipe1 = next((p for p in manhole1_pipes if p['type'] == utility_type), None)
                        pipe2 = next((p for p in pipes if p['manhole_id'] == manhole2_id and p['type'] == utility_type), None)
                        
                        if pipe1 and pipe2:
                            connection = create_manhole_connection(
                                manhole1_data, manhole2_data,
                                pipe1, pipe2,
                                manhole1_id, manhole2_id
                            )
                            if connection:
                                pipes.append(connection)
                                processed_pairs.add(pair_id)

    return manholes, pipes

def create_directional_pipe(row, x, y, z_ground_level, manhole_id):
    """Create a directional pipe for visualization purposes."""
    try:
        azimuth_rad = math.radians(float(row['Exit Azimuth of Utility']))
        dx = math.sin(azimuth_rad)
        dy = math.cos(azimuth_rad)

        diameter = validate_numeric(
            row['Diameter of the Utilities (inch)'] * 0.0254,  # Convert to meters
            default=0.1,
            allow_zero=False
        )

        utility_depth = validate_numeric(
            row['Depth of the Utilities from GL'],
            default=0,
            allow_zero=True
        )

        z_pipe = z_ground_level - utility_depth

        return {
            'start_point': (float(x), float(y), float(z_pipe)),
            'direction': (dx, dy, 0),
            'type': str(row['Type of Utility']),
            'diameter': diameter,
            'pipe_number': str(row['pipeTag']) if 'pipeTag' in row else 'unknown',
            'manhole_id': manhole_id,
            'is_directional': True,
            'azimuth': float(row['Exit Azimuth of Utility']),
            'material': str(row['Material of the Utility'])
        }
    except Exception as e:
        st.warning(f"Error creating directional pipe: {str(e)}")
        return None

def create_manhole_connection(manhole1_data, manhole2_data, pipe1, pipe2, manhole1_id, manhole2_id):
    """Create a connecting pipe between two manholes based on compatible pipes."""
    return {
        'start_point': (manhole1_data['x'], manhole1_data['y'], pipe1['start_point'][2]),
        'end_point': (manhole2_data['x'], manhole2_data['y'], pipe2['start_point'][2]),
        'type': pipe1['type'],
        'diameter': min(pipe1['diameter'], pipe2['diameter']),
        'pipe_number': f"CONN_{manhole1_id}_{manhole2_id}",
        'is_connection': True,
        'connected_manholes': [manhole1_id, manhole2_id],
        'material': pipe1['material']
    }
# Streamlit app
st.set_page_config(layout="wide", page_title="3D Pipe Network Visualization")

st.title("3D Pipe Network Visualization")

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload Excel File", type=["xlsx"])

if uploaded_file is not None:
    # Parameters in sidebar
    st.sidebar.header("Visualization Parameters")
    pipe_length = st.sidebar.slider("Pipe Length (m)", 1, 50, 10)
    
    # Connection parameters
    st.sidebar.header("Connection Parameters")
    pipe_to_pipe_max_distance = st.sidebar.slider("Pipe-to-Pipe Max Distance", 1, 200, 100)
    pipe_to_pipe_tolerance = st.sidebar.slider("Pipe-to-Pipe Tolerance", 1, 45, 20)
    
    pipe_to_pipe_diff_max_distance = st.sidebar.slider("Different Material Pipe Max Distance", 1, 100, 50)
    pipe_to_pipe_diff_tolerance = st.sidebar.slider("Different Material Pipe Tolerance", 1, 45, 20)
    
    pipe_to_manhole_max_distance = st.sidebar.slider("Pipe-to-Manhole Max Distance", 1, 50, 25)
    pipe_to_manhole_tolerance = st.sidebar.slider("Pipe-to-Manhole Tolerance", 1, 30, 10)

    # Create params dictionary
    params = {
        'pipe_length': pipe_length,
        'pipe_to_pipe_max_distance': pipe_to_pipe_max_distance,
        'pipe_to_pipe_tolerance': pipe_to_pipe_tolerance,
        'pipe_to_pipe_diff_material_max_distance': pipe_to_pipe_diff_max_distance,
        'pipe_to_pipe_diff_tolerance': pipe_to_pipe_diff_tolerance,
        'pipe_to_manhole_max_distance': pipe_to_manhole_max_distance,
        'pipe_to_manhole_tolerance': pipe_to_manhole_tolerance
    }

    try:
        # Process the uploaded file
        df = unmerge_cell_copy_top_value_to_df(uploaded_file, "English")
        if df is not None:
            manholes, pipes = process_data(df, params)
            
            if manholes is None or pipes is None:
                st.error("Error processing data. Please check your input file.")
                st.stop()

            # Visualization settings
            st.sidebar.header("Visualization Settings")
            show_manholes = st.sidebar.checkbox("Show Manholes", value=True)
            show_pipes = st.sidebar.checkbox("Show Pipes", value=True)
            
            # Filter utility types
            st.sidebar.header("Utility Filters")
            utility_types = list(set(pipe['type'] for pipe in pipes))
            selected_utilities = st.sidebar.multiselect(
                "Show Utility Types",
                utility_types,
                default=utility_types
            )

            # Create visualization with proper parameters
            fig = create_3d_visualization(
                manholes=manholes,
                pipes=pipes,
                pipe_length=params['pipe_length'],  # Pass pipe_length from params
                show_manholes=show_manholes,
                show_pipes=show_pipes,
                selected_utilities=selected_utilities
            )
            
            st.plotly_chart(fig, use_container_width=True)
            # Create tabs for different visualizations
            tab1, tab2, tab3, tab4 = st.tabs(["3D View", "Network Analysis", "Elevation Profile", "Export"])

            with tab1:
                # Create and display the 3D visualization
                fig = create_3d_visualization(
                    manholes, 
                    pipes, 
                    pipe_length,
                    show_manholes,
                    show_pipes,
                    selected_utilities
                )
                st.plotly_chart(fig, use_container_width=True)

                # Display current visualization parameters
                with st.expander("Current Parameters"):
                    st.write({
                        "Pipe Length": f"{pipe_length}m",
                        "Pipe-to-Pipe Distance": f"{pipe_to_pipe_max_distance}m",
                        "Pipe-to-Pipe Tolerance": f"{pipe_to_pipe_tolerance}°",
                        "Different Material Distance": f"{pipe_to_pipe_diff_max_distance}m",
                        "Different Material Tolerance": f"{pipe_to_pipe_diff_tolerance}°",
                        "Pipe-to-Manhole Distance": f"{pipe_to_manhole_max_distance}m",
                        "Pipe-to-Manhole Tolerance": f"{pipe_to_manhole_tolerance}°"
                    })

            with tab2:
                st.subheader("Network Analysis")
                col1, col2 = st.columns(2)
                
                # Create analysis visualizations
                pipe_type_fig, diameter_fig = create_network_analysis(pipes)
                
                with col1:
                    st.plotly_chart(pipe_type_fig, use_container_width=True)
                with col2:
                    st.plotly_chart(diameter_fig, use_container_width=True)

                # Display network metrics
                metrics = calculate_network_metrics(manholes, pipes)
                with st.expander("Network Metrics"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Manholes", metrics['total_manholes'])
                        st.metric("Total Pipes", metrics['total_pipes'])
                    with col2:
                        st.metric("Average Manhole Depth", f"{metrics['avg_manhole_depth']:.2f} m")
                        st.metric("Average Pipe Diameter", f"{metrics['avg_pipe_diameter']:.1f} mm")
                    with col3:
                        st.metric("Network Extent (X)", 
                                f"{metrics['network_extent']['x_max'] - metrics['network_extent']['x_min']:.1f} m")
                        st.metric("Network Extent (Y)", 
                                f"{metrics['network_extent']['y_max'] - metrics['network_extent']['y_min']:.1f} m")


            with tab4:
                st.subheader("Export Data")
                
                # Export options
                export_format = st.selectbox(
                    "Export Format",
                    ["GeoJSON", "CSV", "Excel"]
                )
                
                if st.button("Export Data"):
                    if export_format == "GeoJSON":
                        # Convert network to GeoJSON
                        features = []
                        
                        # Add manholes as points
                        for manhole_id, data in manholes.items():
                            features.append({
                                "type": "Feature",
                                "geometry": {
                                    "type": "Point",
                                    "coordinates": [data['x'], data['y'], data['z']]
                                },
                                "properties": {
                                    "id": manhole_id,
                                    "type": "manhole",
                                    "depth": data['depth']
                                }
                            })
                        
                        # Add pipes as LineStrings
                        for pipe in pipes:
                            start = pipe['start_point']
                            end = (
                                start[0] + pipe['direction'][0] * pipe_length,
                                start[1] + pipe['direction'][1] * pipe_length,
                                start[2]
                            )
                            features.append({
                                "type": "Feature",
                                "geometry": {
                                    "type": "LineString",
                                    "coordinates": [list(start), list(end)]
                                },
                                "properties": {
                                    "type": "pipe",
                                    "utility_type": pipe['type'],
                                    "diameter": pipe['diameter']
                                }
                            })
                        
                        geojson_data = {
                            "type": "FeatureCollection",
                            "features": features
                        }
                        
                        st.download_button(
                            "Download GeoJSON",
                            data=json.dumps(geojson_data, indent=2),
                            file_name="pipe_network.geojson",
                            mime="application/json"
                        )
                    
                    elif export_format == "CSV":
                        # Create separate DataFrames for manholes and pipes
                        manhole_df = pd.DataFrame.from_dict(manholes, orient='index')
                        pipe_df = pd.DataFrame(pipes)
                        
                        # Export both to CSV
                        col1, col2 = st.columns(2)
                        with col1:
                            st.download_button(
                                "Download Manholes CSV",
                                data=manhole_df.to_csv(index=True),
                                file_name="manholes.csv",
                                mime="text/csv"
                            )
                        with col2:
                            st.download_button(
                                "Download Pipes CSV",
                                data=pipe_df.to_csv(index=False),
                                file_name="pipes.csv",
                                mime="text/csv"
                            )
                    
                    else:  # Excel
                        # Create Excel file with multiple sheets
                        output = StringIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            pd.DataFrame.from_dict(manholes, orient='index').to_excel(
                                writer, sheet_name='Manholes')
                            pd.DataFrame(pipes).to_excel(
                                writer, sheet_name='Pipes')
                            pd.DataFrame([metrics]).to_excel(
                                writer, sheet_name='Network Metrics')
                        
                        st.download_button(
                            "Download Excel",
                            data=output.getvalue(),
                            file_name="pipe_network.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        logging.exception("Error in processing")

else:
    st.info("Please upload an Excel file to begin visualization.")

    # Add example/demo section
    with st.expander("About this Application"):
        st.markdown("""
        This application visualizes underground utility networks in 3D. 
        
        ### Features:
        - Interactive 3D visualization of manholes and pipes
        - Network analysis with statistics and charts
        - Elevation profiles
        - Data export in multiple formats
        
        ### How to use:
        1. Upload an Excel file containing manhole and pipe data
        2. Adjust visualization parameters using the sidebar controls
        3. Explore different views using the tabs
        4. Export the processed data in your preferred format
        
        ### Required Excel Format:
        The Excel file should contain the following columns:
        - MANHOLE NUMBER
        - X, Y coordinates
        - Ground Level Elevation
        - Depth of Manhole
        - Utility Type
        - Pipe Diameter
        - Exit Azimuth
        """)
