import streamlit as st
import folium
from folium import plugins
import pandas as pd
import numpy as np
import math
import json
from streamlit_folium import folium_static
from pyproj import Transformer

# Configure page
st.set_page_config(layout="wide", page_title="2D Pipe Network Visualization")
st.title("Underground Utility Network Visualization")

# Initialize coordinate transformer
def init_coordinate_transformer(source_epsg="EPSG:2039", target_epsg="EPSG:4326"):
    """Initialize coordinate transformer from source to WGS84"""
    return Transformer.from_crs(source_epsg, target_epsg, always_xy=True)

def transform_coordinates(transformer, x, y):
    """Transform coordinates from source CRS to WGS84"""
    lon, lat = transformer.transform(x, y)
    return lat, lon  # Folium expects coordinates as (lat, lon)

def get_color_for_utility_type(utility_type):
    """Get color for different utility types"""
    color_map = {
        'Sewer': 'brown',
        'Storm': 'green',
        'Water': 'blue',
        'Gas': 'red',
        'Electric': 'yellow',
        'Communication': 'orange',
        'Unknown': 'gray',
        'Other': 'purple'
    }
    return color_map.get(utility_type, 'gray')

def create_2d_map(manholes, pipes, selected_utilities=None, show_manholes=True, show_pipes=True):
    """Create a folium map with manholes and pipes"""
    
    # Initialize coordinate transformer
    transformer = init_coordinate_transformer()
    
    # Get center coordinates from average of manhole positions
    center_x = np.mean([m['x'] for m in manholes.values()])
    center_y = np.mean([m['y'] for m in manholes.values()])
    center_lat, center_lon = transform_coordinates(transformer, center_x, center_y)
    
    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=17,
        tiles=None  # We'll add tiles as layers
    )
    
    # Add different tile layers
    folium.TileLayer('OpenStreetMap').add_to(m)
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Esri Satellite',
        overlay=False
    ).add_to(m)
    
    # Create feature groups for different utility types
    utility_groups = {}
    if selected_utilities is None:
        selected_utilities = set(pipe['type'] for pipe in pipes)
    
    for utility_type in selected_utilities:
        utility_groups[utility_type] = folium.FeatureGroup(name=utility_type)
    
    # Add manholes to map
    if show_manholes:
        manholes_group = folium.FeatureGroup(name='Manholes')
        for manhole_id, data in manholes.items():
            lat, lon = transform_coordinates(transformer, data['x'], data['y'])
            folium.CircleMarker(
                location=[lat, lon],
                radius=5,
                color='black',
                fill=True,
                popup=f"""
                <b>Manhole {manhole_id}</b><br>
                Depth: {data['depth']:.2f}m<br>
                Ground Level: {data['z']:.2f}m
                """,
                fill_opacity=0.7
            ).add_to(manholes_group)
        manholes_group.add_to(m)
    
    # Add pipes to map
    if show_pipes:
        for pipe in pipes:
            if pipe['type'] not in selected_utilities:
                continue
                
            # Get pipe coordinates
            if 'is_connection' in pipe and pipe['is_connection']:
                # For connection pipes
                start_lat, start_lon = transform_coordinates(transformer, 
                    pipe['start_point'][0], pipe['start_point'][1])
                end_lat, end_lon = transform_coordinates(transformer,
                    pipe['end_point'][0], pipe['end_point'][1])
            else:
                # For directional pipes
                start_lat, start_lon = transform_coordinates(transformer,
                    pipe['start_point'][0], pipe['start_point'][1])
                end_x = pipe['start_point'][0] + pipe['direction'][0] * pipe_length
                end_y = pipe['start_point'][1] + pipe['direction'][1] * pipe_length
                end_lat, end_lon = transform_coordinates(transformer, end_x, end_y)
            
            # Create pipe line
            line_style = {'color': get_color_for_utility_type(pipe['type']),
                         'weight': 3,
                         'opacity': 0.8,
                         'dashArray': '5, 5' if pipe.get('is_connection', False) else None}
            
            folium.PolyLine(
                locations=[[start_lat, start_lon], [end_lat, end_lon]],
                popup=f"""
                <b>{pipe['type']} Pipe</b><br>
                ID: {pipe['pipe_number']}<br>
                Diameter: {pipe['diameter']*1000:.1f}mm<br>
                {'Connection' if pipe.get('is_connection', False) else 'Direct'}
                """,
                **line_style
            ).add_to(utility_groups[pipe['type']])
    
    # Add utility groups to map
    for group in utility_groups.values():
        group.add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Add draw control for measurements
    plugins.Draw(
        export=False,
        position='topleft',
        draw_options={'polygon': False, 'rectangle': False, 'circlemarker': False}
    ).add_to(m)
    
    # Add measurement control
    plugins.MeasureControl(
        position='bottomleft',
        primary_length_unit='meters',
        secondary_length_unit='kilometers'
    ).add_to(m)
    
    return m

def validate_numeric(value, default=0.0, allow_zero=True):
    """Validate and convert numeric values"""
    try:
        result = float(value)
        if pd.isna(result):
            return default
        if not allow_zero and result == 0:
            return default
        return result
    except (ValueError, TypeError):
        return default

def clean_data(df):
    """Clean and validate the input data"""
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
    
    # Filter out rows without pipe numbers
    if 'pipeTag' in df.columns:
        df['pipeTag'] = df['pipeTag'].astype(str)
        df = df[df['pipeTag'].notna() & (df['pipeTag'] != 'nan') & (df['pipeTag'].str.strip() != '')]
    else:
        st.error("PIPE NUMBER column not found in the Excel file.")
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
    
    # Ensure non-numeric columns are strings
    string_columns = ['Type of Utility', 'Material of the Utility', 'exoTag']
    for col in string_columns:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown').astype(str)
            df = df[df[col].str.strip() != '']
    
    # Remove rows with invalid coordinates
    df = df[df['X'].notna() & df['Y'].notna() & (df['X'] != 0) & (df['Y'] != 0)]
    
    return df

def process_data(df, params):
    """Process the cleaned data into manholes and pipes"""
    # Ensure params has required fields
    pipe_params = {
        'pipe_length': params.get('pipe_length', 10),
    }

    cleaned_data = clean_data(df)
    if cleaned_data is None:
        return None, None
        
    manholes = {}
    manhole_pipes = {}
    pipes = []

    # Process manholes and pipes
    for manhole_id, manhole_group in cleaned_data.groupby('exoTag'):
        manhole_id = str(manhole_id)
        first_row = manhole_group.iloc[0]
        
        # Skip invalid coordinates
        if first_row['X'] == 0 and first_row['Y'] == 0:
            continue
            
        x, y = first_row['X'], first_row['Y']
        z_ground_level = first_row['Z = Ground Level Elevation']
        depth_of_manhole = first_row['Depth of Manhole to GL']

        # Store manhole data
        manholes[manhole_id] = {
            'x': float(x),
            'y': float(y),
            'z': float(z_ground_level),
            'depth': float(depth_of_manhole)
        }
        manhole_pipes[manhole_id] = []

        # Create pipes for each utility in the manhole
        for _, row in manhole_group.iterrows():
            try:
                # Skip rows with missing pipe data
                if (pd.isna(row['Depth of the Utilities from GL']) or 
                    pd.isna(row['Diameter of the Utilities (inch)']) or 
                    pd.isna(row['Exit Azimuth of Utility']) or 
                    pd.isna(row['Type of Utility'])):
                    continue

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

                pipe_data = {
                    'start_point': (float(x), float(y), float(z_pipe)),
                    'direction': (dx, dy, 0),
                    'type': str(row['Type of Utility']),
                    'diameter': diameter,
                    'pipe_number': str(row['pipeTag']) if 'pipeTag' in row else 'unknown',
                    'manhole_id': manhole_id,
                    'azimuth': float(row['Exit Azimuth of Utility']),
                    'material': str(row.get('Material of the Utility', 'Unknown'))
                }
                
                pipes.append(pipe_data)
                manhole_pipes[manhole_id].append(pipe_data)

            except Exception as e:
                st.warning(f"Skipping invalid pipe data for manhole {manhole_id}: {str(e)}")
                continue

    # Calculate connecting pipes
    connecting_pipes = calculate_connecting_pipes(manholes, manhole_pipes)
    pipes.extend(connecting_pipes)

    return manholes, pipes

def calculate_connecting_pipes(manholes, manhole_pipes):
    """Calculate connections between manholes with matching utilities"""
    connecting_pipes = []
    processed_pairs = set()

    for manhole1_id, pipes1 in manhole_pipes.items():
        for manhole2_id, pipes2 in manhole_pipes.items():
            if manhole1_id >= manhole2_id:
                continue

            pair_id = tuple(sorted([manhole1_id, manhole2_id]))
            if pair_id in processed_pairs:
                continue

            # Find matching utilities between manholes
            utilities1 = {pipe['type'] for pipe in pipes1}
            utilities2 = {pipe['type'] for pipe in pipes2}
            common_utilities = utilities1.intersection(utilities2)

            if common_utilities:
                m1 = manholes[manhole1_id]
                m2 = manholes[manhole2_id]
                
                # Create connection for each common utility type
                for utility_type in common_utilities:
                    pipe1 = next((p for p in pipes1 if p['type'] == utility_type), None)
                    pipe2 = next((p for p in pipes2 if p['type'] == utility_type), None)
                    
                    if pipe1 and pipe2:
                        connecting_pipes.append({
                            'start_point': (m1['x'], m1['y'], pipe1['start_point'][2]),
                            'end_point': (m2['x'], m2['y'], pipe2['start_point'][2]),
                            'type': utility_type,
                            'diameter': min(pipe1['diameter'], pipe2['diameter']),
                            'pipe_number': f"CONN_{manhole1_id}_{manhole2_id}_{utility_type}",
                            'is_connection': True,
                            'material': pipe1['material'],
                            'manhole_id': manhole1_id
                        })

                processed_pairs.add(pair_id)

    return connecting_pipes

# File uploader and main app logic
uploaded_file = st.sidebar.file_uploader("Upload Excel File", type=["xlsx"])

if uploaded_file is not None:
    try:
        # Read and process data
        df = pd.read_excel(uploaded_file, sheet_name="English", skiprows=3)
        
        # Process parameters
        pipe_length = st.sidebar.slider("Pipe Length (m)", 1, 50, 10)
        
        # Visualization settings
        show_manholes = st.sidebar.checkbox("Show Manholes", value=True)
        show_pipes = st.sidebar.checkbox("Show Pipes", value=True)
        
        # Process data and create visualization
        manholes, pipes = process_data(df, {'pipe_length': pipe_length})
        
        if manholes and pipes:
            # Utility type filter
            utility_types = list(set(pipe['type'] for pipe in pipes))
            selected_utilities = st.sidebar.multiselect(
                "Show Utility Types",
                utility_types,
                default=utility_types
            )
            
            # Create map
            m = create_2d_map(
                manholes, 
                pipes, 
                selected_utilities=selected_utilities,
                show_manholes=show_manholes,
                show_pipes=show_pipes
            )
            
            # Display map
            folium_static(m, width=1200, height=800)
            
            # Display statistics
            with st.expander("Network Statistics"):
                metrics = calculate_network_metrics(manholes, pipes)
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
                    
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        st.exception(e)
else:
    st.info("Please upload an Excel file to begin visualization.")
