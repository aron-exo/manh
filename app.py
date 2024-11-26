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
