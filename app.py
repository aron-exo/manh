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
        'MANHOLE NUMBER': 'exoTag'
    }
    
    # Rename columns
    df = df.rename(columns=column_mapping)
    
    # Remove rows where both X and Y are 0 or missing
    df = df[~((df['X'].fillna(0) == 0) & (df['Y'].fillna(0) == 0))]
    
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

    # Add pipes
    if show_pipes:
        pipe_endpoints = calculate_pipe_endpoints(manholes, pipes, pipe_length)
        
        # Group pipes by type
        pipe_groups = defaultdict(list)
        for pipe in pipe_endpoints:
            if selected_utilities is None or pipe['type'] in selected_utilities:
                pipe_groups[pipe['type']].append(pipe)

        # Add pipes by type
        for pipe_type, pipes_of_type in pipe_groups.items():
            for pipe in pipes_of_type:
                # Validate and adjust line width
                try:
                    min_width = 2
                    max_width = 10
                    if pd.isna(pipe['diameter']) or pipe['diameter'] <= 0:
                        line_width = min_width
                    else:
                        line_width = min(max_width, max(min_width, pipe['diameter'] * 50))
                except:
                    line_width = min_width

                fig.add_trace(go.Scatter3d(
                    x=[pipe['start'][0], pipe['end'][0]],
                    y=[pipe['start'][1], pipe['end'][1]],
                    z=[pipe['start'][2], pipe['end'][2]],
                    mode='lines',
                    line=dict(
                        color=get_color_for_utility_type(pipe_type),
                        width=line_width
                    ),
                    name=pipe_type,
                    hovertext=f"{pipe_type}<br>Diameter: {pipe['diameter']*1000:.1f}mm",
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

def process_data(df, params):
    cleaned_data = clean_data(df)
    manholes = {}
    pipes = []

    # Process manholes and pipes
    for manhole_id, manhole_group in cleaned_data.groupby('exoTag'):
        first_row = manhole_group.iloc[0]
        
        # Skip invalid coordinates
        if first_row['X'] == 0 and first_row['Y'] == 0:
            continue
            
        x, y = first_row['X'], first_row['Y']
        z_ground_level = first_row['Z = Ground Level Elevation']
        depth_of_manhole = first_row['Depth of Manhole to GL']

        manholes[str(manhole_id)] = {
            'x': float(x),
            'y': float(y),
            'z': float(z_ground_level),
            'depth': float(depth_of_manhole)
        }

        # Create pipes for each utility in the manhole
        for _, row in manhole_group.iterrows():
            try:
                if pd.isna(row['Exit Azimuth of Utility']):
                    continue

                # Skip if coordinates are invalid
                if row['X'] == 0 and row['Y'] == 0:
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

                pipes.append({
                    'start_point': (float(x), float(y), float(z_pipe)),
                    'direction': (dx, dy, 0),
                    'type': str(row['Type of Utility']),
                    'diameter': diameter
                })
            except Exception as e:
                st.warning(f"Skipping invalid pipe data: {str(e)}")
                continue

    return manholes, pipes

# Streamlit app
st.set_page_config(layout="wide", page_title="3D Pipe Network Visualization")

st.title("3D Pipe Network Visualization")

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload Excel File", type=["xlsx"])

if uploaded_file is not None:
    # Parameters in sidebar
    st.sidebar.header("Visualization Parameters")
    pipe_length = st.sidebar.slider("Pipe Length (m)", 1, 50, 10)
    
    st.sidebar.header("Connection Parameters")
    pipe_to_pipe_max_distance = st.sidebar.slider("Pipe-to-Pipe Max Distance", 1, 200, 100)
    pipe_to_pipe_tolerance = st.sidebar.slider("Pipe-to-Pipe Tolerance", 1, 45, 20)
    
    pipe_to_pipe_diff_max_distance = st.sidebar.slider("Different Material Pipe Max Distance", 1, 100, 50)
    pipe_to_pipe_diff_tolerance = st.sidebar.slider("Different Material Pipe Tolerance", 1, 45, 20)
    
    pipe_to_manhole_max_distance = st.sidebar.slider("Pipe-to-Manhole Max Distance", 1, 50, 25)
    pipe_to_manhole_tolerance = st.sidebar.slider("Pipe-to-Manhole Tolerance", 1, 30, 10)

    params = {
        'pipe_to_pipe_max_distance': pipe_to_pipe_max_distance,
        'pipe_to_pipe_min_tolerance': pipe_to_pipe_tolerance,
        'pipe_to_pipe_max_tolerance': pipe_to_pipe_tolerance,
        'pipe_to_pipe_diff_material_max_distance': pipe_to_pipe_diff_max_distance,
        'pipe_to_pipe_diff_material_min_tolerance': pipe_to_pipe_diff_tolerance,
        'pipe_to_pipe_diff_material_max_tolerance': pipe_to_pipe_diff_tolerance,
        'pipe_to_manhole_max_distance': pipe_to_manhole_max_distance,
        'pipe_to_manhole_min_tolerance': pipe_to_manhole_tolerance,
        'pipe_to_manhole_max_tolerance': pipe_to_manhole_tolerance
    }

    try:
        # Process the uploaded file
        df = unmerge_cell_copy_top_value_to_df(uploaded_file, "English")
        if df is not None:
            manholes, pipes = process_data(df, params)

            # Visualization settings
            st.sidebar.header("Visualization Settings")
            show_manholes = st.sidebar.checkbox("Show Manholes", value=True)
# Visualization settings (continued)
            show_pipes = st.sidebar.checkbox("Show Pipes", value=True)
            
            # Filter utility types
            st.sidebar.header("Utility Filters")
            utility_types = list(set(pipe['type'] for pipe in pipes))
            selected_utilities = st.sidebar.multiselect(
                "Show Utility Types",
                utility_types,
                default=utility_types
            )

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
