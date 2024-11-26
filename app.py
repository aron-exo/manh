import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import math
import logging
from io import StringIO

# Configure logging
logging.basicConfig(level=logging.INFO)

def unmerge_cell_copy_top_value_to_df(uploaded_file, sheet_name):
    df = pd.read_excel(uploaded_file, sheet_name=sheet_name, skiprows=3)
    return df

def clean_data(df):
    # Map your Excel columns to the expected column names
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
    
    # Convert numeric columns
    numeric_columns = ['X', 'Y', 'Z = Ground Level Elevation', 'Depth of Manhole to GL',
                      'Depth of the Utilities from GL', 'Diameter of the Utilities (inch)',
                      'Exit Azimuth of Utility']
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

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

def create_3d_visualization(manholes, pipes, pipe_length):
    fig = go.Figure()

    # Add manholes as markers
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
    pipe_endpoints = calculate_pipe_endpoints(manholes, pipes, pipe_length)
    
    # Group pipes by type for better visualization
    pipe_groups = {}
    for pipe in pipe_endpoints:
        pipe_type = pipe['type']
        if pipe_type not in pipe_groups:
            pipe_groups[pipe_type] = []
        pipe_groups[pipe_type].append(pipe)

    # Add pipes by type
    for pipe_type, pipes_of_type in pipe_groups.items():
        for pipe in pipes_of_type:
            fig.add_trace(go.Scatter3d(
                x=[pipe['start'][0], pipe['end'][0]],
                y=[pipe['start'][1], pipe['end'][1]],
                z=[pipe['start'][2], pipe['end'][2]],
                mode='lines',
                line=dict(
                    color=get_color_for_utility_type(pipe_type),
                    width=pipe['diameter'] * 100  # Scale diameter for visibility
                ),
                name=pipe_type
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
        height=800
    )

    return fig

def process_data(df, params):
    cleaned_data = clean_data(df)
    manholes = {}
    pipes = []

    # Process manholes and pipes
    for manhole_id, manhole_group in cleaned_data.groupby('exoTag'):
        first_row = manhole_group.iloc[0]
        x, y = first_row['X'], first_row['Y']
        z_ground_level = first_row['Z = Ground Level Elevation']
        depth_of_manhole = first_row['Depth of Manhole to GL']

        manholes[str(manhole_id)] = {
            'x': x,
            'y': y,
            'z': z_ground_level,
            'depth': depth_of_manhole
        }

        # Create pipes for each utility in the manhole
        for _, row in manhole_group.iterrows():
            if pd.isna(row['Exit Azimuth of Utility']):
                continue

            azimuth_rad = math.radians(row['Exit Azimuth of Utility'])
            dx = math.sin(azimuth_rad)
            dy = math.cos(azimuth_rad)

            diameter = float(row['Diameter of the Utilities (inch)']) * 0.0254  # Convert to meters
            utility_depth = float(row['Depth of the Utilities from GL'])
            z_pipe = z_ground_level - utility_depth

            pipes.append({
                'start_point': (x, y, z_pipe),
                'direction': (dx, dy, 0),
                'type': row['Type of Utility'],
                'diameter': diameter
            })

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
        manholes, pipes = process_data(df, params)

        # Create and display the 3D visualization
        fig = create_3d_visualization(manholes, pipes, pipe_length)
        st.plotly_chart(fig, use_container_width=True)

        # Display statistics
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Network Statistics")
            st.write(f"Total Manholes: {len(manholes)}")
            st.write(f"Total Pipes: {len(pipes)}")

        with col2:
            st.subheader("Utility Types")
            utility_counts = {}
            for pipe in pipes:
                utility_type = pipe['type']
                utility_counts[utility_type] = utility_counts.get(utility_type, 0) + 1
            
            for utility_type, count in utility_counts.items():
                st.write(f"{utility_type}: {count} pipes")

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        logging.exception("Error in processing")
else:
    st.info("Please upload an Excel file to begin visualization.")
