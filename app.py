import streamlit as st
import plotly.graph_objects as go
import numpy as np

# Example data setup
def generate_sample_data():
    """Generate some sample manholes and pipes for visualization."""
    np.random.seed(42)
    manholes = {
        f"MH{i}": {
            "x": np.random.randint(0, 100),
            "y": np.random.randint(0, 100),
            "z": np.random.randint(0, 10),
        } for i in range(1, 6)
    }

    pipes = []
    for i, (mh_id, mh_data) in enumerate(manholes.items()):
        pipes.append({
            "start_x": mh_data["x"],
            "start_y": mh_data["y"],
            "start_z": mh_data["z"],
            "end_x": mh_data["x"] + np.random.randint(-20, 20),
            "end_y": mh_data["y"] + np.random.randint(-20, 20),
            "end_z": mh_data["z"] - np.random.randint(1, 5),
        })

    return manholes, pipes

# Streamlit setup
st.title("Dynamic 3D Pipe Network Visualization")

# Sidebar for parameters
st.sidebar.header("Connection Parameters")
pipe_to_pipe_max_distance = st.sidebar.slider("Pipe-to-Pipe Max Distance", 10, 200, 100)
pipe_to_manhole_max_distance = st.sidebar.slider("Pipe-to-Manhole Max Distance", 10, 50, 25)

# Load or generate data
manholes, pipes = generate_sample_data()

# Visualization
fig = go.Figure()

# Add manholes to the plot
for mh_id, mh_data in manholes.items():
    fig.add_trace(go.Scatter3d(
        x=[mh_data["x"]],
        y=[mh_data["y"]],
        z=[mh_data["z"]],
        mode="markers",
        marker=dict(size=8, color="blue"),
        name=f"Manhole {mh_id}"
    ))

# Add pipes dynamically based on slider values
for pipe in pipes:
    # Calculate distance for filtering
    distance = np.sqrt(
        (pipe["end_x"] - pipe["start_x"])**2 +
        (pipe["end_y"] - pipe["start_y"])**2
    )
    if distance <= pipe_to_pipe_max_distance:  # Filter dynamically based on slider
        fig.add_trace(go.Scatter3d(
            x=[pipe["start_x"], pipe["end_x"]],
            y=[pipe["start_y"], pipe["end_y"]],
            z=[pipe["start_z"], pipe["end_z"]],
            mode="lines",
            line=dict(width=4, color="red"),
            name="Pipe"
        ))

# Configure the layout
fig.update_layout(
    scene=dict(
        xaxis=dict(title="X"),
        yaxis=dict(title="Y"),
        zaxis=dict(title="Z"),
    ),
    margin=dict(l=0, r=0, t=40, b=40)
)

# Display the dynamically updated plot
st.plotly_chart(fig, use_container_width=True)
