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
        direction_x = np.random.uniform(-1, 1)
        direction_y = np.random.uniform(-1, 1)
        direction_z = np.random.uniform(-0.5, 0.5)
        pipes.append({
            "start_x": mh_data["x"],
            "start_y": mh_data["y"],
            "start_z": mh_data["z"],
            "direction_x": direction_x,
            "direction_y": direction_y,
            "direction_z": direction_z,
        })

    return manholes, pipes

# Streamlit setup
st.title("Dynamic 3D Pipe Network Visualization")

# Sidebar for parameters
st.sidebar.header("Pipe Length Adjustment")
pipe_length = st.sidebar.slider("Pipe Length", 1, 50, 10)  # Slider to adjust pipe length

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

# Add pipes dynamically based on the pipe_length slider
for pipe in pipes:
    end_x = pipe["start_x"] + pipe["direction_x"] * pipe_length
    end_y = pipe["start_y"] + pipe["direction_y"] * pipe_length
    end_z = pipe["start_z"] + pipe["direction_z"] * pipe_length

    fig.add_trace(go.Scatter3d(
        x=[pipe["start_x"], end_x],
        y=[pipe["start_y"], end_y],
        z=[pipe["start_z"], end_z],
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
