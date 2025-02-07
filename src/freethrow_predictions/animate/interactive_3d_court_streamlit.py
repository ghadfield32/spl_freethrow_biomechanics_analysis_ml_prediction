import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from mplbasketball.court3d import draw_court_3d
from mplbasketball.utils import transform

# Initialize session state for camera controls
if 'camera' not in st.session_state:
    st.session_state.camera = {
        'elev': 30,
        'azim': 45,
        'dist': 40,       # We'll keep this for reference, but not use it for zoom
        'zoom_factor': 1.0
    }

# We also need a default for data_zoom if it doesn't exist yet
if 'data_zoom' not in st.session_state:
    st.session_state.data_zoom = 1.0

def create_3d_court():
    """Create and configure the 3D basketball court visualization"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw court
    draw_court_3d(ax, origin=np.array([0.0, 0.0]), line_width=1.5)
    
    # Set initial view
    ax.view_init(
        elev=st.session_state.camera['elev'],
        azim=st.session_state.camera['azim']
    )
    # We'll still set ax.dist, but it won't actually zoom in Streamlit.
    ax.dist = st.session_state.camera['dist']
    
    # Set axis limits
    ax.set_xlim(-50, 50)
    ax.set_ylim(-30, 30)
    ax.set_zlim(0, 20)
    
    return fig, ax

def add_sample_data(ax):
    """Add sample trajectory and shot data"""
    # Player trajectory
    x = np.linspace(-40, 40, 100)
    y = np.linspace(-20, 20, 100)
    z = np.abs(np.sin(x * np.pi / 80) * 5)
    ax.plot(x, y, z, 'r-', lw=2, label='Player Trajectory')
    
    # Shot locations
    x_data = np.random.uniform(-40, 40, 50)
    y_data = np.random.uniform(-20, 20, 50)
    z_data = np.random.uniform(0, 20, 50)
    x_hl, y_hl = transform(x_data, y_data, fr="h", to="hl", origin="center")
    ax.scatter(x_hl, y_hl, z_data, color='blue', marker='o', label='Shot Locations')

###############################################################################
# Streamlit UI
###############################################################################
st.title("3D Basketball Court Visualization")

# Create columns for the rotation/tilt/zoom buttons
col1, col2, col3 = st.columns(3)

with col1:
    # Rotation controls
    if st.button('Rotate Left'):
        st.session_state.camera['azim'] -= 5
    if st.button('Rotate Right'):
        st.session_state.camera['azim'] += 5

with col2:
    # Elevation controls
    if st.button('Tilt Up'):
        st.session_state.camera['elev'] += 5
    if st.button('Tilt Down'):
        st.session_state.camera['elev'] -= 5

with col3:
    # CHANGED HERE: "Zoom In" & "Zoom Out" now update 'data_zoom'
    if st.button('Zoom In'):
        # Increase data_zoom but clamp to 2.0 max
        st.session_state.data_zoom = min(2.0, st.session_state.data_zoom * 1.1)
    if st.button('Zoom Out'):
        # Decrease data_zoom but clamp to 0.5 min
        st.session_state.data_zoom = max(0.5, st.session_state.data_zoom * 0.9)

# Data limits zoom slider
st.slider(
    'Data Zoom Level', 
    0.5, 
    2.0, 
    1.0, 
    key='data_zoom',
    help="Adjust the visible court area (0.5 = wide angle, 2.0 = close-up)"
)

# Create and update visualization
fig, ax = create_3d_court()
add_sample_data(ax)

# CHANGED HERE: Apply data-limits zoom with st.session_state.data_zoom
current_zoom = st.session_state.data_zoom
ax.set_xlim(-50/current_zoom, 50/current_zoom)
ax.set_ylim(-30/current_zoom, 30/current_zoom)
ax.set_zlim(0, 20/current_zoom)

# Finalize plot
ax.legend()
ax.set_title("Interactive 3D Basketball Court")

st.pyplot(fig)
