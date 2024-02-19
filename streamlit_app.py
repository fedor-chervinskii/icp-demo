import tempfile
import streamlit as st
import open3d as o3d
from icp_demo.visualization import visualize_pointclouds

MAX_POINTS = 100000


# Streamlit app
def main():
    st.set_page_config(layout="wide")
    st.title("Open3D ICP Demo")

    # File uploaders
    col1, col2 = st.columns(2)
    with col1:
        uploaded_pcd_source = st.file_uploader("Upload Source Pointcloud", type=["ply", "pcd", "xyz", "xyzrgb"])

    with col2:
        uploaded_pcd_target = st.file_uploader("Upload Target Pointcloud", type=["ply", "pcd", "xyz", "xyzrgb"])

    if uploaded_pcd_source and uploaded_pcd_target:
        # save to tempfile, preserving the extension
        source_extension = uploaded_pcd_source.name.split(".")[-1]
        with tempfile.NamedTemporaryFile(delete=True, suffix=f".{source_extension}") as temp_file:
            temp_file.write(uploaded_pcd_source.read())
            temp_file.seek(0)
            pcd_source = o3d.io.read_point_cloud(temp_file.name)

        target_extension = uploaded_pcd_target.name.split(".")[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{target_extension}") as temp_file:
            temp_file.write(uploaded_pcd_target.read())
            temp_file.seek(0)
            pcd_target = o3d.io.read_point_cloud(temp_file.name)

        num_points = len(pcd_source.points) + len(pcd_target.points)
        subsample_factor = max(1, num_points // MAX_POINTS)
        st.plotly_chart(visualize_pointclouds([pcd_source, pcd_target],
                                              subsample_factor=subsample_factor,
                                              marker_size=1))


if __name__ == "__main__":
    main()
