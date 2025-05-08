import tempfile
import json
import numpy as np
import streamlit as st
import open3d as o3d
from icp_demo.visualization import visualize_pointclouds

MAX_POINTS = 100000


@st.cache_resource
def load_icp_demo_data():
    demo_icp_pcds = o3d.data.DemoICPPointClouds()
    pcd_source = o3d.io.read_point_cloud(demo_icp_pcds.paths[0])
    pcd_target = o3d.io.read_point_cloud(demo_icp_pcds.paths[1])
    trans_init = np.asarray([[0.862, 0.011, -0.507, 0.5],
                            [-0.139, 0.967, -0.215, 0.7],
                            [0.487, 0.255, 0.835, -1.4], [0.0, 0.0, 0.0, 1.0]])
    pcd_source.transform(trans_init)
    return pcd_source, pcd_target


@st.cache_resource
def load_custom_pcds(uploaded_pcd_source, uploaded_pcd_target):
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
    return pcd_source, pcd_target


# Streamlit app
def main():
    st.set_page_config(layout="wide")
    st.title("Open3D ICP Demo")
    
    pcd_target, pcd_source = None, None

    if st.toggle("Upload your Pointclouds", value=False):
        # File uploaders
        col1, col2 = st.columns(2)
        with col1:
            uploaded_pcd_source = st.file_uploader("Upload Source Pointcloud", type=["ply", "pcd", "xyz", "xyzrgb"])

        with col2:
            uploaded_pcd_target = st.file_uploader("Upload Target Pointcloud", type=["ply", "pcd", "xyz", "xyzrgb"])

        if uploaded_pcd_source and uploaded_pcd_target:
            pcd_source, pcd_target = load_custom_pcds(uploaded_pcd_source, uploaded_pcd_target)
    else:
        st.write("Using demo pointclouds.")
        pcd_source, pcd_target = load_icp_demo_data()

    if pcd_source is None or pcd_target is None:
        st.stop()

    num_points = len(pcd_source.points) + len(pcd_target.points)
    subsample_factor = max(1, num_points // MAX_POINTS)

    st.title("Input Pointclouds")
    show_rgb = st.checkbox("Show RGB", value=False)
    if not show_rgb:
        pcd_source.paint_uniform_color([0.0, 1.0, 0.0])
        pcd_target.paint_uniform_color([1.0, 0.0, 0.0])
    st.plotly_chart(visualize_pointclouds([pcd_source, pcd_target],
                                            subsample_factor=subsample_factor,
                                            marker_size=1,
                                            names=["🟢 Source", "🔴 Target"]),
                    use_container_width=True)
    
    st.title("Running ICP:")

    with st.expander("ICP Parameters", expanded=False):
        threshold = st.number_input("Threshold", value=0.02, min_value=0.0, max_value=1.0, step=0.01)
        max_iter = st.number_input("Max Iterations", value=30, min_value=1, max_value=10000, step=1)

        trans_init_matrix_str = st.text_input("Initial Transformation Matrix", value="[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]")
        
        try:
            trans_init_matrix = np.array(json.loads(trans_init_matrix_str))
        except json.JSONDecodeError:
            st.error("Invalid transformation matrix format. Please enter a valid JSON array.")
            st.stop()
    
    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd_source, pcd_target, threshold, trans_init_matrix,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter))

    st.code(f"{reg_p2p}\n Transformation is: \n {reg_p2p.transformation}", language=None)

    st.title("Result Pointclouds")
    pcd_source.transform(reg_p2p.transformation)
    if not show_rgb:
        pcd_source.paint_uniform_color([0.0, 1.0, 0.0])
        pcd_target.paint_uniform_color([1.0, 0.0, 0.0])
    st.plotly_chart(visualize_pointclouds([pcd_source, pcd_target],
                                            subsample_factor=subsample_factor,
                                            marker_size=1,
                                            names=["🟢 Source Transformed", "🔴 Target"]),
                    use_container_width=True)


if __name__ == "__main__":
    main()
