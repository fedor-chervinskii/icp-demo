import tempfile
import json
import numpy as np
import streamlit as st
import open3d as o3d
import open3d.t.pipelines.registration as treg
from icp_demo.visualization import visualize_pointclouds

MAX_POINTS = 100000


@st.cache_resource
def load_icp_demo_data():
    demo_icp_pcds = o3d.data.DemoICPPointClouds()
    pcd_source = o3d.t.io.read_point_cloud(demo_icp_pcds.paths[0])
    pcd_target = o3d.t.io.read_point_cloud(demo_icp_pcds.paths[1])
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
        pcd_source = o3d.t.io.read_point_cloud(temp_file.name)

    target_extension = uploaded_pcd_target.name.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{target_extension}") as temp_file:
        temp_file.write(uploaded_pcd_target.read())
        temp_file.seek(0)
        pcd_target = o3d.t.io.read_point_cloud(temp_file.name)
    return pcd_source, pcd_target


# Streamlit app
def main():
    st.set_page_config(layout="wide")
    st.title("Open3D ICP Demo")

    if "icp_ran" not in st.session_state:
        st.session_state.icp_ran = False
    
    pcd_target, pcd_source = None, None

    def reset_icp_ran():
        st.session_state.icp_ran = False

    pointclouds_source = st.selectbox("Select Pointclouds", ["Open3D ICP Demo", "Upload"], on_change=reset_icp_ran)

    if pointclouds_source == "Upload":
        col1, col2 = st.columns(2)
        with col1:
            uploaded_pcd_source = st.file_uploader("Upload Source Pointcloud", type=["ply", "pcd", "xyz", "xyzrgb"])

        with col2:
            uploaded_pcd_target = st.file_uploader("Upload Target Pointcloud", type=["ply", "pcd", "xyz", "xyzrgb"])

        if uploaded_pcd_source and uploaded_pcd_target:
            pcd_source, pcd_target = load_custom_pcds(uploaded_pcd_source, uploaded_pcd_target)
    else:
        pcd_source, pcd_target = load_icp_demo_data()

    if pcd_source is None or pcd_target is None:
        st.stop()

    num_source_points, num_target_points = len(pcd_source.point.positions), len(pcd_target.point.positions)
    st.markdown(f"Source Pointcloud: `{num_source_points}` points, Target Pointcloud: `{num_target_points}` points")

    num_points = num_source_points + num_target_points
    subsample_factor = max(1, num_points // MAX_POINTS)

    st.title("Input Pointclouds")

    pcd_source.paint_uniform_color([0.0, 1.0, 0.0])
    pcd_target.paint_uniform_color([1.0, 0.0, 0.0])
    
    st.plotly_chart(visualize_pointclouds([pcd_source.to_legacy(), pcd_target.to_legacy()],
                                            subsample_factor=subsample_factor,
                                            marker_size=1,
                                            names=["🟢 Source", "🔴 Target"]),
                    use_container_width=True)

    with st.expander("ICP Parameters", expanded=False):
        threshold = st.number_input("Threshold", value=0.02, min_value=0.0, step=0.01, on_change=reset_icp_ran)
        max_iter = st.number_input("Max Iterations", value=30, min_value=1, max_value=10000, step=1, on_change=reset_icp_ran)
        voxel_size = st.number_input("Downsampling Voxel Size", value=0.01, min_value=0.0, step=0.001, on_change=reset_icp_ran)

        trans_init_matrix_str = st.text_input("Initial Transformation Matrix",
            value="[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]",
            on_change=reset_icp_ran)

        try:
            trans_init_matrix = np.array(json.loads(trans_init_matrix_str))
        except json.JSONDecodeError:
            st.error("Invalid transformation matrix format. Please enter a valid JSON array.")
            st.stop()

    if st.button("Run ICP", type="primary"):
        st.session_state.icp_ran = True
        st.session_state.logs = []

    if st.session_state.icp_ran:
        log_container = st.empty()

        def callback(updated_result_dict):
            log_record = "Iteration Index: {}, Fitness: {}, Inlier RMSE: {},".format(
                updated_result_dict["iteration_index"].item(),
                updated_result_dict["fitness"].item(),
                updated_result_dict["inlier_rmse"].item())
            st.session_state.logs.append(log_record)
            log_container.text("\n".join(st.session_state.logs))

        criteria = treg.ICPConvergenceCriteria(relative_fitness=0.000001,
                                            relative_rmse=0.000001,
                                            max_iteration=max_iter)

        reg_p2p = treg.icp(
            pcd_source, pcd_target, threshold, trans_init_matrix,
            treg.TransformationEstimationPointToPoint(),
            criteria, voxel_size, callback)

        st.success("ICP Finished!")
        st.text("Final Transformation:\n" + str(reg_p2p.transformation))

        st.title("Result Pointclouds")
        pcd_source_transformed = pcd_source.clone()
        pcd_source_transformed.transform(reg_p2p.transformation)

        st.plotly_chart(visualize_pointclouds([pcd_source_transformed.to_legacy(), pcd_target.to_legacy()],
                                                subsample_factor=subsample_factor,
                                                marker_size=1,
                                                names=["🟢 Source Transformed", "🔴 Target"]),
                        use_container_width=True)


if __name__ == "__main__":
    main()
