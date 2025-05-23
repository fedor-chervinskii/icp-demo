import numpy as np
import open3d as o3d
import plotly.graph_objects as go


def get_pcd_plot(np_points, np_colors, subsample_factor: int = 1, marker_size: int = 1, name: str = ""):
    return go.Scatter3d(
        x=np_points[::subsample_factor, 0], y=np_points[::subsample_factor, 1], z=np_points[::subsample_factor, 2],
        mode='markers',
        marker=dict(size=marker_size, color=np_colors[::subsample_factor]),
        name=name,
    )


def visualize_pointclouds(pointclouds: list[o3d.geometry.PointCloud],
                          subsample_factor: int = 1,
                          marker_size: int = 1,
                          camera: dict = None,
                          names: list[str] = None) -> go.Figure:
    data = []
    for i, pcd in enumerate(pointclouds):
        points = np.asarray(pcd.points)
        if pcd.colors is None:
            # generate a random color
            color = np.random.rand(3)
            colors = np.array([color] * len(points))
        else:
            colors = np.asarray(pcd.colors)
        if names is not None:
            name = names[i]
        else:
            name = f"PointCloud {i + 1}"
        data.append(get_pcd_plot(points, colors, subsample_factor, marker_size=marker_size, name=name))

    if camera is None:
        camera = dict(
            up=dict(x=0, y=-1, z=0),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.0, y=-0.4, z=-0.7)
        )

    layout = dict(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            camera=camera
        ),
        width=800,
        height=600
    )
    fig = go.Figure(data=data, layout=layout)
    return fig