mport open3d as o3d

def visualize_scene(reconstructed_scene, synthesized_scene):
    """
    Visualize reconstructed and synthesized 3D scenes.

    Args:
        reconstructed_scene (o3d.geometry.PointCloud): Reconstructed 3D scene.
        synthesized_scene (o3d.geometry.PointCloud): Synthesized 3D scene.
    """
    # Visualize reconstructed scene
    o3d.visualization.draw_geometries([reconstructed_scene], window_name="Reconstructed Scene")

    # Visualize synthesized scene
    o3d.visualization.draw_geometries([synthesized_scene], window_name="Synthesized Scene")
