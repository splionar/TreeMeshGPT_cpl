import open3d as o3d
import numpy as np

# This is to cut mesh in z-axis. Only for testing.

def cut_mesh_half_z(mesh_path, keep="top", z_cut=None):
    """
    Load mesh and cut it in half along Z axis.

    Parameters
    ----------
    mesh_path : str
        Path to mesh file.
    keep : str
        "top"  -> keep z >= z_cut
        "bottom" -> keep z <= z_cut
    z_cut : float or None
        Cutting plane. If None, use mesh center along Z.

    Returns
    -------
    o3d.geometry.TriangleMesh
        Cut mesh.
    """

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    # default cut = middle of mesh bounding box
    if z_cut is None:
        z_cut = (vertices[:, 2].min() + vertices[:, 2].max()) / 2.0

    # decide which vertices to keep
    if keep == "top":
        valid_vertex = vertices[:, 2] >= z_cut
    elif keep == "bottom":
        valid_vertex = vertices[:, 2] <= z_cut
    else:
        raise ValueError("keep must be 'top' or 'bottom'")

    # keep triangles whose ALL vertices are valid
    tri_mask = np.all(valid_vertex[triangles], axis=1)
    new_triangles = triangles[tri_mask]

    # remove unused vertices and reindex
    used_vertices = np.unique(new_triangles.flatten())
    new_vertices = vertices[used_vertices]

    reindex = -np.ones(len(vertices), dtype=int)
    reindex[used_vertices] = np.arange(len(used_vertices))
    new_triangles = reindex[new_triangles]

    cut_mesh = o3d.geometry.TriangleMesh()
    cut_mesh.vertices = o3d.utility.Vector3dVector(new_vertices)
    cut_mesh.triangles = o3d.utility.Vector3iVector(new_triangles)
    cut_mesh.compute_vertex_normals()

    return cut_mesh

mesh = cut_mesh_half_z("demo/objaverse_nut.obj", keep="top")
o3d.io.write_triangle_mesh("demo/objaverse_nut_half.obj", mesh)
