import open3d as o3d
import numpy as np
import trimesh


def pcd_to_mesh(pcd):
    pcd.estimate_normals()
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 1.5 * avg_dist

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd,
        o3d.utility.DoubleVector([radius, radius * 2]))
    return mesh

def main():
    pcd = o3d.io.read_point_cloud("../bunny/reconstruction/bun_zipper.ply")
    pcd.estimate_normals()

    # estimate radius for rolling ball
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 1.5 * avg_dist

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
               pcd,
               o3d.utility.DoubleVector([radius, radius * 2]))
    # print(mesh.get_surface_area())
    o3d.visualization.draw_geometries([mesh], window_name='Open3D downSample', width=800, height=600, left=50,
                                      top=50, point_show_normal=True, mesh_show_wireframe=True, mesh_show_back_face=True,)
    o3d.io.write_triangle_mesh("D:\\AI\\3Dpointcloud\\Point-cloud-registration\\RPMNet\\RPMNet\\src\\bunny\\reconstruction\\bun_mesh.ply", mesh, write_ascii=True)
    # create the triangular mesh with the vertices and faces from open3d
    tri_mesh = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles),
                              vertex_normals=np.asarray(mesh.vertex_normals))

    trimesh.convex.is_convex(tri_mesh)

    # o3d.io.write_triangle_mesh("D:\\AI\\3Dpointcloud\\Point-cloud-registration\\RPMNet\\RPMNet\\src\\bunny\\reconstruction\\bun_mesh.ply", tri_mesh, write_ascii=True)


if __name__=='__main__':
    main()


