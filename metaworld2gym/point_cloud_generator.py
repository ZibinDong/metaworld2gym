import math
from typing import List

import numpy as np
import open3d as o3d


def quat2Mat(quat):
    if len(quat) != 4:
        print("Quaternion", quat, "invalid when generating transformation matrix.")
        raise ValueError

    w = quat[0]
    x = quat[1]
    y = quat[2]
    z = quat[3]

    x2 = x * x
    y2 = y * y
    z2 = z * z
    w2 = w * w

    xy = x * y
    zw = z * w
    xz = x * z
    yw = y * w
    yz = y * z
    xw = x * w

    rot_mat_arr = [
        x2 - y2 - z2 + w2,
        2 * (xy - zw),
        2 * (xz + yw),
        2 * (xy + zw),
        -x2 + y2 - z2 + w2,
        2 * (yz - xw),
        2 * (xz - yw),
        2 * (yz + xw),
        -x2 - y2 + z2 + w2,
    ]
    np_rot_mat = rotMatList2NPRotMat(rot_mat_arr)
    return np_rot_mat


def rotMatList2NPRotMat(rot_mat_arr):
    np_rot_arr = np.array(rot_mat_arr)
    np_rot_mat = np_rot_arr.reshape((3, 3))
    return np_rot_mat


def posRotMat2Mat(pos, rot_mat):
    t_mat = np.eye(4)
    t_mat[:3, :3] = rot_mat
    t_mat[:3, 3] = np.array(pos)
    return t_mat


def cammat2o3d(cam_mat, width, height):
    cx = cam_mat[0, 2]
    fx = cam_mat[0, 0]
    cy = cam_mat[1, 2]
    fy = cam_mat[1, 1]

    return o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)


class PointCloudGenerator(object):
    def __init__(self, sim, cam_names: List, img_size=84):
        super(PointCloudGenerator, self).__init__()

        self.sim = sim

        # this should be aligned with rgb
        self.img_width = img_size
        self.img_height = img_size

        self.cam_names = cam_names

        # List of camera intrinsic matrices
        self.cam_mats = []

        for idx in range(len(self.cam_names)):
            # get camera id
            cam_id = self.sim.model.camera_name2id(self.cam_names[idx])
            fovy = math.radians(self.sim.model.cam_fovy[cam_id])
            f = self.img_height / (2 * math.tan(fovy / 2))
            cam_mat = np.array(
                ((f, 0, self.img_width / 2), (0, f, self.img_height / 2), (0, 0, 1))
            )
            self.cam_mats.append(cam_mat)

    def generateCroppedPointCloud(self, device_id=0):
        o3d_clouds = []
        depths = []
        for cam_i in range(len(self.cam_names)):
            color_img, depth = self.captureImage(
                self.cam_names[cam_i], capture_depth=True, device_id=device_id
            )
            depths.append(depth)

            od_cammat = cammat2o3d(
                self.cam_mats[cam_i], self.img_width, self.img_height
            )
            od_depth = o3d.geometry.Image(depth)

            o3d_cloud = o3d.geometry.PointCloud.create_from_depth_image(
                od_depth, od_cammat
            )
            # Compute world to camera transformation matrix
            cam_body_id = self.sim.model.cam_bodyid[cam_i]
            cam_pos = self.sim.model.body_pos[cam_body_id]
            c2b_r = rotMatList2NPRotMat(self.sim.model.cam_mat0[cam_i])
            b2w_r = quat2Mat([0, 1, 0, 0])
            c2w_r = np.matmul(c2b_r, b2w_r)
            c2w = posRotMat2Mat(cam_pos, c2w_r)
            transformed_cloud = o3d_cloud.transform(c2w)
            o3d_clouds.append(transformed_cloud)

        combined_cloud = o3d.geometry.PointCloud()
        for cloud in o3d_clouds:
            combined_cloud += cloud
        combined_cloud_points = np.asarray(combined_cloud.points)

        combined_cloud_colors = color_img.reshape(-1, 3)
        combined_cloud = np.concatenate(
            (combined_cloud_points, combined_cloud_colors), axis=1
        )
        depths = np.array(depths).squeeze()
        return combined_cloud, depths

    # https://github.com/htung0101/table_dome/blob/master/table_dome_calib/utils.py#L160
    def depthimg2Meters(self, depth):
        extent = self.sim.model.stat.extent
        near = self.sim.model.vis.map.znear * extent
        far = self.sim.model.vis.map.zfar * extent
        image = near / (1 - depth * (1 - near / far))
        return image

    def verticalFlip(self, img):
        return np.flip(img, axis=0)

    # Render and process an image
    def captureImage(self, camera_name, capture_depth=True, device_id=0):
        rendered_images = self.sim.render(
            self.img_width,
            self.img_height,
            camera_name=camera_name,
            depth=capture_depth,
            device_id=device_id,
        )
        if capture_depth:
            img, depth = rendered_images
            depth = self.verticalFlip(depth)

            depth_convert = self.depthimg2Meters(depth)
            img = self.verticalFlip(img)
            return img, depth_convert
        else:
            img = rendered_images
            # Rendered images appear to be flipped about vertical axis
            return self.verticalFlip(img)
