"""
camera_utils.py — Utilities for 3D FPV camera pose manipulation.
Supports clean discrete arrow key poses: 'Center', 'Up', 'Down', 'Left', 'Right'.
"""
import math

CAMERA_POSES = ["Center", "Up", "Down", "Left", "Right"]

def transform_to_camera_frame(vx, vy, vz, cam_x, cam_y, cam_z, yaw_rad, pitch_rad):
    """
    Transforms world vertex (vx, vy, vz) into camera coordinate frame (dx_c, dy_c, d_c).
    Camera looks towards the lander:
      +d_c  : forward depth along the optical axis
      +dx_c : right
      +dy_c : up
    """
    # 1. World displacement from camera to vertex
    rx = vx - cam_x
    ry = vy - cam_y
    rz = vz - cam_z

    cos_y = math.cos(yaw_rad)
    sin_y = math.sin(yaw_rad)
    cos_p = math.cos(pitch_rad)
    sin_p = math.sin(pitch_rad)

    # 2. Camera basis vectors in world space
    # Forward vector (looking from camera toward target)
    fwd_x = -sin_y * cos_p
    fwd_y = -sin_p
    fwd_z = -cos_y * cos_p

    # Right vector
    right_x = cos_y
    right_y = 0.0
    right_z = -sin_y

    # Up vector (Right x Forward)
    up_x = -sin_y * sin_p
    up_y = cos_p
    up_z = -cos_y * sin_p

    # 3. Dot product to project into camera coordinates
    dx_c = rx * right_x + ry * right_y + rz * right_z
    dy_c = rx * up_x    + ry * up_y    + rz * up_z
    d_c  = rx * fwd_x   + ry * fwd_y   + rz * fwd_z

    return d_c, dx_c, dy_c

def apply_camera_pose(depth, lateral, vertical, camera_pose="Center", yaw_deg=20.0, pitch_deg=15.0):
    """
    Applies 3D pitch/yaw rotation corresponding to Up/Down/Left/Right arrow keys.
    
    Coordinate conventions:
      depth:    distance along forward optical axis (> 0 in front of camera)
      lateral:  horizontal coordinate (right > 0, left < 0)
      vertical: vertical coordinate (up > 0, down < 0)
      camera_pose: 'Center' | 'Up' | 'Down' | 'Left' | 'Right'
      
    Returns:
      (rotated_depth, rotated_lateral, rotated_vertical)
    """
    pose = str(camera_pose or "Center").strip().lower()
    yaw = 0.0
    pitch = 0.0
    if pose == "left":
        yaw = -math.radians(yaw_deg)
    elif pose == "right":
        yaw = math.radians(yaw_deg)
    elif pose == "up":
        pitch = math.radians(pitch_deg)
    elif pose == "down":
        pitch = -math.radians(pitch_deg)

    if yaw == 0.0 and pitch == 0.0:
        return depth, lateral, vertical

    # 1. Yaw rotation around vertical axis (turns view left/right)
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)
    d1 = depth * cos_y + lateral * sin_y
    lat1 = -depth * sin_y + lateral * cos_y

    # 2. Pitch rotation around lateral axis (tilts view up/down)
    cos_p = math.cos(pitch)
    sin_p = math.sin(pitch)
    d2 = d1 * cos_p + vertical * sin_p
    vert2 = -d1 * sin_p + vertical * cos_p

    return d2, lat1, vert2
