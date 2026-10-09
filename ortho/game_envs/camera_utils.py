"""
camera_utils.py — Utilities for 3D FPV camera pose manipulation.
Supports clean discrete arrow key poses: 'Center', 'Up', 'Down', 'Left', 'Right'.
"""
import math

CAMERA_POSES = ["Center", "Up", "Down", "Left", "Right"]


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
