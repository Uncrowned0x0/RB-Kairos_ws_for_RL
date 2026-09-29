"""
Gazebo Harmonic utilities for KAIROS Pick & Place RL environment.
Handles spawning/deletion of cubes and model pose querying.
"""
import os
import subprocess
import random
import xml.etree.ElementTree as ET


def generate_cube_sdf(name: str, size: float, mass: float) -> str:
    """Generate in-memory SDF model string for a cube with given size and mass."""
    inertia = (1.0 / 6.0) * mass * size * size
    color_r = random.uniform(0.2, 0.9)
    color_g = random.uniform(0.2, 0.9)
    color_b = random.uniform(0.2, 0.9)

    sdf = f"""<?xml version="1.0" ?>
<sdf version="1.8">
  <model name="{name}">
    <link name="link">
      <inertial>
        <mass>{mass}</mass>
        <inertia>
          <ixx>{inertia}</ixx><iyy>{inertia}</iyy><izz>{inertia}</izz>
          <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz>
        </inertia>
      </inertial>
      <collision name="collision">
        <geometry>
          <box><size>{size} {size} {size}</size></box>
        </geometry>
        <surface>
          <friction>
            <ode><mu>1.0</mu><mu2>1.0</mu2></ode>
          </friction>
        </surface>
      </collision>
      <visual name="visual">
        <geometry>
          <box><size>{size} {size} {size}</size></box>
        </geometry>
        <material>
          <ambient>{color_r} {color_g} {color_b} 1</ambient>
          <diffuse>{color_r} {color_g} {color_b} 1</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>"""
    return sdf


def spawn_cube(name: str, x: float, y: float, z: float,
               size: float, mass: float, world: str = "laboratoire") -> bool:
    """Spawn a cube inside Gazebo Harmonic via gz service using a temporary SDF file."""
    sdf_str = generate_cube_sdf(name, size, mass)
    
    # Write SDF to a unique temporary file per process to avoid race conditions between parallel workers
    tmp_path = f"/tmp/rl_cube_{os.getpid()}_{random.randint(1000, 9999)}.sdf"
    with open(tmp_path, 'w') as f:
        f.write(sdf_str)

    cmd = (
        f'gz service -s /world/{world}/create '
        f'--reqtype gz.msgs.EntityFactory '
        f'--reptype gz.msgs.Boolean '
        f'--timeout 5000 '
        f'--req \'sdf_filename: "{tmp_path}" name: "{name}" '
        f'pose: {{position: {{x: {x}, y: {y}, z: {z}}}}}\''
    )
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True,
                                text=True, timeout=10)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def delete_model(name: str, world: str = "laboratoire") -> bool:
    """Delete an entity model from Gazebo Harmonic via gz service."""
    cmd = (
        f'gz service -s /world/{world}/remove '
        f'--reqtype gz.msgs.Entity '
        f'--reptype gz.msgs.Boolean '
        f'--timeout 5000 '
        f'--req \'name: "{name}" type: MODEL\''
    )
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True,
                                text=True, timeout=10)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def set_model_pose(name: str, x: float, y: float, z: float,
                   yaw: float = 0.0, world: str = "laboratoire") -> bool:
    """
    Reposition an existing Gazebo model via gz service set_pose.
    Much faster than delete + spawn (no SDF file parsing).

    Args:
        name: Name of Gazebo model (e.g. 'rl_cube')
        x, y, z: New target position in meters (world frame)
        yaw: Orientation angle around Z axis (radians)
        world: Gazebo world name
    Returns:
        True if command succeeded, False otherwise
    """
    import math
    qw = math.cos(yaw / 2.0)
    qz = math.sin(yaw / 2.0)
    cmd = (
        f'gz service -s /world/{world}/set_pose '
        f'--reqtype gz.msgs.Pose '
        f'--reptype gz.msgs.Boolean '
        f'--timeout 3000 '
        f'--req \'name: "{name}" position: {{x: {x}, y: {y}, z: {z}}} '
        f'orientation: {{w: {qw}, x: 0.0, y: 0.0, z: {qz}}}\''
    )
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True,
                                text=True, timeout=5)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False



def get_model_pose(name: str, world: str = "laboratoire"):
    """
    Retrieve model pose via gz topic.
    Returns (x, y, z) tuple or None if unreachable.
    """
    cmd = (
        f'gz topic -e -t /model/{name}/pose -n 1'
    )
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True,
                                text=True, timeout=5)
        if result.returncode == 0 and result.stdout:
            # Parse the protobuf text format output
            output = result.stdout
            x = _extract_field(output, 'x')
            y = _extract_field(output, 'y')
            z = _extract_field(output, 'z')
            if x is not None and y is not None and z is not None:
                return (x, y, z)
    except subprocess.TimeoutExpired:
        pass
    return None


def _extract_field(text: str, field_name: str):
    """Extract a floating-point numeric value from protobuf text format output."""
    import re
    # Match field_name: value in protobuf text format
    pattern = rf'{field_name}\s*:\s*([-\d.e+]+)'
    match = re.search(pattern, text)
    if match:
        return float(match.group(1))
    return None


def pause_simulation(world: str = "laboratoire") -> bool:
    """Pause Gazebo simulation."""
    cmd = (
        f'gz service -s /world/{world}/control '
        f'--reqtype gz.msgs.WorldControl '
        f'--reptype gz.msgs.Boolean '
        f'--timeout 5000 '
        f'--req \'pause: true\''
    )
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True,
                                text=True, timeout=10)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def unpause_simulation(world: str = "laboratoire") -> bool:
    """Resume Gazebo simulation."""
    cmd = (
        f'gz service -s /world/{world}/control '
        f'--reqtype gz.msgs.WorldControl '
        f'--reptype gz.msgs.Boolean '
        f'--timeout 5000 '
        f'--req \'pause: false\''
    )
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True,
                                text=True, timeout=10)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def spawn_target_marker(name: str, x: float, y: float, z: float,
                        radius: float = 0.03, world: str = "laboratoire") -> bool:
    """
    Spawn a static visual sphere marker (no gravity, no collision)
    to visualize target coordinate in Gazebo.
    """
    sdf = f"""<?xml version="1.0" ?>
<sdf version="1.8">
  <model name="{name}">
    <static>true</static>
    <link name="link">
      <visual name="visual">
        <geometry>
          <sphere><radius>{radius}</radius></sphere>
        </geometry>
        <material>
          <ambient>0.0 1.0 0.0 0.8</ambient>
          <diffuse>0.0 1.0 0.0 0.8</diffuse>
          <emissive>0.0 0.6 0.0 1.0</emissive>
        </material>
      </visual>
      <!-- No collision, no inertial = purely visual marker -->
    </link>
  </model>
</sdf>"""
    tmp_path = f"/tmp/rl_marker_{os.getpid()}_{random.randint(1000, 9999)}.sdf"
    with open(tmp_path, 'w') as f:
        f.write(sdf)

    cmd = (
        f'gz service -s /world/{world}/create '
        f'--reqtype gz.msgs.EntityFactory '
        f'--reptype gz.msgs.Boolean '
        f'--timeout 5000 '
        f'--req \'sdf_filename: "{tmp_path}" name: "{name}" '
        f'pose: {{position: {{x: {x}, y: {y}, z: {z}}}}}\''
    )
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True,
                                text=True, timeout=10)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def sample_cube_on_table(arm_mount_x: float = 0.1873,
                         arm_mount_y: float = 0.0,
                         arm_mount_z: float = 0.844,
                         table_z: float = 0.76,
                         min_reach: float = 0.28,
                         max_reach: float = 0.72) -> tuple:
    """
    Sample an (x, y) position strictly on the training tables
    (front table or diagonal wings) with safety margins to prevent
    falling into empty space and ensure accessibility within UR5e reach.
    """
    import math
    for _ in range(50):
        # 50% on front table, 25% left wing, 25% right wing
        choice = random.choice(['front', 'front', 'left', 'right'])
        if choice == 'front':
            x = random.uniform(0.47, 0.77)
            y = random.uniform(-0.25, 0.25)
        elif choice == 'left':
            u = random.uniform(-0.12, 0.12)
            v = random.uniform(-0.16, 0.16)
            yaw = 0.61
            x = 0.55 + u * math.cos(yaw) - v * math.sin(yaw)
            y = 0.62 + u * math.sin(yaw) + v * math.cos(yaw)
        else:
            u = random.uniform(-0.12, 0.12)
            v = random.uniform(-0.16, 0.16)
            yaw = -0.61
            x = 0.55 + u * math.cos(yaw) - v * math.sin(yaw)
            y = -0.62 + u * math.sin(yaw) + v * math.cos(yaw)

        # Verify point is within UR5e kinematic reach
        d = math.sqrt((x - arm_mount_x)**2 + (y - arm_mount_y)**2 + (table_z - arm_mount_z)**2)
        if min_reach <= d <= max_reach:
            return x, y

    return 0.58, 0.0



