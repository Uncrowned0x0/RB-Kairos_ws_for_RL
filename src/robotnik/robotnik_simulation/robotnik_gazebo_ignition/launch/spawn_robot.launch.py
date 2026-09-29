# Copyright (c) 2025, Robotnik Automation S.L.L.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the Robotnik Automation S.L.L. nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL Robotnik Automation S.L.L. BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import tempfile
import yaml
import os


from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import LaunchConfiguration
from launch.substitutions import SubstitutionFailure
from launch.substitutions import Command, FindExecutable
from launch.substitutions import PathJoinSubstitution

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from robotnik_common.launch import AddArgumentParser, ExtendedArgument


from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Union, Optional
from launch import SomeSubstitutionsType, SomeSubstitutionsType_types_tuple
from launch.frontend.parse_substitution import parse_substitution
from launch.utilities import normalize_to_list_of_substitutions, perform_substitutions
from launch.utilities.typing_file_path import FilePath
from launch.substitution import Substitution

from launch import LaunchContext
from launch.conditions import IfCondition


# TODO: move this utility class into robotnik_common
class ConfigFile(Substitution):
    """Substitution to get the path of the configuration file."""

    def __init__(
        self,
        param_file: Union[FilePath, SomeSubstitutionsType],
    ) -> None:
        """
        Construct a parameter file description.

        :param param_file: The path to the parameter file or a substitution that resolves to it.
        """
        self.__evaluated_param_file: Optional[Path] = None
        self.__created_tmp_file = False

        self.__param_file = param_file
        if isinstance(param_file, SomeSubstitutionsType_types_tuple):
            self.__param_file = normalize_to_list_of_substitutions(param_file)  # type: ignore

    def perform(self, context: LaunchContext) -> str:
        """Substitute the parameter file path."""
        param_file = self.__param_file
        if isinstance(param_file, list):
            # list of substitutions
            param_file = perform_substitutions(context, self.__param_file)  # type: ignore

        param_file_path: Path = Path(param_file)  # type: ignore
        with open(param_file_path, 'r') as f, NamedTemporaryFile(
                mode='w', prefix='launch_params_', delete=False
            ) as h:
                parsed = perform_substitutions(context, parse_substitution(f.read()))  # type: ignore
                # Strip only the YAML document separator '---', but keep '/**:' 
                # because the ROS 2 Jazzy spawner requires either the exact namespace 
                # or the /** wildcard in the parameter file.
                if parsed.startswith('---\n'):
                    parsed = parsed[len('---\n'):]
                try:
                    yaml.safe_load(parsed)
                except Exception:
                    raise SubstitutionFailure(
                        'The substituted parameter file is not a valid yaml file')
                h.write(parsed)
                param_file_path = Path(h.name)
                self.__created_tmp_file = True
        self.__evaluated_param_file = param_file_path
        return str(param_file_path)

    def cleanup(self) -> None:
        """Remove the temporary file if it was created."""
        if self.__created_tmp_file and self.__evaluated_param_file is not None:
            try:
                self.__evaluated_param_file.unlink()
            except FileNotFoundError:
                # The file may have been deleted already, ignore this error
                pass
            self.__evaluated_param_file = None

    def __del__(self):
        """Clean up the temporary file when the object is deleted."""
        self.cleanup()


def load_yaml(package_path, relative_path):
    full_path = os.path.join(package_path, relative_path)
    with open(full_path, "r") as f:
        return yaml.safe_load(f)

def substitute_param_context(param, context):
    """Resolve a parameter if it is a LaunchConfiguration."""
    if isinstance(param, LaunchConfiguration):
        return param.perform(context)
    return param

def launch_setup(context, params):
    ret = []
    
    # Inject frame_prefix into context so that ConfigFile can evaluate $(var frame_prefix)
    robot_id_val = substitute_param_context(params['robot_id'], context)
    context.launch_configurations['frame_prefix'] = robot_id_val + '_'
    gripper_type = substitute_param_context(params['gripper_type'], context)
    # Strict gripper validation: only schunk_egk50 and tesollo_dg5f are supported
    if gripper_type not in ['schunk_egk50', 'tesollo_dg5f']:
        print(f"[WARN] Gripper '{gripper_type}' is unsupported or obsolete. Defaulting to 'schunk_egk50'. Supported grippers: 'schunk_egk50', 'tesollo_dg5f'.")
        gripper_type = 'schunk_egk50'
    params['gripper_type'] = gripper_type

    # Robot Description
    ret.append(IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('robotnik_description'), '/launch/robot_description.launch.py'
        ]),
        launch_arguments={
            'verbose': 'false',
            'robot_xacro_path': params['robot_xacro_path'],
            'frame_prefix': [params['robot_id'], '_'],
            'namespace': params['robot_id'],
            'gazebo_ignition': 'true',
            'arm_type': params['arm_type'],
            'gripper_type': gripper_type,
            'low_performance_simulation': params['low_performance_simulation']
        }.items(),
    ))

    # Spawner
    ret.append(Node(
        package='ros_gz_sim',
        executable='create',
        namespace=params['robot_id'],
        arguments=[
            '-name', params['robot_id'],
            '-topic', "robot_description",
            '-robot_namespace', params['robot_id'],
            '-x', params['x'],
            '-y', params['y'],
            '-z', params['z'],
        ],
        output='screen',
    ))

    # Gazebo bridge
    def generate_bridge_yaml(params) -> str:
        robot_id = substitute_param_context(params['robot_id'], context)
        bridge_raw = [
            (f"/{robot_id}/imu/data", f"/{robot_id}/imu/data", "sensor_msgs/msg/Imu", "ignition.msgs.IMU", "GZ_TO_ROS"),
            (f"/{robot_id}/gps/data", f"/{robot_id}/gps/fix", "sensor_msgs/msg/NavSatFix", "ignition.msgs.NavSat", "GZ_TO_ROS"),
        ]
        def add_camera(camera_name):
            bridge_raw.extend([
                (f"/{robot_id}/{camera_name}_camera_color/color/camera_info", f"/{robot_id}/{camera_name}_rgbd_camera/color/camera_info", "sensor_msgs/msg/CameraInfo", "gz.msgs.CameraInfo", "GZ_TO_ROS"),
                (f"/{robot_id}/{camera_name}_camera_color/color/image_raw", f"/{robot_id}/{camera_name}_rgbd_camera/color/image_raw", "sensor_msgs/msg/Image", "gz.msgs.Image", "GZ_TO_ROS"),
            ])
        def add_laser(laser_name):
            bridge_raw.extend([
                (f"/{robot_id}/{laser_name}_laser/scan", f"/{robot_id}/{laser_name}_laser/scan", "sensor_msgs/msg/LaserScan", "gz.msgs.LaserScan", "GZ_TO_ROS"),
            ])
        def add_pointcloud(points_name):
            bridge_raw.extend([
                ( f"/{robot_id}/{points_name}_lidar/scan/points", f"/{robot_id}/{points_name}_laser/points", "sensor_msgs/msg/PointCloud2", "gz.msgs.PointCloudPacked", "GZ_TO_ROS"),
            ])

        def add_depth_camera(camera_name):
            bridge_raw.extend([
                (f"/{robot_id}/{camera_name}_camera_depth/depth/camera_info", f"/{robot_id}/{camera_name}_rgbd_camera/depth/camera_info", "sensor_msgs/msg/CameraInfo", "gz.msgs.CameraInfo", "GZ_TO_ROS"),
                (f"/{robot_id}/{camera_name}_camera_depth/depth/image_raw", f"/{robot_id}/{camera_name}_rgbd_camera/depth/image_raw", "sensor_msgs/msg/Image", "gz.msgs.Image", "GZ_TO_ROS"),
            ])

        add_camera("front")
        add_camera("rear")
        add_camera("top_ptz")
        #add_depth_camera("front")
        add_laser("front")
        add_laser("rear")
        add_pointcloud("top")

        gripper_type = substitute_param_context(params['gripper_type'], context)
        if gripper_type == 'tesollo_dg5f':
            for i in range(1, 5):
                bridge_raw.append(
                    (f"/contact/tip_{i}", f"/{robot_id}/contact/tip_{i}", "ros_gz_interfaces/msg/Contacts", "gz.msgs.Contacts", "GZ_TO_ROS")
                )

        bridge_config = [{"ros_topic_name": ros, "gz_topic_name": gz, "ros_type_name": ros_type, "gz_type_name": gz_type, "direction": direction} for gz, ros, ros_type, gz_type, direction in bridge_raw]
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp:
            yaml.dump(bridge_config, tmp)
            return tmp.name

    bridge_yaml = generate_bridge_yaml(params)
    ret.append(Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        parameters=[
            {'config_file': bridge_yaml},
        ],
        namespace=params['robot_id'],
    ))

    # Xela Mock Node & Tactile Estimator
    if gripper_type == 'tesollo_dg5f':
        ret.append(Node(
            package='tesollo_tactile_mock',
            executable='xela_mock_node',
            namespace=params['robot_id'],
            output='screen',
        ))
        ret.append(Node(
            package='tesollo_tactile_mock',
            executable='tesollo_tactile_estimator',
            namespace=params['robot_id'],
            output='screen',
        ))


    def extract_controllers_from_yaml(yaml_path):

        data = {}
        existing_controllers = []
        # Load the YAML file
        with open(yaml_path, 'r') as f:

            # Read the file content
            content = f.read()
            # Remove the string "---\n/**:" if it exists at the beginning
            if content.startswith('---\n/**:'):
                content = content[len('---\n/**:'):]
            # Move file pointer back to start for yaml.safe_load
            f.seek(0)
            f = tempfile.SpooledTemporaryFile(mode='w+')
            f.write(content)
            f.seek(0)

            try:
                data = yaml.safe_load(f)
            except Exception as e:
                raise RuntimeError(f"Failed to parse YAML file '{yaml_path}': {e}")

        for controller in data:
            if controller != 'controller_manager':
                existing_controllers.append(controller)
        return existing_controllers

    def get_ros2_control_yaml_path(params):
        base_path = (
            Path(FindPackageShare('robotnik_gazebo_ignition').perform(context))
            / 'config'
            / 'profile'
            / substitute_param_context(params['robot'], context)
        )
        robot_model = substitute_param_context(params['robot_model'], context)
        return str(base_path / f'{robot_model}_ros2_control.yaml')

    path = get_ros2_control_yaml_path(params)
    new_controllers = extract_controllers_from_yaml(path)

    # ROS2 control
    controllers =  ['--controller-manager-timeout', '60', '--service-call-timeout', '60', 'joint_state_broadcaster']
    # Replace default joint_state_broadcaster by the one defined in the specific
    # ros2_control.yaml for the robot model
    if 'joint_state_broadcaster' in new_controllers:
        new_controllers.remove('joint_state_broadcaster')
    
    # Filter out obsolete/unsupported controllers unconditionally
    if 'qbhand_synergy_controller' in new_controllers:
        new_controllers.remove('qbhand_synergy_controller')
    if 'onrobot_rg6_controller' in new_controllers:
        new_controllers.remove('onrobot_rg6_controller')

    # Filter schunk controller if gripper is not schunk_egk50
    if gripper_type != 'schunk_egk50' and 'schunk_egk50_controller' in new_controllers:
        new_controllers.remove('schunk_egk50_controller')
        
    # Filter tesollo controller if gripper is not tesollo_dg5f
    if gripper_type != 'tesollo_dg5f' and 'tesollo_dg5f_controller' in new_controllers:
        new_controllers.remove('tesollo_dg5f_controller')

    controllers.extend(new_controllers)
    print("Controllers to be spawned:", controllers)

    robot_controller_config = ConfigFile(path)

    controllers.append('--param-file')
    controllers.append(
         robot_controller_config, # type: ignore
    )

    # Separate joint_state_broadcaster to ensure it loads even if other controllers fail
    ret.append(Node(
        package='controller_manager',
        executable='spawner',
        namespace=params['robot_id'],
        arguments=['--controller-manager-timeout', '60', '--service-call-timeout', '60', 'joint_state_broadcaster'],
        output='screen',
    ))

    # Spawn each other controller individually
    for c in new_controllers:
        ret.append(Node(
            package='controller_manager',
            executable='spawner',
            namespace=params['robot_id'],
            arguments=['--controller-manager-timeout', '60', '--service-call-timeout', '60', c, '--param-file', robot_controller_config],
            output='screen',
        ))

    # Check if rviz config path is modified, if not use default fixed frame
    rviz_config_default = str(
        Path(
            FindPackageShare('robotnik_gazebo_ignition').perform(context)
        )
        / 'config'
        / 'rviz_config.rviz'
    )
    use_fixed_frame = False
    # Determine if fixed frame should be used
    if isinstance(params['rviz_config'], LaunchConfiguration):
        rviz_config_value = params['rviz_config'].perform(context)
        use_fixed_frame = (rviz_config_value == "")
    else:
        use_fixed_frame = (params['rviz_config'] == "")

    if use_fixed_frame:
        params['rviz_config'] = rviz_config_default

    use_sim_time = {"use_sim_time": True}

    # RViz
    ret.append(Node(
        package="rviz2",
        executable="rviz2",
        namespace=params['robot_id'],
        arguments=[
            # Fixed frame
            ['-f', params['robot_id'], '_odom'] if use_fixed_frame else [],
            # Config file
            '-d', [params['rviz_config']],
            # Window name
            '-t', [params['robot_id'], ' - ', params['robot_model'], ' - navigation RViz'],
        ],
        parameters=[
            use_sim_time,
            ],
        condition=IfCondition(params['run_rviz'])
    ))

    return ret


def generate_launch_description():
    # Ensure resource paths are configured for Gazebo
    from ament_index_python.packages import get_package_share_directory, PackageNotFoundError
    resource_paths = []
    for pkg in ["robotnik_gazebo_ignition", "robotnik_description", "robotnik_sensors", "ur_description", "rbkairos_description"]:
        try:
            pkg_share = get_package_share_directory(pkg)
            parent_share = os.path.dirname(pkg_share)
            if parent_share not in resource_paths:
                resource_paths.append(parent_share)
        except PackageNotFoundError:
            pass

    for env_var in ["GZ_SIM_RESOURCE_PATH", "IGN_GAZEBO_RESOURCE_PATH", "GAZEBO_RESOURCE_PATH"]:
        current_value = os.environ.get(env_var, "")
        new_values = [p for p in resource_paths if p not in current_value.split(":")]
        if new_values:
            os.environ[env_var] = ":".join(new_values) + (":" + current_value if current_value else "")

    raw_args = [
        ("robot_id", "Unique Robot Identifier", "robot", "ROBOT_ID"),
        ("robot", "Robot Model Name", "rbwatcher", "ROBOT"),
        ("robot_model", "Robot Variant or Type", LaunchConfiguration('robot'), "ROBOT_MODEL"),
        ("robot_xacro_path", "Path to Robot Xacro File", [FindPackageShare('robotnik_description'), '/robots/', LaunchConfiguration('robot'), '/', LaunchConfiguration('robot_model'), '.urdf.xacro'], "ROBOT_XACRO_PATH"),
        ("x", "Initial X Coordinate", "0.0", "X"),
        ("y", "Initial Y Coordinate", "0.0", "Y"),
        ("z", "Initial Z Coordinate", "0.0", "Z"),
        ("arm_type", "Type of robotic arm", "ur10e", "ARM_TYPE"),
        ("gripper_type", "Type of gripper (schunk_egk50 | tesollo_dg5f)", "schunk_egk50", "GRIPPER_TYPE"),
        ("run_rviz", "Run RViz", "True", "RUN_RVIZ"),
        ("rviz_config", "RViz configuration file", "", "CONFIG_RVIZ"),
        ("use_sim_time", "Use simulation time", "True", "USE_SIM_TIME"),
        ("low_performance_simulation", "Enable Low Performance Simulation", "False", "LOW_PERFORMANCE_SIMULATION"),
    ]

    ld = LaunchDescription()
    add_to_launcher = AddArgumentParser(ld)
    for arg in raw_args:
        extended_arg = ExtendedArgument(
            name=arg[0],
            description=arg[1],
            default_value=arg[2],
            use_env=True,
            environment=arg[3],
        )
        add_to_launcher.add_arg(extended_arg)
    params = add_to_launcher.process_arg()
    ld.add_action(OpaqueFunction(function=launch_setup, args=[params]))
    return ld
