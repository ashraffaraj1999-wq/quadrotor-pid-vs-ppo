"""ROS 2/PX4 offboard setpoint adapter for a trained policy or PID supervisor.

This intentionally sends position/yaw setpoints: PX4 retains estimation and inner-loop
stabilization. It is not imported by the Gym environment, keeping training independent.
"""
import rclpy
from rclpy.node import Node
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleOdometry

class OffboardAdapter(Node):
    def __init__(self):
        super().__init__('quadrotor_offboard_adapter'); self.count=0; self.odom=None
        qos=10
        self.mode_pub=self.create_publisher(OffboardControlMode,'/fmu/in/offboard_control_mode',qos)
        self.sp_pub=self.create_publisher(TrajectorySetpoint,'/fmu/in/trajectory_setpoint',qos)
        self.cmd_pub=self.create_publisher(VehicleCommand,'/fmu/in/vehicle_command',qos)
        self.create_subscription(VehicleOdometry,'/fmu/out/vehicle_odometry',self._odom,qos)
        self.create_timer(0.05,self.tick)  # 20 Hz, safely above PX4's 2 Hz proof-of-life minimum
    def _odom(self,msg): self.odom=msg
    def command(self,command,**params):
        m=VehicleCommand(); m.timestamp=self.get_clock().now().nanoseconds//1000; m.command=command
        m.target_system=1; m.target_component=1; m.source_system=1; m.source_component=1; m.from_external=True
        for i,(k,v) in enumerate(params.items(),1): setattr(m,f'param{i}',float(v))
        self.cmd_pub.publish(m)
    def tick(self):
        now=self.get_clock().now().nanoseconds//1000
        mode=OffboardControlMode(); mode.timestamp=now; mode.position=True; self.mode_pub.publish(mode)
        sp=TrajectorySetpoint(); sp.timestamp=now; sp.position=[0.0,0.0,-2.0]; sp.yaw=0.0; self.sp_pub.publish(sp)
        self.count+=1
        if self.count==20:
            self.command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE,param1=1.0,param2=6.0)
            self.command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,param1=1.0)
def main():
    rclpy.init(); node=OffboardAdapter(); rclpy.spin(node); node.destroy_node(); rclpy.shutdown()
if __name__=='__main__': main()

