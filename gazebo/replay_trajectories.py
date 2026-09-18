"""ROS 2 node that replays saved NED/FRD trajectories in Gazebo ENU/FLU."""
from pathlib import Path
import numpy as np
import rclpy
from rclpy.node import Node
from ros_gz_interfaces.srv import SetEntityPose
from geometry_msgs.msg import Pose

def rot(q):
    w,x,y,z=q/np.linalg.norm(q); return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def quat(R):
    t=np.trace(R); w=np.sqrt(max(0,1+t))/2; x=np.copysign(np.sqrt(max(0,1+R[0,0]-R[1,1]-R[2,2]))/2,R[2,1]-R[1,2]); y=np.copysign(np.sqrt(max(0,1-R[0,0]+R[1,1]-R[2,2]))/2,R[0,2]-R[2,0]); z=np.copysign(np.sqrt(max(0,1-R[0,0]-R[1,1]+R[2,2]))/2,R[1,0]-R[0,1]); return w,x,y,z
class Replay(Node):
    def __init__(self):
        super().__init__('quadrotor_comparison_replay'); data=np.load(Path(__file__).parents[1]/'results/final_comparison/comparison_trajectories.npz'); self.tr={'pid_quad':data['pid'],'ppo_quad':data['ppo']}; self.k=0
        self.cli=self.create_client(SetEntityPose,'/world/quad_comparison/set_pose'); self.create_timer(.05,self.tick)
    def tick(self):
        T=np.array([[0,1,0],[1,0,0],[0,0,-1]]); B=np.diag([1,-1,-1])
        for name,s in self.tr.items():
            k=min(self.k*5,len(s)-1); p=s[k,:3]; q=quat(T@rot(s[k,6:10])@B); req=SetEntityPose.Request(); req.entity.name=name; req.entity.type=2; req.pose.position.x=float(p[1]); req.pose.position.y=float(p[0]); req.pose.position.z=float(-p[2]); req.pose.orientation.w,req.pose.orientation.x,req.pose.orientation.y,req.pose.orientation.z=map(float,q); self.cli.call_async(req)
        self.k+=1
def main(): rclpy.init(); n=Replay(); rclpy.spin(n); n.destroy_node(); rclpy.shutdown()
if __name__=='__main__': main()

