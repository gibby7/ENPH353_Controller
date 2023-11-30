#! /usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from rosgraph_msgs.msg import Clock
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

start = False
start_time = 0
current_time_sec = 0


def clock_callback(msg):
   global current_time_sec 
   current_time_sec = msg.clock.to_sec()
   #rospy.loginfo("Current simulation time (seconds): %f", current_time_sec)

def image_callback(msg):
   try:
      bridge = CvBridge()
      cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
         
   except Exception as e:
      print(e)

   

rospy.init_node('pub_sub_node')
cmd = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1)
score = rospy.Publisher('/score_tracker', String, queue_size=1)
clk = rospy.Subscriber('/clock', Clock, clock_callback)
cam = rospy.Subscriber('/R1/pi_camera/image_raw topic', Image, image_callback)
rate = rospy.Rate(2)
move = Twist()
move.linear.x = 0.5
move.angular.z = 0.0

rospy.sleep(1)

while not rospy.is_shutdown():
   if not start:
      score.publish("TEAM,pword,0,BALLS")
      start = True
      start_time = rospy.Time.now().to_sec()

   #if current_time_sec - start_time > 3:
   #   move.linear.x = 0
   #   score.publish("TEAM,pword,-1,BALLS")

   print(start_time)
   print(current_time_sec)

   cmd.publish(move)
   rate.sleep()