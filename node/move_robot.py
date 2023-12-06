#! /usr/bin/env python3

import rospy
import os
from geometry_msgs.msg import Twist
from rosgraph_msgs.msg import Clock
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from tensorflow.keras.models import load_model


start = False
start_time = 0
current_time_sec = 0

chars = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',' ']

size_entries = []
victim_entries = []
crime_entries = []
time_entries = []
place_entries = []
motive_entries = []
weapon_entries = []
bandit_entries = []

clue_values = {
    "SIZE": {'value': 1, 'entries': size_entries, 'submitted': False},
    "VICTIM": {'value': 2, 'entries': victim_entries, 'submitted': False},
    "CRIME": {'value': 3, 'entries': crime_entries, 'submitted': False},
    "TIME": {'value': 4, 'entries': time_entries, 'submitted': False},
    "PLACE": {'value': 5, 'entries': place_entries, 'submitted': False},
    "MOTIVE": {'value': 6, 'entries': motive_entries, 'submitted': False},
    "WEAPON": {'value': 7, 'entries': weapon_entries, 'submitted': False},
    "BANDIT": {'value': 8, 'entries': bandit_entries, 'submitted': False}
}

# For checking state
passed_pedestrian = False
passed_truck = False
on_grass = False
passed_marked_road = False
through_tunnel = False

# Load the model

current_directory = os.path.dirname(os.path.abspath(__file__))
model_file_path = os.path.join(current_directory, "reader.h5")
conv_model = load_model(model_file_path)

# PID CONTROLLER

class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint):
        self.Kp = Kp  # Proportional gain
        self.Ki = Ki  # Integral gain
        self.Kd = Kd  # Derivative gain
        self.setpoint = setpoint  # Desired position (line)

        self.error_prior = 0
        self.output_prior = 0
        self.integral = 0

    def compute(self, position):
        # Calculate the error
        error = self.setpoint - position

        # Proportional term
        P = self.Kp * error

        # Integral term with anti-windup
        self.integral += error
        I = self.Ki * self.integral

        # Derivative term
        derivative = error - self.error_prior
        D = self.Kd * derivative

        # PID controller output
        output = P + I + D

        # Update error for next iteration
        self.error_prior = error

        return output
    
controller = PIDController(.006, .000001, .001, 640)



class MotionDetector:
    def __init__(self, buffer_size=5, threshold=250000):
        self.has_moved = True  # Default behavior: Assume movement initially
        self.buffer = []       # Buffer to store previous frames
        self.buffer_size = buffer_size  # Size of the buffer
        self.threshold = threshold  # Threshold for significant change

    def safe_to_move(self, image):
        image = cv2.blur(image, (7,7))
        if len(self.buffer) == self.buffer_size:
            # Check for significant change between the current frame and the frame 10 frames ago
            diff = np.sum(np.abs(self.buffer[-1] - self.buffer[0]))
            print(diff)
            if diff > self.threshold:
                self.has_moved = False
                self.buffer = []  # Clear buffer after detecting no movement
                return not self.has_moved

            # Remove the oldest frame to maintain the buffer size
            self.buffer.pop(0)

        self.buffer.append(image)  # Add current frame to the buffer
        return not self.has_moved

pedestrian_detector = MotionDetector()



def clock_callback(msg):
   global current_time_sec 
   current_time_sec = msg.clock.to_sec()
   #rospy.loginfo("Current simulation time (seconds): %f", current_time_sec)



def get_four_points(points):
    # Convert the list of points to a NumPy array
    points = np.array(points)

    # Find the convex hull of the points
    hull = cv2.convexHull(points)

    # Approximate the convex hull with a quadrilateral
    epsilon = 0.02 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)

    # If the approximation has exactly four points, calculate the area
    if len(approx) == 4:
        area = cv2.contourArea(approx)
        return approx
    else:
        return None
   


def image_callback(msg):
   try:
      global score
      bridge = CvBridge()
      cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")

      ### FROM COLAB

      # Define the lower and upper bounds for the blue color in HSV
      lower_blue = np.array([120, 60, 60])
      upper_blue = np.array([130, 255, 255])

      # Convert the image to HSV
      image_hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

      # Create a mask for the blue color
      mask = cv2.inRange(image_hsv, lower_blue, upper_blue)

      contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

      # Find the contour with the second largest area
      largest_contour = max(contours, key=cv2.contourArea)
      contours = [contour for contour in contours if contour is not largest_contour]
      second_largest_contour = max(contours, key=cv2.contourArea)

      height, width = cv_image.shape[:2]

      # Define the coordinates of the rectangle in the original image
      rect_points = get_four_points(second_largest_contour)

      def euclidean_norm(point):
          return np.linalg.norm(point)
      sorted_points = np.array(sorted(rect_points, key=euclidean_norm))

      # Define the coordinates of the rectangle in the new image (the rectangle will become the full image)
      new_rect_points = np.array([[0, 0], [0, 399], [599, 0], [599, 399]], dtype=np.float32)

      # Find the homography matrix
      homography_matrix, _ = cv2.findHomography(sorted_points, new_rect_points)

      # Warp the perspective to get the new image
      result_image = cv2.warpPerspective(cv_image, homography_matrix, (width, height))

      # and finally, isolate the letters

      top_letters = []
      bottom_letters = []

      for i in range(6):
        predict_img = result_image[50:120,250+45*i:295+45*i,:]
        top_letters.append(predict_img)

      for i in range(12):
        predict_img = result_image[260:330,30+45*i:75+45*i,:]
        bottom_letters.append(predict_img)

      top_letters = np.array(top_letters)
      bottom_letters = np.array(bottom_letters)

      prediction = conv_model.predict(top_letters)

      clue_type = ""

      for i in prediction:
        max_index = np.argmax(i)
        clue_type += chars[max_index]
      
      clue_type = clue_type.replace(" ","")

      # Add to list if a clue type

      if clue_type in clue_values:

         clue_string = ""

         prediction = conv_model.predict(bottom_letters)
         for i in prediction:
            max_index = np.argmax(i)
            clue_string += chars[max_index]
         
         clue_string = clue_string.replace(" ","")
         
         clue_values[clue_type]['entries'].append(clue_string)
            
   except Exception as e:
      print(e)




def road_pid(img):
   hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

   # Define lower and upper threshold values for dark gray in HSV
   lower_gray = np.array([0, 0, 50])  
   upper_gray = np.array([3, 3, 100])  

   # Create a mask using inRange() to detect dark gray regions in HSV
   mask_gray = cv2.inRange(hsv_image, lower_gray, upper_gray)

   mask_gray = cv2.morphologyEx(mask_gray, cv2.MORPH_ERODE, (3,3))

   pid_row = mask_gray[450, :]

   nonzero_indices = np.nonzero(pid_row)[0]  # Get the indices of non-zero elements

   if len(nonzero_indices) > 0:
      # Calculate the centroid (average position)
      location = np.mean(nonzero_indices)
   else:
      location = 640

   location -= 70

   turn = controller.compute(location)

   move.linear.x = 0.25
   move.angular.z = turn

   #print("Location = " + str(location))
   #print("Turn = " + str(turn))



def grass_pid(img):
   hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

   # Define lower and upper threshold values for dark gray in HSV
   lower_road = np.array([0, 28, 160])  
   upper_road = np.array([50, 90, 255])  

   # Create a mask using inRange() to detect dark gray regions in HSV
   mask_road = cv2.inRange(hsv_image, lower_road, upper_road)

   mask_road = cv2.morphologyEx(mask_road, cv2.MORPH_ERODE, (3,3))

   # Find contours
   contours, _ = cv2.findContours(mask_road, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

   # Sort contours by area (largest to smallest)
   contours = sorted(contours, key=cv2.contourArea, reverse=True)

   # Get the two largest contours
   largest_contours = contours[:15]

   # Create a mask image to draw the contours
   contour_mask = np.zeros_like(img)

   # Draw the largest contours on the mask
   cv2.drawContours(contour_mask, largest_contours, -1, (255), thickness=cv2.FILLED)

   # Create an image with only the two largest contours
   contour_image = np.zeros_like(img)
   contour_image[contour_mask == 255] = img[contour_mask == 255]

   pid_row = contour_image[430:470, :]

   nonzero_indices = [np.nonzero(row)[0] for row in pid_row]

   # Calculate centroids along each row
   centroids = []
   for indices in nonzero_indices:
      if len(indices) > 0:
         centroid = np.mean(indices)
         centroids.append(centroid)
      else:
         centroids.append(640)  # Default value if no non-zero elements are found

   # Compute the average centroid position
   location = np.mean(centroids)

   #location -= 50

   print("Location = " + str(location))

   turn = controller.compute(location)

   move.linear.x = 0.25
   move.angular.z = turn

   #print("Location = " + str(location))
   #print("Turn = " + str(turn))



def check_red(img):
  red_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  lower_red = np.array([0, 100, 100])
  upper_red = np.array([10, 255, 255])  
  mask_red = cv2.inRange(red_hsv, lower_red, upper_red)
  if np.sum(mask_red == 255)/np.size(mask_red) > .0165:
    return True
  return False



def check_pink(img):
  pink_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  lower_pink = np.array([140, 100, 100]) 
  upper_pink = np.array([170, 255, 255])
  mask_pink = cv2.inRange(pink_hsv, lower_pink, upper_pink)
  if np.sum(mask_pink == 255)/np.size(mask_pink) > .015:
    return True
  return False



def check_truck(img):
  lower_black = np.array([0, 0, 0], dtype=np.uint8)      # Lower threshold for black
  upper_black = np.array([15, 15, 15], dtype=np.uint8)  # Upper threshold for black
  black_mask = cv2.inRange(img, lower_black, upper_black)
  print("Black Pixels: " + str(np.sum(black_mask == 255)))
  if np.sum(black_mask == 255) > 300:
    return True
  return False



def pid_callback(msg):   
   try:
      bridge = CvBridge()
      img = bridge.imgmsg_to_cv2(msg, "bgr8")  
      global passed_pedestrian
      global passed_truck
      global on_grass

      # State 1: Start - Red Line

      if not passed_pedestrian:
         if check_red(img):

            hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # Define a lower and upper HSV threshold for white skin tones
            lower_blue = np.array([80, 40, 40])
            upper_blue = np.array([110, 255, 255])  # Upper threshold for blue

            # Create a mask using inRange to filter white skin tones
            blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

            move.linear.x = 0
            move.angular.z = 0
            safe = pedestrian_detector.safe_to_move(blue_mask[360:,450:920])
            if safe:      
               passed_pedestrian = True
               pass
         else:
            road_pid(img)
            
      # State 2: Red Line - Roundabout

      elif not passed_truck:

         # check for truck when 3 clues submitted

         submitted_count = sum(1 for clue in clue_values.values() if clue["submitted"])
         if submitted_count >= 3:
            move.linear.x = 0
            move.angular.z = 0

            if check_truck(img):
               passed_truck = True

         else:
            road_pid(img)

      # State 3: Roundabout - Pink Line

      elif not on_grass:
         if check_pink(img):
            on_grass = True
         else:
            road_pid(img)

      # State 4: Pink Line - End of Marked Road
      
      elif not passed_marked_road:
         grass_pid(img)
         pass
      else:
         grass_pid(img)

   except Exception as e:
      print(e)



rospy.init_node('pub_sub_node')
cmd = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1)
score = rospy.Publisher('/score_tracker', String, queue_size=1)
clk = rospy.Subscriber('/clock', Clock, clock_callback)
cam = rospy.Subscriber('/R1/pi_camera/image_raw', Image, image_callback)
pid = rospy.Subscriber('/R1/pi_camera/image_raw', Image, pid_callback)
rate = rospy.Rate(2)
move = Twist()
#move.linear.x = 0.0
#move.angular.z = 0.0



rospy.sleep(1)

while not rospy.is_shutdown():
   if not start:
      score.publish("T12,123456,0,BALLS")
      start = True
      start_time = rospy.Time.now().to_sec()
   
   for case_name, case_data in clue_values.items():
      entries = case_data['entries']
      submitted = case_data['submitted']

      threshold = 2
      submitted_count = sum(1 for clue in clue_values.values() if clue['submitted'])

      if submitted_count >= 3:
         threshold = 1

      if len(entries) >= threshold:
         unique_elements, counts = np.unique(entries, return_counts=True)
         most_common_index = np.argmax(counts)
         most_common_entry = unique_elements[most_common_index]
         print(f"Case {case_name}: Most common entry:", most_common_entry)
         score.publish("T12,123456," + str(clue_values[case_name]['value']) + "," + most_common_entry)

         clue_values[case_name]['submitted'] = True

   cmd.publish(move)
   rate.sleep()
