import cv2 as cv
import numpy as np 

def within_threshold(distance, target, threshold) -> bool:
  return distance >= target - threshold and distance <= target + threshold

def is_kinda_vertical(line): 
   x1, y1, x2, y2 = line
   return abs(y2 - y1) > abs(x2 - x1)

def write_image(image, name): 
   cv.imwrite(f"../data/testing/{name}.jpeg", image)

def draw_points(image, point_list): 
   for point in point_list: 
      cv.circle(image, (int(point[0]), int(point[1])), 25, (0, 0, 255), -1) 

def distance_to(point_1, point_2): 
  return np.hypot(abs(point_1[0] - point_2[0]), abs(point_1[1] - point_2[1]))