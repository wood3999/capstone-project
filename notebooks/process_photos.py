import cv2 as cv 
import numpy as np 
from sklearn.cluster import DBSCAN
from utils import draw_points, write_image, is_kinda_vertical, distance_to, within_threshold
from chessboard import Chessboard
import pandas as pd
from pathlib import Path
import sys

# How many pixels away a point can be to still be considered horizontal or vertical to another point
VERTICAL_HORIZONTAL_THRESHOLD = 25 

# How big the window is when I find most_common_dist (sliding window through a list of distances)
DISTANCE_BETWEEN_POINTS_THRESHOLD = 30

# How far away from most_common_dist two points can be to still be considered a square length away
DISTANCE_THRESHOLD = 15

# How far away from most_common_dist the distance between two points can be 
CLOSENESS_TO_VERIFIED_POINT = 30

# What a point must score in order to be "verified". 
# A point's score is the distance to the four nearest neighbors, - most_common_dist, averaged 
SCORE_THRESHOLD = 30

# What a point needs to score according to my scoring function that verifies it as on the chessboard
INITIAL_VERIFICATION_SCORE_REQUIREMENT = 80

def main(): 
  subdirectories = ["train", "test", "valid"]

  if len(sys.argv) < 2: 
    print("Debugging - Enter the directory containing the photos as the command line argument to run on a dataset")
    image = cv.imread("/Users/tatewoodward/Coding/Projects/capstone-project/data/testing/wood_background/IMG_6400.jpeg")
    process_photo(image, file_name="test")
    sys.exit()

  if sys.argv[1].endswith(".yolo26"): 
    for directory in subdirectories: 
      for file in Path(f"../data/{sys.argv[1]}/{directory}/images").iterdir(): 
        print(f"{file.name}")
        if (file.name.endswith(".jpg")):
          image = cv.imread(f"../data/{sys.argv[1]}/{directory}/images/{file.name}")
          process_photo(image, file_name=file.name)
  else: 
    for file in Path(f"../data/testing/{sys.argv[1]}").iterdir():
      print(f"{file.name}")
      if file.name.endswith(".jpg") or file.name.endswith(".jpeg"):
        image = cv.imread(f"../data/testing/{sys.argv[1]}/{file.name}")
        process_photo(image, file_name=file.name)
  return None


def process_photo(image, file_name): 
  kernel_size = 9
  grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
  blur = cv.GaussianBlur(grayscale, (kernel_size, kernel_size), 3)
  edges = cv.Canny(blur, 50, 100)
  lines = cv.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=20, maxLineGap=150)

  clustered_corners = find_points_from_lines(lines)
  most_common_dist = find_most_common_dist(clustered_corners) 
  filtered_corners = []

  # temp = image.copy()
  # draw_points(temp, clustered_corners)
  # cv.line(temp, (int(clustered_corners[15][0]), int(clustered_corners[15][1])), (int(clustered_corners[15][0] + most_common_dist), 
                                                                                #  int(clustered_corners[15][1])), (255, 0, 0), 10)
  # write_image(temp, f"clustered-{file_name}")


  # filtering the corners by how many good neighbors they have 
  # adds a point to filtered_corners if at least two neighbors are within most_common_distance +/- the buffer 
  for point_1 in clustered_corners:
    num_times = 0
    for point_2 in clustered_corners: 
      if horizontal_or_vertical(point_1, point_2):
        if within_threshold(distance_to(point_1, point_2), most_common_dist, DISTANCE_THRESHOLD): 
          num_times += 1
          if num_times == 2: 
            filtered_corners.append(point_1)

  # filtered = image.copy()
  # draw_points(filtered, filtered_corners)
  # write_image(filtered, "filtered")

  verified_points = []

  for x, point_1 in enumerate(filtered_corners): 
    score = calculate_score(point_1, filtered_corners, most_common_dist, avg=True)
    if score < INITIAL_VERIFICATION_SCORE_REQUIREMENT:
      verified_points.append(point_1) 

  length = 0

  while len(verified_points) != length: 
    length = len(verified_points)
    verified_points = verify_neighbors(verified_points, clustered_corners, most_common_dist)

  temp = image.copy()
  # draw_points(temp, verified_points)
  # write_image(temp, f"verified_{file_name}")

  test = Chessboard(most_common_dist)
  try: 
    test.add_points(verified_points)
    test.impute_points()
  except RuntimeError as err: 
    print(err, file_name)
    return None

  draw_points(image, test.get_corners())
  write_image(image, f"corners-{file_name}")

def verify_neighbors(verified_points, unverified_points, most_common_dist): 
  verified_and_buddies = verified_points.copy()
  for point_1 in verified_points:
    for point_2 in unverified_points:
      if within_threshold(distance_to(point_1, point_2), most_common_dist, CLOSENESS_TO_VERIFIED_POINT) and horizontal_or_vertical(point_1, point_2): 
        skip = False
        for point_3_index, point_3 in enumerate(verified_and_buddies):
          if distance_to(point_2, point_3) <= 0.4 * most_common_dist: 
            if calculate_score(point_2, verified_and_buddies, most_common_dist) < calculate_score(point_3, verified_and_buddies, most_common_dist):
              # replacing the current verified point with the lower score point
              verified_and_buddies[point_3_index] = point_2
            skip = True
        if not skip: 
          verified_and_buddies.append(point_2)

  return verified_and_buddies

def horizontal_or_vertical(point_1, point_2) -> bool: 
  return (abs(point_1[0] - point_2[0]) < VERTICAL_HORIZONTAL_THRESHOLD or abs(point_1[1] - point_2[1]) < VERTICAL_HORIZONTAL_THRESHOLD)

def find_most_common_dist(point_list) -> float: 
  distance = []
  for point_1 in point_list: 
    for point_2 in point_list:
      # filter to only check points that are more or less vertical / horizontal
      if horizontal_or_vertical(point_1, point_2): 
        if (point_1[0] != point_2[0] and point_1[1] != point_2[1]):
          distance.append(distance_to(point_1, point_2))
        
  distance = pd.DataFrame({"distance": distance})
  distance = distance['distance'].sort_values().reset_index(drop=True)

  left = 0
  right = 0 
  max_difference = 0

  # finds the most common difference between points by using a sliding window through distance, (distance_between_points defines the window) 
  # where distance is a list of the distance to the closest 4 neighboring points (for each point). len(distance) = n * 4, where n = # of points 
  while (left < len(distance) and right < len(distance)): 
    if (distance[left] + DISTANCE_BETWEEN_POINTS_THRESHOLD > distance[right]): 
      right += 1 
    else: 
      left += 1
    if right - left > max_difference: 
      max_difference = right - left 
      most_common_dist = distance[right - (max_difference // 2)]
  return most_common_dist

def find_line_intersection(line1, line2) -> tuple[float, float]:
  x1, y1, x2, y2 = map(float, line1)
  x3, y3, x4, y4 = map(float, line2)

    # Ax + By = C
  A1 = y2 - y1
  B1 = x1 - x2
  C1 = A1 * x1 + B1 * y1

  A2 = y4 - y3
  B2 = x3 - x4
  C2 = A2 * x3 + B2 * y3

  # determinant
  det = A1 * B2 - A2 * B1

  if det == 0: 
     return (None, None)

  # intersection
  x = (B2 * C1 - B1 * C2) / det
  y = (A1 * C2 - A2 * C1) / det

  buffer = 10
  lowest_x = min(x1, x2)
  lowest_x = min(x, lowest_x - buffer)
  highest_x = max(x1, x2)
  highest_x = max(x, highest_x + buffer)
  if (x == highest_x or x == lowest_x):
     return (None, None)

  lowest_y = min(y1, y2)
  lowest_y = min(y, lowest_y - buffer)
  highest_y = max(y1, y2)
  highest_y = max(y, highest_y + buffer)
  if (y == highest_y or y == lowest_y):
     return (None, None)

  return (x, y)

def find_points_from_lines(hough_lines): 
  horizontal_lines = []
  vertical_lines = []
  corners = []
  if hough_lines is not None: 
    for line in hough_lines: 
        if is_kinda_vertical(line): 
          vertical_lines.append(line)
        else: 
          horizontal_lines.append(line)

  for line in vertical_lines:
    for line2 in horizontal_lines: 
        x, y = find_line_intersection(line, line2)
        if x != None and y!= None:
          corners.append((round(x), round(y))) 

  pts = np.asarray(corners)
  labels = DBSCAN(eps=20, min_samples=3).fit_predict(pts)
  return [tuple(pts[labels == k].mean(axis=0).tolist()) for k in np.unique(labels)]


def calculate_score(point, verified_points, most_common_dist, avg=False) -> float:
  score = []
  for verified in verified_points: 
    if distance_to(point, verified) <= most_common_dist + DISTANCE_THRESHOLD and horizontal_or_vertical(point, verified):
      score.append(abs(distance_to(point, verified) - most_common_dist))
  if avg: 
    return sum(score) / len(score)
  else: 
    return sum(score)

def point_by_point(image, points): 
  for x, point in enumerate(points): 
    new_image = image.copy()
    cv.circle(new_image, (int(point[0]), int(point[1])), 20, (0, 0, 255), -1) 
    cv.imwrite(f"/Users/tatewoodward/Coding/Projects/capstone-project/data/testing/{x}.jpeg", new_image)

if __name__ == "__main__": 
  main()
