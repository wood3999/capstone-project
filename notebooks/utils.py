import cv2 as cv

def find_line_intersection(line1, line2):
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
     return (0,0)

  # intersection
  x = (B2 * C1 - B1 * C2) / det
  y = (A1 * C2 - A2 * C1) / det

  buffer = 10
  lowest_x = min(x1, x2)
  lowest_x = min(x, lowest_x - buffer)
  highest_x = max(x1, x2)
  highest_x = max(x, highest_x + buffer)
  if (x == highest_x or x == lowest_x):
     return (0,0)

  lowesty = min(y1, y2)
  lowesty = min(y, lowesty - buffer)
  highesty = max(y1, y2)
  highesty = max(y, highesty + buffer)
  if (y == highesty or y == lowesty):
     return (0,0)

  return (x, y)

def is_kinda_vertical(line): 
   x1, y1, x2, y2 = line
   return abs(y2 - y1) > abs(x2 - x1)

def write_image(image, name): 
   cv.imwrite(f"../data/testing/{name}.jpeg", image)

def draw_points(image, point_list): 
   for point in point_list: 
      cv.circle(image, (int(point[0]), int(point[1])), 15, (0, 0, 255), -1) 