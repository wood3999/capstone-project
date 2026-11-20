import sys
import cv2 as cv
import numpy as np
from sklearn.cluster import DBSCAN
from utils import draw_points, write_image, distance_to, within_threshold

'''
Once I've processed a photo of a chessboard, it's likely that not all of the corners were picked up. So, this class 
contains the logic required to reconstruct a full grid of points that makes up an 8x8 chessboard, given a partial/incomplete set of points 

I am assuming the input contains at least one point from each of the board's edges. Otherwise it would be a guessing game 
(if I only have enough points to define a 8x7 grid, in my current implementation there's no way to know if the missing column/row would be on the left or right) 

'''

# testing data
# MCD = 306.86818842858963
# POINT_ARRAY = [[656.5, 1326.7777777777778],[357.7692307692308, 1007.2307692307693],[1266.0, 2244.0],[656.0, 711.8421052631579],[1570.5, 2244.0],[961.0, 2244.0],[965.5, 717.9473684210526],[347.53125, 2244.0],[964.5142857142857, 1326.4571428571428],[653.0, 2243.714285714286],[962.5, 1632.1666666666667],[962.0, 1937.5],[656.5, 1016.1],[1874.5, 2244.0],[965.5882352941177, 1020.7058823529412],[1271.575, 715.65],[2505.5666666666666, 2554.5],[1268.0, 1632.8333333333333],[655.5, 1937.5],[655.5, 1631.5],[1267.0, 1937.5],[353.53333333333336, 1937.6],[2488.5, 1635.1666666666667],[1571.0, 2554.5],[1571.5, 1633.3333333333333],[1875.5, 1634.0],[1571.0, 1937.5],[2797.0, 1937.5],[1266.3333333333333, 2554.5],[1270.7272727272727, 1023.4242424242424],[352.6666666666667, 1630.9444444444443],[1269.5, 1325.6666666666667],[352.0, 1327.3333333333333],[1572.5588235294117, 1326.1176470588234],[2496.3333333333335, 2244.0],[957.1, 2554.5],[1878.9, 2554.5],[1573.5263157894738, 1024.4736842105262],[2191.3, 2554.5],[645.8, 2554.5],[1571.5, 2866.875],[1876.4, 1023.4],[331.9, 2554.5],[2518.777777777778, 2875.1666666666665],[1883.7272727272727, 2868.0],[2198.5416666666665, 2866.9166666666665],[637.0909090909091, 2868.4545454545455],[655.5, 1631.5],[964.5142857142857, 1326.4571428571428],[656.5, 1016.1],[352.0, 1327.3333333333333],[350.2142857142857, 706.3333333333334],[1570.5, 2244.0],[1267.0, 1937.5],[1266.3333333333333, 2554.5],[961.0, 2244.0],[965.5, 717.9473684210526],[1571.0, 1937.5],[1571.0, 2554.5],[1874.5, 2244.0],[1266.0, 2244.0],[957.1, 2554.5],[653.0, 2243.714285714286],[962.0, 1937.5],[965.5882352941177, 1020.7058823529412],[1271.575, 715.65],[656.0, 711.8421052631579],[353.53333333333336, 1937.6],[331.9, 2554.5],[1269.5, 1325.6666666666667],[962.5, 1632.1666666666667],[656.5, 1326.7777777777778],[655.5, 1937.5],[347.53125, 2244.0],[645.8, 2554.5],[1268.0, 1632.8333333333333],[357.7692307692308, 1007.2307692307693],[1875.0, 1937.5],[1878.9, 2554.5],[2185.5, 2244.0],[1270.7272727272727, 1023.4242424242424],[1574.475, 726.225],[2496.3333333333335, 2244.0],[2191.3, 2554.5],[1571.5, 1633.3333333333333],[352.6666666666667, 1630.9444444444443],[2180.0, 1634.6666666666667],[2789.777777777778, 1635.8333333333333],[2491.75, 1937.5],[2485.6666666666665, 1327.25],[1571.5, 2866.875],[1572.5588235294117, 1326.1176470588234],[1875.5, 1634.0],[1875.5, 1327.25],[2808.6666666666665, 2244.0],[1264.0, 2866.75],[1573.5263157894738, 1024.4736842105262],[2505.5666666666666, 2554.5],[952.0, 2866.625],[1883.7272727272727, 2868.0],[1876.4, 1023.4],[2198.5416666666665, 2866.9166666666665],[637.0909090909091, 2868.4545454545455],[2180.5, 1024.4285714285713],[2529.25, 3190.875],[2830.5, 2878.25]] 

# pts = np.asarray(POINT_ARRAY)
# labels = DBSCAN(eps=20, min_samples=1).fit_predict(pts)
# clustered_corners = np.array([pts[labels == k].mean(axis=0) for k in np.unique(labels)])

class Chessboard: 

  def __init__(self, most_common_dist):
    self.most_common_dist = most_common_dist
    self.board = [[(None, None) for _ in range(9)] for _ in range(9)]


  def print(self): 
    for x in range(9):
      for y in range(9): 
        print("(", int(self.board[x][y][0]), ", ", int(self.board[x][y][1]), ")", sep="", end=" ")
      print()


  def visualize_board(self): 
    for x in range(9): 
      for y in range(9): 
        if self.board[x][y][0] == None: 
          print("0", end="")
        else: 
          print("x", end="")
      print()


  def add_points(self, points):
    lowest_point = points[0]
    index = 0
    for x, point in enumerate(points):
      if lowest_point[1] < point[1]: 
        lowest_point = point
        index = x

    self.__insert_in_row(lowest_point, 0)
    points.pop(index)
      
    for point in points: 
      self.__insert_in_row(point, self.__find_row(point))

    
  def __find_row(self, point) -> int: 
    row_index = 0
    while row_index < 9: 
      if self.board[row_index][0][1] == None:
        return row_index

      # taking the y value of the potential point and comparing it to the y value of the first point in the row
      # if within 0.4 * most_common_dist of the row's y, then it belongs to the row. 
      # if abs(point[1] - self.board[row_index][0][1]) <= 0.4 * self.most_common_dist: 
      #   return row_index

      # Trying to get a more accurate y estimation for tilted photos by accounting for slope
      # Approach A: compare to the average Y value of the row, not the first Y value
      # Approach B (implemented): Find the slope of any horizontal line, and guess where the Y value should land based on how far away the x value is 
      if within_threshold(point[1], self.__guess_next_points_y(row_index, point), 0.4 * self.most_common_dist): 
        return row_index

      if point[1] < self.board[row_index][0][1]: 
        self.__row_juggle(row_index)
        return row_index

      row_index += 1
    return row_index - 1

  # guess the next point's placement based on what we know so far
  def __guess_next_points_y(self, row_index, point) -> float: 
    dist = None
   
    for x in range(9): 
      if self.board[row_index][x][0] != None: 
        temp_dist = abs(point[0] - self.board[row_index][x][0])
        if dist == None or temp_dist < dist: 
          closest_point_index = x
          dist = temp_dist
          if point[1] < self.board[row_index][x][1]: 
            left = -1
          else: 
            left = 1
        
    # makes assumptions that are only true when all of the points are filled in. Do not use 
    # self.__find_avg_displacements()
    est_y_disp = self.__guess_y_disp()
    est_num_spots_over = round(dist / self.most_common_dist)
    return self.board[row_index][closest_point_index][1] + (est_num_spots_over * est_y_disp * left)

  def __guess_y_disp(self) -> float: 
    # first find the row with the most entries so far
    most_entries = None 
    most_entries_row = None
    for row in range(9): 
      curr_entries = 0
      for column in range(9): 
        if self.board[row][column][0] != None: 
          curr_entries += 1
      if most_entries == None or curr_entries > most_entries: 
        most_entries = curr_entries
        most_entries_row = row

    displacement = []
    for x in range(1, 9): 
      if self.board[most_entries_row][x-1][0] != None and self.board[most_entries_row][x][0] != None:
        x_dist = self.board[most_entries_row][x - 1][0] - self.board[most_entries_row][x][0]
        spots_over = round(x_dist / self.most_common_dist) 
        y_disp = self.board[most_entries_row][x-1][1] - self.board[most_entries_row][x][1] 
        displacement.append(abs(y_disp / spots_over))

    if len(displacement) == 0: 
      return 0
    else: 
      return sum(displacement) / len(displacement)


  def __row_juggle(self, stop_index): 
    bottom = 8
    if self.board[bottom][0][0] != None:
      print("Chessboard overflowing, input data must be cleaned more!")
      raise RuntimeError("Trying to add a ninth row")

    while bottom > stop_index: 
      self.board[bottom] = self.board[bottom - 1]
      bottom -= 1

    self.board[stop_index] = [(None, None) for _ in range(9)]


  def __insert_in_row(self, point, row_index): 
    column_index = 0
    while column_index < 9: 
      if self.board[row_index][column_index][0] == None: 
        self.board[row_index][column_index] = point
        break

      if abs(point[0] - self.board[row_index][column_index][0]) <= 0.4 * self.most_common_dist: 
        if int(point[0]) == int(self.board[row_index][column_index][0]) and int(point[1]) == int(self.board[row_index][column_index][1]): 
          break
        else:
          self.visualize_board()
          print("Points too close together (potentially duplicates?)")
          print(point, self.board[row_index][column_index])
          raise RuntimeError("Points too close together")
          # sys.exit()
   
      if point[0] < self.board[row_index][column_index][0]: 
        temp = point
        point = self.board[row_index][column_index]
        self.board[row_index][column_index] = temp

      column_index += 1


  def impute_points(self): 
    self.lowest_x, self.lowest_y = self.board[0][0]
    self.highest_y = 0
    self.highest_x = 0

    for row in range(9): 
      for column in range(9): 
        if self.board[row][column][0] != None: 
          self.lowest_x = min(self.lowest_x, self.board[row][column][0])
          self.highest_x = max(self.highest_x, self.board[row][column][0])

    for row in range(9) :
      self.__shift_to_proper_spots(row)

    self.__find_avg_displacements()

    for row in range(9): 
      for column in range(9): 
        if self.board[row][column][0] == None: 
          self.__impute_point(row, column)

    # The old way of doing it: 
    # for row in range(9): 
    #   self.__impute_row(row) 

  
  def __impute_point(self, row, column): 
    nearest_col_index = self.__find_closest_horizontal(row, column) 
    nearest_row_index = self.__find_closest_vertical(row, column)

    if nearest_col_index == None and nearest_row_index == None: 
      self.__impute_row(row) 
    else: 
      if nearest_col_index is None or abs(nearest_row_index - row) <= abs(nearest_col_index - column): 
        x = self.board[nearest_row_index][column][0] + (self.avg_x_disp * (row - nearest_row_index))
        y = self.board[nearest_row_index][column][1] + (self.most_common_dist * (row - nearest_row_index)) + (self.avg_y_disp * (row - nearest_row_index))
      else: 
        x = self.board[row][nearest_col_index][0] + (self.most_common_dist * (column - nearest_col_index)) + (self.avg_x_disp * (column - nearest_col_index))
        y = self.board[row][nearest_col_index][1] + (self.avg_y_disp * (column - nearest_col_index))
      self.board[row][column] = (x,y) 


  def __find_closest_vertical(self, row, column) -> int: 
    up = row - 1
    down = row + 1
    while True:
      if up >= 0: 
        if self.board[up][column][0] != None: 
          row_index = up 
          break
      if down < 9: 
        if self.board[down][column][0] != None: 
          row_index = down
          break
      down += 1
      up -= 1
      if up < 0 and down > 8: 
        row_index = None 
        break 
    return row_index


  def __find_closest_horizontal(self, row, column) -> int: 
    left = column - 1
    right = column + 1
    while True:
      if left >= 0: 
        if self.board[row][left][0] != None: 
          col_index = left 
          break
      if right < 9: 
        if self.board[row][right][0] != None: 
          col_index = right
          break
      right += 1
      left -= 1
      if left < 0 and right > 8: 
        col_index = None 
        break 
    return col_index
 

  def __shift_to_proper_spots(self, row_index): 
    starting_index = 0 
    first = 0

    while starting_index < 8: 
      times_run = 0
      if first == 0: 
        distance = self.board[row_index][starting_index][0] - self.lowest_x
      elif self.board[row_index][starting_index][0] == None or self.board[row_index][starting_index + 1][0] == None: 
        distance = 0
      else: 
        distance = self.board[row_index][starting_index + 1][0] - self.board[row_index][starting_index][0]

      while distance - self.most_common_dist * (times_run + first) > 0.4 * self.most_common_dist: 
        # If can't shift right w/o losing a point 
        if self.board[row_index][8][0] != None: 
          print("Too many points in row:", row_index)
          raise RuntimeError("Error")
          # sys.exit()

        # shifting everything to the right
        for x in range (8 - starting_index):
          self.board[row_index][8 - x] = self.board[row_index][8 - x - 1]

        times_run += 1
        self.board[row_index][starting_index + (times_run * first)] = (None, None)

      starting_index += 1
      if first == 0: 
        starting_index = 0 
      first = 1

  def __find_avg_displacements(self): 
    col_with_most_entries = None
    row_with_most_entries = None
    column_index = None
    row_index = None

    # find column with most filled in values 
    for y in range(9): 
      current_col_existing = 0
      current_row_existing = 0
      for x in range(9): 
        if self.board[x][y][0] != None:
          current_col_existing += 1
        if self.board[y][x][1] != None: 
          current_row_existing += 1

      if col_with_most_entries == None or current_col_existing > col_with_most_entries: 
        col_with_most_entries = current_col_existing
        column_index = y

      if row_with_most_entries == None or current_row_existing > row_with_most_entries: 
        row_with_most_entries = current_row_existing
        row_index = x

    x_disp = []
    y_disp = []
    for x in range(1, 9): 
      if self.board[x][column_index][0] != None and self.board[x - 1][column_index][0] != None:
        x_disp.append(self.board[x][column_index][0] - self.board[x-1][column_index][0]) 
      if self.board[row_index][x][1] != None and self.board[row_index][x-1][1] != None: 
        y_disp.append(self.board[row_index][x][1] - self.board[row_index][x-1][1])

    if len(x_disp) == 0: 
      self.avg_x_disp = 0
    else: 
      self.avg_x_disp = sum(x_disp) / len(x_disp)

    if len(y_disp) == 0: 
      self.avg_y_disp = 0
    else: 
      self.avg_y_disp = sum(y_disp) / len(y_disp)

# i do this row by row. Instead I could do this point by point, picking that point's nearest neighbor to impute it 
  def __impute_row(self, row_index): 
    for x in range(9): 
      if self.board[row_index][x][0] != None:
        first_filled_spot = x
        break

    while first_filled_spot > 0: 
      x = self.board[row_index][first_filled_spot][0] - self.most_common_dist + self.avg_x_disp
      y = self.board[row_index][first_filled_spot][1] + self.avg_y_disp  
      self.board[row_index][first_filled_spot - 1] = (x, y)
      first_filled_spot -= 1

    for index in range(8): 
      if self.board[row_index][index + 1][0] == None: 
        x = self.board[row_index][index][0] + self.most_common_dist + self.avg_x_disp
        y = self.board[row_index][index][1] + self.avg_y_disp
        self.board[row_index][index + 1] = (x,y)


  def get_corners(self): 
    return [self.board[0][0], self.board[0][8], self.board[8][8], self.board[8][0]]


  def get_points(self): 
    points = [] 
    for row in range(9): 
      for column in range(9): 
        if self.board[row][column][0] != None:
          points.append(self.board[row][column])

    return points


# image = cv.imread("/Users/tatewoodward/Coding/Projects/capstone-project/data/testing/board_images/empty_board.jpeg")
# test = Chessboard(MCD)
# test.add_points(clustered_corners)
# test.impute_points()
# test.visualize_board()

# draw_points(image, test.get_points())
# write_image(image, "impute")

