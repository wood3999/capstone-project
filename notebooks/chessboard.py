import sys
import numpy as np
from sklearn.cluster import DBSCAN

'''
Once I've processed a photo of a chessboard, it's likely that not all of the corners were picked up. So, this class 
contains the logic required to reconstruct a full grid of points that makes up an 8x8 chessboard, given a partial/incomplete set of points 

I am assuming the input contains at least one point from each of the board's edges. Otherwise it would be a guessing game 
(if I only have enough points to define a 8x7 grid, in my current implementation there's no way to know if the missing column would be on the left or right) 

'''

# testing data
MCD = 306.86818842858963
POINT_ARRAY = [[656.5, 1326.7777777777778],[357.7692307692308, 1007.2307692307693],[1266.0, 2244.0],[656.0, 711.8421052631579],[1570.5, 2244.0],[961.0, 2244.0],[965.5, 717.9473684210526],[347.53125, 2244.0],[964.5142857142857, 1326.4571428571428],[653.0, 2243.714285714286],[962.5, 1632.1666666666667],[962.0, 1937.5],[656.5, 1016.1],[1874.5, 2244.0],[965.5882352941177, 1020.7058823529412],[1271.575, 715.65],[2505.5666666666666, 2554.5],[1268.0, 1632.8333333333333],[655.5, 1937.5],[655.5, 1631.5],[1267.0, 1937.5],[353.53333333333336, 1937.6],[2488.5, 1635.1666666666667],[1571.0, 2554.5],[1571.5, 1633.3333333333333],[1875.5, 1634.0],[1571.0, 1937.5],[2797.0, 1937.5],[1266.3333333333333, 2554.5],[1270.7272727272727, 1023.4242424242424],[352.6666666666667, 1630.9444444444443],[1269.5, 1325.6666666666667],[352.0, 1327.3333333333333],[1572.5588235294117, 1326.1176470588234],[2496.3333333333335, 2244.0],[957.1, 2554.5],[1878.9, 2554.5],[1573.5263157894738, 1024.4736842105262],[2191.3, 2554.5],[645.8, 2554.5],[1571.5, 2866.875],[1876.4, 1023.4],[331.9, 2554.5],[2518.777777777778, 2875.1666666666665],[1883.7272727272727, 2868.0],[2198.5416666666665, 2866.9166666666665],[637.0909090909091, 2868.4545454545455],[655.5, 1631.5],[964.5142857142857, 1326.4571428571428],[656.5, 1016.1],[352.0, 1327.3333333333333],[350.2142857142857, 706.3333333333334],[1570.5, 2244.0],[1267.0, 1937.5],[1266.3333333333333, 2554.5],[961.0, 2244.0],[965.5, 717.9473684210526],[1571.0, 1937.5],[1571.0, 2554.5],[1874.5, 2244.0],[1266.0, 2244.0],[957.1, 2554.5],[653.0, 2243.714285714286],[962.0, 1937.5],[965.5882352941177, 1020.7058823529412],[1271.575, 715.65],[656.0, 711.8421052631579],[353.53333333333336, 1937.6],[331.9, 2554.5],[1269.5, 1325.6666666666667],[962.5, 1632.1666666666667],[656.5, 1326.7777777777778],[655.5, 1937.5],[347.53125, 2244.0],[645.8, 2554.5],[1268.0, 1632.8333333333333],[357.7692307692308, 1007.2307692307693],[1875.0, 1937.5],[1878.9, 2554.5],[2185.5, 2244.0],[1270.7272727272727, 1023.4242424242424],[1574.475, 726.225],[2496.3333333333335, 2244.0],[2191.3, 2554.5],[1571.5, 1633.3333333333333],[352.6666666666667, 1630.9444444444443],[2180.0, 1634.6666666666667],[2789.777777777778, 1635.8333333333333],[2491.75, 1937.5],[2485.6666666666665, 1327.25],[1571.5, 2866.875],[1572.5588235294117, 1326.1176470588234],[1875.5, 1634.0],[1875.5, 1327.25],[2808.6666666666665, 2244.0],[1264.0, 2866.75],[1573.5263157894738, 1024.4736842105262],[2505.5666666666666, 2554.5],[952.0, 2866.625],[1883.7272727272727, 2868.0],[1876.4, 1023.4],[2198.5416666666665, 2866.9166666666665],[637.0909090909091, 2868.4545454545455],[2180.5, 1024.4285714285713],[2529.25, 3190.875],[2830.5, 2878.25]] 

pts = np.asarray(POINT_ARRAY)
labels = DBSCAN(eps=20, min_samples=1).fit_predict(pts)
clustered_corners = np.array([pts[labels == k].mean(axis=0) for k in np.unique(labels)])

class Chessboard: 
  board = [[(0, 0) for _ in range(9)] for _ in range(9)]

  def __init__(self, most_common_dist):
    self.most_common_dist = most_common_dist


  def print(self): 
    for x in range(9):
      for y in range(9): 
        print("(", int(self.board[x][y][0]), ", ", int(self.board[x][y][1]), ")", sep="", end=" ")
      print()


  def visualize_board(self): 
    for x in range(9): 
      for y in range(9): 
        if self.board[x][y][0] == 0: 
          print("0", end="")
        else: 
          print("x", end="")
      print()


  def add_points(self, points):
    for point in points: 
      self.__insert_in_row(point, self.__find_row(point))

    
  def __find_row(self, point) -> int: 
    row_index = 0
    while row_index < 9: 
      if self.board[row_index][0][1] == 0:
        return row_index

      if abs(point[1] - self.board[row_index][0][1]) <= 0.4 * self.most_common_dist: 
        return row_index

      if point[1] < self.board[row_index][0][1]: 
        self.__row_juggle(row_index)
        return row_index

      row_index += 1
    return row_index - 1
        

  def __row_juggle(self, stop_index): 
    bottom = 8
    if self.board[bottom][0][0] != 0:
      print("Chessboard overflowing, input data must be cleaned more!")
      sys.exit()

    while bottom > stop_index: 
      self.board[bottom] = self.board[bottom - 1]
      bottom -= 1

    self.board[stop_index] = [(0,0) for _ in range(9)]


  def __insert_in_row(self, point, row_index): 
    column_index = 0
    while column_index < 9: 
      if self.board[row_index][column_index][0] == 0: 
        self.board[row_index][column_index] = point
        break

      if abs(point[0] - self.board[row_index][column_index][0]) <= 0.4 * self.most_common_dist: 
        self.print()
        print("Points too close together (potentially duplicates?)")
        print(point, self.board[row_index][column_index])
        sys.exit()
      else:
        continue

      if point[0] < self.board[row_index][column_index][0]: 
        temp = point
        point = self.board[row_index][column_index]
        self.board[row_index][column_index] = temp

      column_index += 1


  def imputate_points(self): 
    self.lowest_x, self.lowest_y = self.board[0][0]
    self.highest_y = 0
    self.highest_x = 0

    for row in range(9): 
      for column in range(9): 
        if self.board[row][column][0] != 0: 
          self.lowest_x = min(self.lowest_x, self.board[row][column][0])
          self.highest_x = max(self.highest_x, self.board[row][column][0])

    for row in range(9) :
      self.__imputate_row(row)


  def __imputate_row(self, row_index): 
    while self.board[row_index][0][0] - self.lowest_x > 0.4 * self.most_common_dist:
      if self.board[row_index][8][0] != 0: 
        print("Row", row_index, "already full! Clean input data")
        sys.exit()
      # shifting existing points to the right 
      for x in range (8): 
        self.board[row_index][8 - x] = self.board[row_index][8 - x - 1]

      # creating new point in slot 0. X value is slot 1's x - most_common_dist. Y value is slot 1's Y + the difference between slot 1 and slot 2's Y values (to follow the slope)  
      self.board[row_index][0][0] = self.board[row_index][1][0] - self.most_common_dist
      self.board[row_index][0][1] = self.board[row_index][1][1] + (self.board[row_index][1][1] - self.board[row_index][2][1])

    last_filled_spot = 0
    for x in range(9):
      if self.board[row_index][x][0] != 0: 
        last_filled_spot = x

    while last_filled_spot < 8: 
      # same logic as previous imputation
      x = self.board[row_index][last_filled_spot][0] + self.most_common_dist
      y = self.board[row_index][last_filled_spot][1] + (self.board[row_index][last_filled_spot][1] - self.board[row_index][last_filled_spot - 1][1])
      self.board[row_index][last_filled_spot + 1] = (x, y)
      last_filled_spot += 1


  def get_corners(self): 
    return [self.board[0][0], self.board[0][8], self.board[8][8], self.board[8][0]]


test = Chessboard(MCD)
test.add_points(clustered_corners)
test.imputate_points()
test.visualize_board()
print(test.get_corners())