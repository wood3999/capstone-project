import cv2 as cv
import sys
from pathlib import Path

'''
When I download my datasets from Roboflow, all of the images are rotated 90 degrees counterclockwise. 
Pretty sure it has something to do with the fact that iPhones store photos rotated 90 degrees, and
for some reason downloading the dataset causes the photos to get stuck like that.

So now I run this script whenever I download my data from Roboflow
'''

PARENT_DIR = Path(__file__).resolve().parent.parent
data_types = ["test", "train", "valid"]
rotated = False

if len(sys.argv) > 1: 
  directory_name = sys.argv[1]
else:
  print("Enter a directory name as a command line argument")
  sys.exit()


with open(f"{PARENT_DIR}/data/rotation_history.txt", "r", encoding="utf-8") as file: 
  for line in file: 
    if line.strip()[0] == '#': 
      continue
    elif line.strip() == sys.argv[1]: 
      rotated = True

# Photos that have gone through the preprocessing pipeline do not need to be rotated!
if directory_name.contains("Cropped"): 
  rotated = True

if not rotated: 
  print("Rotating training images")
  for data_type in data_types: 
    file_names = []

    try: 
      for file in Path(f"{PARENT_DIR}/data/{directory_name}/{data_type}/images").iterdir(): 
        if (file.name.endswith(".jpg")):
          file_names.append(file.name)
    except FileNotFoundError: 
      print(f"Directory not found: {PARENT_DIR}/data/{directory_name}/{data_type}/images") 
      sys.exit()

    for file in file_names: 
      image = cv.imread(f"{PARENT_DIR}/data/{directory_name}/{data_type}/images/{file}")
      image = cv.rotate(image, cv.ROTATE_90_CLOCKWISE)
      cv.imwrite(f"{PARENT_DIR}/data/{directory_name}/{data_type}/images/{file}", image)
      # print(f"Wrote: {PARENT_DIR}/data/{directory_name}/{data_type}/images/{file}")
  with open(f"{PARENT_DIR}/data/rotation_history.txt", "a") as file: 
    file.write(f"{sys.argv[1]}\n")
else: 
  print("Training images already rotated!")
