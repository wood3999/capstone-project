import cv2 as cv
import sys
from pathlib import Path

PARENT_DIR = Path(__file__).resolve().parent.parent
data_types = ["test", "train", "valid"]

if len(sys.argv) > 1: 
  directory_name = sys.argv[1]
else:
  print("Enter a directory name as a command line argument")
  sys.exit()

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
