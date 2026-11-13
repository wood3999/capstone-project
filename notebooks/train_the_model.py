from ultralytics import YOLO
from pathlib import Path
import sys
import subprocess

PARENT_DIR = Path(__file__).resolve().parent.parent

if len(sys.argv) < 2: 
  print("Enter the training data directory name as a command line argument (directory that contains data.yaml)")
  print("Ex: FirstDataset.yolo26")
  sys.exit()
elif len(sys.argv) < 3: 
  print("Second command line argument should be the name of the output directory")
  print("Ex: 'ExampleModel' would be located in /models/ExampleModel")

# I setup this script to run my rotate script first, so I don't have to worry about doing it manually 
rotating_images = subprocess.run([sys.executable, f"{PARENT_DIR}/notebooks/rotate_images.py", sys.argv[1]], check=True, capture_output=True, text=True)
print(rotating_images.stdout)

model = YOLO("yolo26n.pt")
try: 
  model.train(data=f"{PARENT_DIR}/data/{sys.argv[1]}/data.yaml",
              epochs=200, 
              imgsz=1280, 
              batch=8, 
              device="mps", 
              workers=0,
              hsv_h=0, 
              hsv_s=0.1, 
              hsv_v=0.1, 
              translate=0,
              scale=0.1, 
              flipud=0.75, 
              fliplr=0.75,
              save_dir=f"{PARENT_DIR}/models/{sys.argv[2]}"
            )
except: 
  print(f"{PARENT_DIR}/data/{sys.argv[1]}/data.yaml not found or setup incorrectly")