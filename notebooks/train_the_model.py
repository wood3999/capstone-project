from ultralytics import YOLO
from pathlib import Path
import sys


PARENT_DIR = Path(__file__).resolve().parent.parent

if len(sys.argv) < 1: 
  print("Enter the training data directory name as a command line argument (directory that contains data.yaml)")
  print("Ex: FirstDataset.yolo26")
  sys.exit()

model = YOLO("yolo26n.pt")
try: 
  model.train(data=f"{PARENT_DIR}/data/{sys.argv[1]}/data.yaml", epochs=100, imgsz=1280, batch=4, device="mps", workers=0)
except: 
  print(f"{PARENT_DIR}/data/{sys.argv[1]}/data.yaml not found or setup incorrectly")