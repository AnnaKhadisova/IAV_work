from ultralytics import YOLO
import os

model = YOLO("yolov8n.pt") 


input_video = "proba.mp4" 
script_dir = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(script_dir, input_video)


results = model.predict(
    source=video_path,
    save=True,
    save_txt=False,
    project=script_dir,  
    name="yolo_output"  
)


