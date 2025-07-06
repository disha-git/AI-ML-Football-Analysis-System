
from ultralytics import YOLO 

model = YOLO('D:/NitA/project/models/best.pt')

results = model.predict(source= r"D:/NitA/project/input video/08fd33_4.mp4", save=True)
print(results[0])
print('=====================================')
for box in results[0].boxes:
    print(box)