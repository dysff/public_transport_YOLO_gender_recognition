from ultralytics import YOLO
from deep_sort_pytorch.deep_sort import DeepSort
import cv2
import torch

model = YOLO('yolov8n.pt')
model.conf = 0.5
VIDEO_PATH = 'busfinal.mp4'

deepsort = DeepSort(
    "deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7",
    max_dist=0.7,
    min_confidence=0.3,
    nms_max_overlap=1.0,
    max_iou_distance=0.7,
    max_age=1000,
    n_init=3,
    nn_budget=100
)

cap = cv2.VideoCapture(VIDEO_PATH)

# def draw_UI(img):
  

while cap.isOpened():
  ret, frame = cap.read()
  if not ret:
    break

  results = model(frame, classes=[0])
  xywhs_boxes = []
  confs_boxes = []
  oids = []

  for result in results:
    for bbox in result.boxes:
      xywh = [int(i) for i in bbox.xywh[0]]
      conf = bbox.conf[0].item()
      cls = int(bbox.cls)

      xywhs_boxes.append(xywh)
      confs_boxes.append(conf)
      oids.append(cls)

  # Convert lists to tensors
  xywhs = torch.tensor(xywhs_boxes)
  confs = torch.tensor(confs_boxes)

  # Update DeepSORT with YOLO detections
  outputs = deepsort.update(xywhs, confs, oids, frame)
    
  for bbox_data in outputs[0]:  # Access the first array only
    x1, y1, x2, y2, _, track_id = bbox_data
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

  cv2.imshow("Frame", frame)
  if cv2.waitKey(1) & 0xFF == ord("q"):
      break

cap.release()
cv2.destroyAllWindows()