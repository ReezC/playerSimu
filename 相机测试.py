from ultralytics import YOLO
import cv2

model = YOLO("yolo26n.pt")
results = model(
    source=0,
    stream=True,
)

if __name__ == "__main__":
    for result in results:
        plotted = result.plot() # type: ignore
        cv2.imshow("YOLOv8 Detection", plotted)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break