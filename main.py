from flask import Flask, Response, render_template
from ultralytics import YOLO
import cv2

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('video.html')

def detect_objects():
    model = YOLO('models/best.pt')
    cap = cv2.VideoCapture(0)

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            frame = cv2.flip(frame, 1)
            results = model(frame)

            pred_boxes = results[0].boxes.xyxy.cpu().numpy()
            pred_scores = results[0].boxes.conf.cpu().numpy()
            pred_classes = results[0].boxes.cls.cpu().numpy()

            for i, box in enumerate(pred_boxes):
                x1, y1, x2, y2 = map(int, box)
                label = f'{model.names[int(pred_classes[i])]} {pred_scores[i]:.2f}'
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)

            if not ret:
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    finally:
        cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(detect_objects(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
