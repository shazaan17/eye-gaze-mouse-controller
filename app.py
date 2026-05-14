from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import pyautogui
import time

app = Flask(__name__)

camera = cv2.VideoCapture(0)

face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

screen_w, screen_h = pyautogui.size()


def generate_frames():
    while True:
        success, frame = camera.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        output = face_mesh.process(rgb_frame)

        landmark_points = output.multi_face_landmarks

        frame_h, frame_w, _ = frame.shape

        if landmark_points:
            landmarks = landmark_points[0].landmark

            # Right eye tracking
            for id, landmark in enumerate(landmarks[474:478]):

                x = int(landmark.x * frame_w)
                y = int(landmark.y * frame_h)

                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

                if id == 1:
                    screen_x = screen_w / frame_w * x
                    screen_y = screen_h / frame_h * y

                    pyautogui.moveTo(screen_x, screen_y)

            # Blink detection
            left = [landmarks[145], landmarks[159]]

            for landmark in left:
                x = int(landmark.x * frame_w)
                y = int(landmark.y * frame_h)

                cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)

            if (left[0].y - left[1].y) < 0.004:
                pyautogui.click()
                time.sleep(1)

        ret, buffer = cv2.imencode('.jpg', frame)

        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)