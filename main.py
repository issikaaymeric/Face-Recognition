import threading
import cv2
from deepface import DeepFace

# Start video capture
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Load reference image
reference_img = cv2.imread("elon.jpg")

# Shared variables
face_match = False
lock = threading.Lock()
counter = 0

def check_face(frame):
    global face_match
    try:
        result = DeepFace.verify(frame, reference_img.copy(), enforce_detection=False)
        with lock:
            face_match = result['verified']
    except Exception as e:
        with lock:
            face_match = False
        print(f"Error verifying face: {e}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Check every 30 frames
    if counter % 30 == 0:
        threading.Thread(target=check_face, args=(frame.copy(),)).start()

    # Display match result
    with lock:
        if face_match:
            text = "MATCH!"
            color = (0, 255, 0)
        else:
            text = "NO MATCH!"
            color = (0, 0, 255)

    cv2.putText(frame, text, (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)
    cv2.imshow("Live Face Verification", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

    counter += 1

cap.release()
cv2.destroyAllWindows()
