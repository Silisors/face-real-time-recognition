import cv2
import os
from deepface import DeepFace
from datetime import datetime

reference_images = [
    ('images/david1.jpg', 'David'),
    ('images/david2.jpg', 'David'),
    ('images/julian3.jpg', 'Persona 2'),
    ('images/julian4.jpg', 'Persona 2')
]

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

save_folder = 'detected_photos'
os.makedirs(save_folder, exist_ok=True)

cap = cv2.VideoCapture(0)
print("Encendiendo la cámara. Presiona 'q' para salir.")

frame_count = 0
process_every_n_frames = 60

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    height, width, _ = frame.shape
    roi_width, roi_height = width // 2, height // 2
    roi_x_start = (width - roi_width) // 2
    roi_y_start = (height - roi_height) // 2
    roi_x_end = roi_x_start + roi_width
    roi_y_end = roi_y_start + roi_height
    roi = frame[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

    if frame_count % process_every_n_frames == 0:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(roi, (x, y), (x+w, y+h), (0, 255, 0), 2)

            face_img = roi[y:y+h, x:x+w]
            verification_results = []

            for img_path, person_name in reference_images:
                try:
                    result = DeepFace.verify(face_img, img_path, enforce_detection=False)
                    verification_results.append((result, person_name))
                except Exception as e:
                    print("Error en la verificación en tiempo real:", str(e))

            verified = False
            for result, person_name in verification_results:
                if result["verified"]:
                    cv2.putText(roi, f"¡{person_name} eres tú!", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    print(f"¡{person_name} eres tú!")  

                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    photo_path = os.path.join(save_folder, f'{person_name}_{timestamp}.jpg')
                    cv2.imwrite(photo_path, frame)
                    print(f"Foto guardada en: {photo_path}")

                    verified = True
                    break

            if not verified:
                cv2.putText(roi, "No eres tú.", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.rectangle(frame, (roi_x_start, roi_y_start), (roi_x_end, roi_y_end), (255, 0, 0), 2)

    cv2.imshow('Cámara en tiempo real', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
