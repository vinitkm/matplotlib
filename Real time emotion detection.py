import cv2
from deepface import DeepFace

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

# getting emojis
emojis = {
    'angry': cv2.imread('angry.png', cv2.IMREAD_UNCHANGED),
    'happy': cv2.imread('happy.png', cv2.IMREAD_UNCHANGED),
    'surprise': cv2.imread('surprise.png', cv2.IMREAD_UNCHANGED),
    'neutral': cv2.imread('neutral.png', cv2.IMREAD_UNCHANGED),
    'sad': cv2.imread('sad.png', cv2.IMREAD_UNCHANGED),
    'fear': cv2.imread('fear.png', cv2.IMREAD_UNCHANGED)
}

emoji_size = (190, 190)  # change the size of emoji

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Action and emotion analysis
        results = DeepFace.analyze(frame[y:y+h, x:x+w], actions=('emotion'), enforce_detection=False)

        # Face detection checking
        if len(results) > 0:
            # Access the first face's analysis result
            face_result = results[0]

            # Retrieve dominant emotion
            emotion = face_result['dominant_emotion']
            cv2.putText(frame, "Emotion: " + emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # Display emoji
            emoji = emojis.get(emotion)
            if emoji is not None:
                emoji_resized = cv2.resize(emoji, emoji_size)
                emoji_h, emoji_w = emoji_resized.shape[:2]
                overlay = frame[y:y+emoji_h, x:x+emoji_w]
                emoji_resized_rgb = emoji_resized[:, :, :3] 
                overlay = cv2.resize(overlay, (emoji_w, emoji_h))
                blended = cv2.addWeighted(emoji_resized_rgb, 1.0, overlay, 0.7, 0)
                frame[y:y+emoji_h, x:x+emoji_w] = blended

    cv2.imshow('Emotion Detection', frame)

    key = cv2.waitKey(1)
    if key == 13 or key == 27:  # Check if Enter or Esc key is pressed
        break

cap.release()
cv2.destroyAllWindows()
