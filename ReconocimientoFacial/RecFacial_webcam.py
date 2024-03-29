from imutils import paths
from imutils.video import VideoStream
import imutils, face_recognition, cv2, os, pickle, time
from collections import Counter

ti = time.time()

data = pickle.loads(open(os.getcwd() + '\\encodings_webcam_uni.pickle', 'rb').read())

vs = VideoStream(src=0).start()
time.sleep(12.0)
writer = None

while True:
    frame = vs.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = imutils.resize(frame, width=750)
    r = frame.shape[1] / float(rgb.shape[1])
    boxes = face_recognition.face_locations(rgb, model='hog')
    encodings = face_recognition.face_encodings(rgb, boxes)
    names = []
    for encoding in encodings:
        votes = face_recognition.compare_faces(data['encodings'], encoding)
        if True in votes:
            names.append(Counter([name for name, vote in list(zip(data['names'], votes)) if vote == True]).most_common()[0][0])
        else:
            names.append('Unknown')
    for ((top, right, bottom, left), name) in zip(boxes, names):
        top, right, bottom, left = int(top * r), int(right * r), int(bottom * r), int(left * r)
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    if writer is None:
        writer = cv2.VideoWriter(os.getcwd() + '\\webcam_test\\output.avi', cv2.VideoWriter_fourcc(*'MJPG'), 24, (frame.shape[1], frame.shape[0]), True)
    writer.write(frame)
    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cv2.destroyAllWindows()
vs.stop()
writer.release()
print('Done! \nTime taken: {:.1f} minutes'.format((time.time() - ti)/60))