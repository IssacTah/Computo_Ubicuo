from imutils import paths
from imutils.video import VideoStream
import imutils, face_recognition, cv2, os, pickle
from collections import Counter

print('[INFO] creating facial embeddings...')
try:
    # Cargar el archivo de encodings indicado
    data = pickle.loads(open(os.getcwd() + '\\encodings_webcam_uni.pickle', 'rb').read())
    print('Archivo encodings.pickle encontrado. Procediendo con el resto del código.')
except FileNotFoundError:
    print('No se ha encontrado el archivo encodings.pickle...')
knownEncodings, knownNames = [], []
imagePaths = list(paths.list_images(os.getcwd() + '\\dataset_webcam'))
for (i, imagePath) in enumerate(imagePaths):
    print('{}/{}'.format(i+1, len(imagePaths)), end=', ')
    image, name = cv2.imread(imagePath), imagePath.split(os.path.sep)[-2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb,  model='cnn')
    for encoding in face_recognition.face_encodings(rgb, boxes):
        knownEncodings.append(encoding)
        knownNames.append(name)
data = {'encodings': knownEncodings, 'names': knownNames}
print(data)

# Guardar en el nuevo archivo generado
with open(os.getcwd() + '\\encodings_webcam_uni.pickle', 'wb') as f:
    f.write(pickle.dumps(data))
