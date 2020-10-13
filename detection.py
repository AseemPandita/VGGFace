from mtcnn.mtcnn import MTCNN
from matplotlib import pyplot
from numpy import asarray
from PIL import Image

def detect_face(imageFile, size=((224,224))):
    fullImage = pyplot.imread(imageFile)
    detector = MTCNN()
    results = detector.detect_faces(fullImage)
    x1, y1, w, h = results[0]['box']
    x2, y2 = x1 + w, y1 + h
    face = fullImage[y1:y2, x1:x2]
    image = Image.fromarray(face)
    image = image.resize(size)
    face_array = asarray(image)
    return face_array