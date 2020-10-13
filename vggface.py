from numpy import expand_dims
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from keras_vggface.utils import decode_predictions

def identify_faces(detected_faces):
    faceArray1D = expand_dims(detected_faces, axis = 0)
    faceArray1D = preprocess_input(faceArray1D, version=2)
    model = VGGFace(model = 'resnet50')
    yhat = model.predict(faceArray1D)
    results = decode_predictions(yhat)
    return results