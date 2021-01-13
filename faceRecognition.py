from detection import detect_face
from vggface import identify_faces
import os
# Show predicted name if the confidence is more than 80% for the given test image
THRESHOLD = 0.6

names = []
groundTruth = []



images = os.listdir('images')

for image in images:
    IMAGEPATH = 'images/'+image
    detected_faces = detect_face(IMAGEPATH).astype('float32')
    identities = identify_faces(detected_faces)


    confidence = 0
    name = 'Unknown'
    for identity in identities[0]:
        print(identity[0], identity[1]*100)
        if identity[1] > THRESHOLD:
            confidence = identity[1]
            name = identity[0]

    print('\n\n\n\n')

    if name != 'Unknown':
        name = name.split(' ')[1].split("'")[0]
        names.append(name)
        groundTruth.append(image)
        print('Face in the image: {} [{}] \nConfidence:{}'.format(name, image, confidence*100))

    else:
        print('Could not identify; {}'.format(image))
        names.append('unknown')
        groundTruth.append(image)
    print('\n\n\n\n')

print(names)
print(groundTruth)