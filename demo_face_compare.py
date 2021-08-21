import boto3
import cv2
import io
import numpy as np
import os
import pathlib

from typing import NewType, Tuple, List


Path = NewType('path', pathlib.Path)


def compare_faces(source: Path, target: bytes) -> Tuple[float]:
    '''Send pictures and evoke rekognition compare faces service.

    * Parameters:
      source: image of grid of preprocessed customer portraits (set to
              local file to do demostration)
      target: image of customer from camera at entrance

    * Return value: (left, top)
      left: origin point of horizontal axis of detected face frame in the
      fraction of whole picture
      top: origin point of vertical axis of detected face frame in the
      fraction of whole picture

    (left, top)   (left + width, top)
      |              |
      v              v
      +--------------+
      |              |
      |              |
      |   detected   |
      |     face     |  PS. "width" and "height" are not in this function
      |    frame     |
      |              |
      |              |
      +--------------+ <--(left + width, top + height)
    '''
    client = boto3.client('rekognition')

    with open(source, 'rb') as imageSource:
        response: dict = client.compare_faces(
                SimilarityThreshold=80,
                SourceImage={'Bytes': imageSource.read()},
                TargetImage={'Bytes': target.getvalue()}
                )

    if len(response['FaceMatches']) > 0:
        left = response['SourceImageFace']['BoundingBox']['Left']
        top = response['SourceImageFace']['BoundingBox']['Top']
    else:
        left = top = -1

    return left, top


def extract_features_from_aws_response(response,
                                       confidence_threshold=80) -> list:
    '''Extract the infomations we care from response of AWS Rekognition
    Face detection.

    This function is dedicated to get_face_features.

    * Parameters:
      response: the response from AWS Face detection
      confidence_threshold: tags will be pick up only when their
                            Confidence is higher than this number

    * Return:
      A list of features.
      1st element is a tuple, tells (min, max) of posible age.
      2nd element is gender, a str
      3rd element is emotions, a list of str
      The rest are str of features

      e.g. [(30, 46), 'Male', ['HAPPY'], 'Smile', 'Beard']
    '''
    features = []
    features.append((response['AgeRange']['Low'],
                     response['AgeRange']['High']))
    features.append(response['Gender']['Value'])

    features.append([])
    for emotion in response['Emotions']:
        if emotion['Confidence'] >= confidence_threshold:
            features[2].append(emotion['Type'])

    del response['Emotions']
    del response['MouthOpen']
    del response['EyesOpen']
    del response['AgeRange']
    del response['Confidence']
    del response['Quality']
    del response['Pose']
    del response['Landmarks']
    del response['BoundingBox']
    del response['Gender']

    for key in response.keys():
        if (response[key]['Confidence'] >= confidence_threshold
                and response[key]['Value']):
            features.append(key)

    return features


def fine_tune(frame: tuple) -> Tuple[int]:
    '''Make the square of face great than 80x80 pixels.

    (Since acceptable picture size of AWS Rekognition Face comparison is
    80x80 pixels.)

    This function is dedicated to get_face_features.
    '''
    x, y, dx, dy = frame

    if dy < 80:
        y -= int((80 - dy) / 2)
        dy = 80

    if dx < 80:
        x -= int((80 - dx) / 2)
        dx = 80

    return x, y, dx, dy


def get_face_features(target: bytes) -> List[str]:
    '''Send image to AWS Rekgnition face-detection and extract features.

    We divide the process into 3 parts:
    1. Recognize each faces in target picture and cut them down saparately
    2. Send each face to AWS Rekognition face analysis
    3. Extract infomation in AWS response

    * Parameters:
      target: the picture we send to AWS.
              According to official document, this should be .jpg or .png

    * Return:
      A list of features of detected faces.

      [face1_features, face2_features, ...]

      For each list element:
        1st element is the face picture in bytes (encoded to PNG)
        2nd element is a tuple, tells (min, max) of posible age.
        3rd element is gender, a str
        4th element is emotions, a list of str
        The rest are str of features

      e.g. [<_io.BytesIO>, (30, 46), 'Male', ['HAPPY'], 'Smile', 'Beard']
    '''
    client = boto3.client('rekognition')

    lbp_face_cascade = cv2.CascadeClassifier(
            'data/lbpcascade_frontalface.xml'
            )
    imgarr = np.frombuffer(target, np.uint8)
    img_np = cv2.imdecode(imgarr, cv2.COLOR_BGR2GRAY)
    faces = lbp_face_cascade.detectMultiScale(img_np,
                                              scaleFactor=1.1,
                                              minNeighbors=5)

    analysis = []
    if len(faces):  # If there is any face
        for index, value in enumerate(faces):
            x, y, delta_x, delta_y = fine_tune(value)

            is_success, buffer = cv2.imencode(".png", img_np[y:y+delta_y,
                                                             x:x+delta_x])
            io_buf = io.BytesIO(buffer)

            analysis.append([io_buf])
            analysis[index] = analysis[index] + (
                extract_features_from_aws_response(
                    client.detect_faces(
                        Image={'Bytes': io_buf.getvalue()},
                        Attributes=['ALL']
                    )['FaceDetails'][0]
                )
            )

    return analysis


def entrance_camera_handler() -> bytes:
    '''This function simulates camera shot at entrance.

    * Return value: image itself
    '''
#    TODO: Rewrite it to real handler that returns a binary data.
#    TODO: It should able to deal with multiple face in one shot
#          Take look of it:
#          https://www.superdatascience.com/blogs/opencv-face-recognition
    with open('fake_users/Bradley_Cooper5.jpeg', 'rb') as img_file:
        img_data = img_file.read()

    return img_data


def find_customer(picture: str, left: float, top: float) -> str:
    '''Figure out who is the customer by the position in source image.

    * Parameters:
      picture: the name of the picture
      left: left of detected face in fraction of picture width
      top: top of detected face in fraction of picture height

    * Return value:
      a string of customer ID
      (For the purpose of demostration, now we return an integer.)

    Pictures' number should count from 0, and saparate number with other
    words by double underscore. Otherwise, this function would failed.

    e.g.
    male__0.png
    male__1.png
    male__2.png

    e.g.
    32foo_bar__0.png
    32foo_bar__100.png
    '''
    # The maximum number of people that a target picture can contain
    GRID_SIZE_OF_PIC: int = 4

    pic_label: int = int(picture.name.split('.')[0].split('__')[1])
    row_size: int = GRID_SIZE_OF_PIC ** .5  # how many people each row
    unit = 1 / row_size
    id_offset = pic_label * GRID_SIZE_OF_PIC

    # TODO: customer ID table should be queried from database
    id_table = [[1 + id_offset, 2 + id_offset],
                [3 + id_offset, 4 + id_offset]]
    return id_table[int(top // unit)][int(left // unit)]


def main() -> None:
    ''''''
    os.chdir(os.path.dirname(__file__))

    group_root: Path = pathlib.Path('fake_users/groups/')
    # TODO: write the argorithm to deal with mutiple features
    target_file: bytes = entrance_camera_handler()
    customers: list = get_face_features(target_file)
    # [<_io.BytesIO>, (30, 46), 'Male', ['HAPPY'], 'Smile', 'Beard']

    # TODO: rewrite this block to multi-threaded to accelerate
    if len(customers):  # If there is any faces in target
        for person in customers:
            customer_face: bytes = person.pop(0)
            customer_emotion: list = person.pop(2)  # later use
            customer_age: tuple = person.pop(0)
            # Maybe average age is not a good idea?
            customer_age: str = str(
                    int((customer_age[0] + customer_age[1]) / 2)
                    )

            # TODO: sort "person" to the form of that the feature belongs
            #       to the lesser customer the higher prior to compare
            person.insert(0, customer_age)
            person.reverse()  # minor first

            customer_id: str = str()
            for feature in person:
                flag: bool = False  # True if confirm who is he/she
                feature = feature.lower()  # everything in local are lower

                for picture in group_root.glob(feature + '/*'):
                    source_file: Path = pathlib.Path(picture).resolve()
                    face_left, face_top = compare_faces(source_file,
                                                        customer_face)
                    # What if customer has a twin silbin?
                    if face_left >= 0:  # Stop iteration at first match
                        customer_id = str(find_customer(picture,
                                                        face_left,
                                                        face_top))
                        print('Welcome back! Customer', customer_id)
                        flag = True
                        break

                if flag:
                    break

    # TODO: query from database and prepare data for mobile app


if __name__ == "__main__":
    main()
