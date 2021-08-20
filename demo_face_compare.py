# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# PDX-License-Identifier: MIT-0 (For details, see https://github.com/awsdocs/amazon-rekognition-developer-guide/blob/master/LICENSE-SAMPLECODE.)

import boto3
import os
import pathlib

from typing import NewType, Tuple, List


Path = NewType('path', pathlib.Path)


def compare_faces(source: Path, target: bytes) -> Tuple[float]:
    '''Send pictures and evoke rekognition compare faces service.

    * Parameters:
      source: image of grid of preprocessed customer portraits (set to
              local file to do demostration)
      target: image of customer at entrance

    * Return value: (left, top)
      left: horizontal axis origin point of detected face in the fraction
            of whole picture
      top: vertical axis origin point of detected face in the fraction
            of whole picture

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
                TargetImage={'Bytes': target}
                )

    if len(response['FaceMatches']) > 0:
        left = response['SourceImageFace']['BoundingBox']['Left']
        top = response['SourceImageFace']['BoundingBox']['Top']
    else:
        left = top = -1

    return left, top


def face_features() -> List[str]:
    '''Send image to AWS Rekgnition face-detection to extract features.

    Set to only "male" to demostration customer comparason.
    '''
#    TODO: Rewrite it to really doing face analysis.
    return ['male']


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


def resolve_customer(picture: str, left: float, top: float) -> str:
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
    os.chdir(os.path.dirname(__file__))

    group_root: Path = pathlib.Path('fake_users/groups/')
    # TODO: write the argorithm to deal with mutiple features
    features: List[str] = face_features()
    target_file: bytes = entrance_camera_handler()

    # TODO: rewrite this block to multi-threaded to accelerate
    for feature in features:
        for picture in group_root.glob(feature + '/*'):
            source_file: Path = pathlib.Path(picture).resolve()
            face_left, face_top = compare_faces(source_file, target_file)
            # What if customer has a twin silbin?
            if face_left >= 0:  # Stop iteration at first match
                customer: str = resolve_customer(picture, face_left,
                                                 face_top)
                break

    print('Here you are! Dear customer', customer)


if __name__ == "__main__":
    main()
