# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# PDX-License-Identifier: MIT-0 (For details, see https://github.com/awsdocs/amazon-rekognition-developer-guide/blob/master/LICENSE-SAMPLECODE.)

import boto3
import json
import os
import pathlib


def compare_faces(sourceFile, targetFile):
    '''Send pictures and evoke rekognition compare faces service.

    Returns resolved response.
    '''
    client = boto3.client('rekognition')

    with open(sourceFile, 'rb') as imageSource, \
            open(targetFile, 'rb') as imageTarget:
        response = client.compare_faces(
                SimilarityThreshold=80,
                SourceImage={'Bytes': imageSource.read()},
                TargetImage={'Bytes': imageTarget.read()}
                )

    for faceMatch in response['FaceMatches']:
        position = faceMatch['Face']['BoundingBox']
        similarity = str(faceMatch['Similarity'])
        print('The face at ' +
              str(position['Left']) + ' ' +
              str(position['Top']) +
              ' matches with ' + similarity + '% confidence')

    with open('response', 'a') as f:  # for inspecting response only
        json.dump(response, f, indent=True)

    return len(response['FaceMatches'])


def face_features():
    '''Set to only "male" to demostration customer comparason.

    TODO: Rewrite it to really doing face analysis.
    '''
    return ['male']


def entrance_camera_handler():
    '''This function simulates camera shot at entrance.

    TODO: Rewrite it to real handler that returns a binary data.
    '''
    return 'fake_users/Bradley_Cooper5.jpeg'


def main():
    os.chdir(os.path.dirname(__file__))

    # for inspecting response only
    pathlib.Path('response').unlink(missing_ok=True)

    group_root = pathlib.Path('fake_users/groups/')

    # Let's say, the result of features of the incoming customer be
    # only "male"
    # The argorithm to deal with mutiple features wanted...
    features = face_features()

    target_file = entrance_camera_handler()

    for feature in features:
        for picture in group_root.glob(feature + '/*'):
            source_file = pathlib.Path(picture).resolve()
            #TODO: rewrite it to multi-threaded compare to accelerate

            print('Subgroup in', feature, ':', pathlib.Path(picture).name)
            face_matches = compare_faces(source_file, target_file)
            print("Face matches: " + str(face_matches))


if __name__ == "__main__":
    main()
