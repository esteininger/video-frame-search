from imageai.Detection import ObjectDetection
from imageai.Classification import ImageClassification

import os
import cv2
import pymongo

# because config is in parent directory
import sys
sys.path.insert(0,'..')
from config import mongo_uri


abs_filepath = '/Users/ethan.steininger/Desktop/Dev/customers/Discovery/video-frame-search/static/img/frames/'
video_path = 'mythbusters.mp4'


class Mongo:
    def __init__(self, mongo_url):
        self.client = pymongo.MongoClient(mongo_uri)
        self.collection = self.client[db][collection]

    def insert(self, doc):
        self.collection.insert_one(doc)

class AI:
    def __init__(self):
        self.execution_path = os.getcwd()

    def init_object_detection(self):
        #import the model
        self.detector = ObjectDetection()
        self.detector.setModelTypeAsRetinaNet()
        self.detector.setModelPath(os.path.join(self.execution_path , "resnet50_coco_best_v2.1.0.h5"))
        self.detector.loadModel()

    def grab_object_tags(self, frame, timestamp):
        detections = self.detector.detectObjectsFromImage(
            input_image=frame,
            output_image_path=frame,
            minimum_percentage_probability=30
        )

        tags = []
        for eachObject in detections:
            t = {
                "tag": eachObject["name"],
                "probability": eachObject["percentage_probability"],
                "location": eachObject["box_points"]
            }
            tags.append(t)

        return {"timestamp": timestamp, "tags": tags}


    def init_image_classification(self):
        # import the model
        self.prediction = ImageClassification()
        self.prediction.setModelTypeAsResNet50()
        self.prediction.setModelPath(os.path.join(self.execution_path, "resnet50_imagenet_tf.2.0.h5"))
        self.prediction.loadModel()

    def grab_image_tags(self, frame, timestamp):
        predictions, probabilities = self.prediction.classifyImage(frame, result_count=10)
        for eachPrediction, eachProbability in zip(predictions, probabilities):
            print(eachPrediction , " : " , eachProbability)


def main():
    mongo = Mongo()
    ai = AI()
    ai.init_object_detection()
    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    # capture every N seconds of frames
    success, image = vidcap.read()
    #
    seconds = 0.5
    multiplier = fps * seconds
    count = 0
    while success:
        # current frame number, rounded b/c sometimes you get
        # frame intervals which aren't integers...this adds a
        # little imprecision but is likely good enough
        frame_id = int(round(vidcap.get(1)))
        success, image = vidcap.read()
        # skip based on multiplier
        # every 10th frame:
        if frame_id % multiplier == 0:
            timestamp = count/fps
            filepath = abs_filepath + f"{frame_id}.jpg"
            cv2.imwrite(filepath, image)
            # grab tags
            tag_obj = ai.grab_object_tags(filepath, timestamp)
            # only save frames that have tags
            if tag_obj['tags']:
                # insert
                mongo.insert(tag_obj)
            else:
                os.remove(filepath)
        count += 1


if __name__ == '__main__':
    main()
