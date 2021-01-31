# from imageai.Prediction import ImagePrediction
from imageai.Detection import ObjectDetection
import os
import cv2
import pymongo
from config import atlas_url

client = pymongo.MongoClient(atlas_url)

execution_path = os.getcwd()

# import the model
# prediction = ImagePrediction()
# prediction.setModelTypeAsResNet()
# prediction.setModelPath(os.path.join(execution_path, "resnet50_weights_tf_dim_ordering_tf_kernels.h5"))
# prediction.loadModel()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()


def insert(obj):
    print(obj)
    # client['discovery']['video_search'].insert_one(obj)

def file_path(id):
    return f'/Users/ethan.steininger/Desktop/Dev/customers/Discovery/video-frame-search/app/static/img/frames/${id}.jpg'


def grab_tags(frame, timestamp, frame_id):
    # predictions, probabilities = prediction.predictImage(os.path.join(execution_path, frame), result_count=5)
    tags = []
    # tag_str = ''
    # for pred, prob in zip(predictions, probabilities):
    #     clean_tag_prediction = pred.replace("_", " ")
    #     # tags.append(clean_tag_prediction)
    #     # tag_str += f"{clean_tag_prediction} "
    #     t = {
    #         "tag": clean_tag_prediction,
    #         "probability": prob
    #     }
    #     tags.append(t)
    #
    # return {"timestamp": timestamp, "tags": tags, "video": "Mythbusters: Defying Gravity, Levitating a Car "}
    detections = detector.detectObjectsFromImage(
        input_image=frame,
        output_image_path=file_path(frame_id),
        minimum_percentage_probability=30
    )

    for eachObject in detections:
        t = {
            "tag": eachObject["name"],
            "probability": eachObject["percentage_probability"],
            "location": eachObject["box_points"]
        }
        tags.append(t)

    return {"timestamp": timestamp, "tags": tags, "video": "Mythbusters: Defying Gravity, Levitating a Car "}




def video_to_frames():
    vidcap = cv2.VideoCapture('mythbusters.mp4')
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    seconds = 0.5
    success, image = vidcap.read()
    multiplier = fps * seconds
    count = 0

    while success:
        #current frame number, rounded b/c sometimes you get frame intervals which aren't integers...this adds a little imprecision but is likely good enough
        frame_id = int(round(vidcap.get(1)))
        success, image = vidcap.read()
        # skip based on multiplier
        # every 10th frame:
        if frame_id % multiplier == 0:
            filepath = file_path(frame_id)
            timestamp = count/fps
            cv2.imwrite(filepath, image)
            # grab tags
            tag_obj = grab_tags(filepath, timestamp, frame_id)
            # os.remove(filepath)
            # insert
            print(tag_obj)
            # tag_obj['img'] = frame_id
            # # make sure there are actually tags !
            # if tag_obj['tags']:
            #     insert(tag_obj)
            # else:
            #     os.remove(filepath)
            # next !
        print('Read a new frame: ', success)
        count += 1


if __name__ == '__main__':
    video_to_frames()
