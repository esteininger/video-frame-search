import youtube_dl
# from imageai.Prediction import ImagePrediction
from imageai.Detection import ObjectDetection
import os
import cv2
import pymongo

video_url = 'https://www.youtube.com/watch?v=M672sEfGZhg'
mongo_url = "mongodb+srv://ethan:1RrQIU5UZrp5Gci2@dev.v7a3k.mongodb.net/test?ssl=true&ssl_cert_reqs=CERT_NONE"
db = 'discovery'
collection = 'video_search'
frame_path = "/Users/ethan.steininger/Desktop/Dev/customers/Discovery/video-frame-search/app/static/img/frames/"
video_file_name = 'video.mp4'


def download_video():
    ydl_opts = {
        'outtmpl': video_file_name,
        'format':'137',
        'nocheckcertificate': True
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


class Mongo:
    def __init__(self, mongo_url):
        self.client = pymongo.MongoClient(mongo_url)
        self.collection = self.client[db][collection]

    def insert_one(self, doc):
        self.collection.insert_one(doc)


class AI:
    def __init__(self):
        self.execution_path = os.getcwd()

    def init_object_detection(self):
        #import the model
        self.detector = ObjectDetection()
        self.detector.setModelTypeAsRetinaNet()
        self.detector.setModelPath(os.path.join(self.execution_path , "resnet50_coco_best_v2.0.1.h5"))
        self.detector.loadModel()


    def init_image_prediction(self):
        # import the model
        self.prediction = ImagePrediction()
        self.prediction.setModelTypeAsResNet()
        self.prediction.setModelPath(os.path.join(self.execution_path, "resnet50_weights_tf_dim_ordering_tf_kernels.h5"))
        self.prediction.loadModel()


    def grab_tags(self, type, frame, timestamp, frame_id):
        if type == 'image':
            # get predictions & probabilities for frame
            predictions, probabilities = self.prediction.predictImage(
                os.path.join(self.execution_path, frame),
                result_count=5
            )
            tags = []
            for prediction, probability in zip(predictions, probabilities):
                # remove underscores
                t = {
                    "tag": pred.replace("_", " "),
                    "probability_score": probability
                }
                tags.append(t)

            return {"timestamp": timestamp, "tags": tags}

        if type == 'object':
            detections = detector.detectObjectsFromImage(
                input_image=frame,
                output_image_path=f"{frame_path}/{frame_id}.jpg",
                minimum_percentage_probability=30
            )

            for object_tagged in detections:
                t = {
                    "tag": object_tagged["name"],
                    "probability": object_tagged["percentage_probability"],
                    "location": object_tagged["box_points"]
                }
                tags.append(t)

            return {"timestamp": timestamp, "tags": tags}




def video_to_frames():
    vidcap = cv2.VideoCapture(video_file_name)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    # capture a frame every 1/2 a second
    seconds = 0.5
    success, image = vidcap.read()
    multiplier = fps * seconds
    count = 0
    ai = AI()

    while success:
        # every 10th frame:
        frame_id = int(round(vidcap.get(1))) #current frame number, rounded b/c sometimes you get frame intervals which aren't integers...this adds a little imprecision but is likely good enough
        # save frame as JPEG file
        success, image = vidcap.read()
        # skip based on multiplier
        if frame_id % multiplier == 0:
            filename = f"frame-{frame_id}.jpg"
            filepath = f"{frame_path}{filename}"
            timestamp = count/fps
            cv2.imwrite(filepath, image)
            # grab tags
            tag_obj = ai.grab_tags(filepath, timestamp, frame_id)
            os.remove(filepath)
            # # insert
            tag_obj['img'] = frame_id
            # make sure there are actually tags !
            if tag_obj['tags']:
                insert(tag_obj)
            else:
                os.remove(f"{frame_path}{frame_id}.jpg")
            # next !
        print('Read a new frame: ', success)
        count += 1


if __name__ == '__main__':
    # download video first
    download_video()
    # video to frames
    video_to_frames()
