from ultralytics import YOLO
import cv2
import CONST4Interaction as CONST 
import pandas
import time
import requests
import json

class VideoAnalyzer:
    def __init__(self):
        # Initialize YOLO models
        self.interaction_model = YOLO(CONST.INTERACTION_WEIGHTS)
        self.people_model = YOLO(CONST.PEOPLE)

    def interaction_detection(self,frame,model):
        results = model(source=frame,
                    show=False,
                    device='cuda',
                    stream=False,
                    vid_stride=1,
                    conf=0.3,
                    max_det=50,
                    iou=0.5,
                    verbose=False)
        data = results[0].boxes.data.tolist()
        output = pandas.DataFrame(data, columns=['x1', 'y1', 'x2', 'y2', 'conf', 'class'])
        class_counts = output['class'].value_counts().reset_index()
        class_counts.columns = ['class', 'count']
        return output,class_counts

    def people_detection (self,img,model):
        results = model(source=img,
                    show=False,
                    device='cuda',
                    stream=False,
                    vid_stride=1,
                    conf=0.3,
                    max_det=50,
                    iou=0.5,
                    classes=[0],
                    verbose=False)
        data = results[0].boxes.data.tolist()
        output = pandas.DataFrame(data, columns=['x1', 'y1', 'x2', 'y2', 'conf', 'class'])
        number_persons = output[output['class'] == 0].shape[0]
        return number_persons

    def person_counter(self,interactions,frame):
        total_people = 0
        for n in interactions.index:
            x1, y1, x2, y2 = map(int, interactions.loc[n, ['x1', 'y1', 'x2', 'y2']])
            person_img = frame[y1:y2, x1:x2]
            people = self.people_detection(person_img,self.people_model)
            total_people += people
        return total_people

    def video_capture(self, source, enrollment = 25):
        """
        Capture video from a specified source, analyze frames, and send the results.

        Args:
            source (int or str): Video source (camera index or file path).
            enrollment (int): The class enrollment, whihc is an input parameter that should be known.
        """
        cap = cv2.VideoCapture(source)

        while True:
            start_time = time.time()
            end_time = start_time + 5  # Reporting the interaction levels every five seconds

            frame_counter = 0
            no_interaction = 0
            object_interaction = 0
            prof_interaction = 0
            student_interaction = 0
            

            fps = cap.get(cv2.CAP_PROP_FPS)

            while time.time() <= end_time:
                ret, frame = cap.read()
                if ret:
                    frame = cv2.resize(frame, (640, 480))

                    output, class_counts = self.interaction_detection(frame, self.interaction_model)
                    output_valid_interactions_stud_prof = output[output['class']==2.0]
                    output_valid_interactions_stud_stud = output[output['class']==3.0]


                    persons_no_int_count = len(output[output['class']==0.0].index)
                    persons_stud_obj_count = len(output[output['class']==1.0].index)
                    persons_stud_prof_count = self.person_counter(output_valid_interactions_stud_prof,frame)
                    persons_stud_stud_count = self.person_counter(output_valid_interactions_stud_stud,frame)

                    persons_count = persons_no_int_count + persons_stud_obj_count + persons_stud_prof_count + persons_stud_stud_count
                    persons_count = min(persons_count, enrollment)

                    no_interaction_level = 0.0
                    student_objt_interaction_level = 0.0
                    student_prof_interaction_level = 0.0
                    student_student_interaction_level = 0.0

                    if persons_count > 0:
                        no_interaction_level = persons_no_int_count/persons_count
                        student_objt_interaction_level = persons_stud_obj_count/persons_count
                        student_prof_interaction_level = min(persons_stud_prof_count/persons_count, 1.0)
                        student_student_interaction_level = min(persons_stud_stud_count/persons_count, 1.0)

                    no_interaction += no_interaction_level
                    object_interaction += student_objt_interaction_level
                    prof_interaction += student_prof_interaction_level
                    student_interaction += student_student_interaction_level
                    frame_counter += 1
                else:
                    pass

            # We are assuming a moderated approach by averaging the interactions during the 5 seconds. 
            # Alternatively, we could use the maximum or minimum value.
            object_interaction = object_interaction / frame_counter
            prof_interaction = prof_interaction / frame_counter
            student_interaction = student_interaction / frame_counter
            no_interaction = no_interaction / frame_counter

            interaction_level = min((object_interaction + prof_interaction + student_interaction), 1.0) - no_interaction

            values = {
                'no_interaction_ratio':no_interaction,
                'prof_stud_ratio': prof_interaction,
                'stud_stud_ratio': student_interaction,
                'stud_obj_ratio': object_interaction,
                'interaction_level': interaction_level
            }

            #self.send_info(values)
            print(values)
            print("Frames per second (FPS):", fps)

    def send_info(self, values):
        """
        Send information to a specified endpoint.

        Args:
            values (dict): Dictionary containing the values to be sent.
        """
        url = CONST.ENDPOINT
        json_data = json.dumps(values)
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, data=json_data, headers=headers)

        if response.status_code == 200:
            print("Request was successful")
        else:
            print("Request failed with status code:", response.status_code)

if __name__ == "__main__":
    analyzer = VideoAnalyzer()
    analyzer.video_capture(CONST.SOURCE)
