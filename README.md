**Classroom Interaction Estimator**

__Overview__

VideoAnalyzer is a Python-based tool utilizing YOLOv8 models to detect and analyze interaction levels in face-to-face lessons. It captures video frames, detects interactions among students, professors, and objects, and estimates interaction levels over time.

__Features__

<ol>
<li>Interaction Detection: Identifies various interaction types, including no interaction, student-object, student-professor, and student-student interactions.</li>

<li> People Detection: Uses YOLOv8 to count people in frames.</li>

<li> Person Counter: Estimates the number of individuals involved in interactions.</li>

<li> Real-time Video Processing: Analyzes video feeds and calculates interaction metrics every five seconds.</li>

<li> Data Transmission: Sends interaction level data to a specified endpoint.</li>
</ol>

__Requirements__

The following dependencies are required:

pip install ultralytics opencv-python pandas requests

Additionally, ensure that you have CUDA installed for GPU acceleration.

__Usage__

Initialization

The VideoAnalyzer class initializes two YOLO models:

interaction_model: Detects interaction types.

people_model: Counts the number of individuals in interactions.

analyzer = VideoAnalyzer()
analyzer.video_capture(CONST.SOURCE)

__Methods__

> interaction_detection(frame, model)
> >
> > Runs YOLO on a given frame to detect interactions.
> >
> > Returns bounding box coordinates and interaction class counts.
>
> people_detection(img, model)
> 
> > Detects the number of people in a given image.
>
> person_counter(interactions, frame)
> > 
> > Counts the number of people involved in detected interactions.
>
> video_capture(source, enrollment=25)
> > 
> > Captures video from a source, processes frames, and calculates interaction levels every five seconds.
> >
> > enrollment: The total number of students in the class.
>  
> send_info(values)
> > 
> > Sends the computed interaction levels to a defined endpoint.

__Output__

The video_capture method outputs interaction levels as a dictionary:
```
{
    'no_interaction_ratio': 0.35,
    'prof_stud_ratio': 0.25,
    'stud_stud_ratio': 0.30,
    'stud_obj_ratio': 0.10,
    'interaction_level': 0.65
}
```

It also prints the video frame rate (FPS) for performance monitoring.

__Configuration__
> Update the CONST variables in the script:
>
> CONST.INTERACTION_WEIGHTS: Path to the interaction detection model.
>
> CONST.PEOPLE: Path to the people detection model.
>
> CONST.SOURCE: Video source (camera index or file path).
>
> CONST.ENDPOINT: API endpoint for sending interaction data.

__License__

This project is licensed under the Creative Common 1.0 License.
