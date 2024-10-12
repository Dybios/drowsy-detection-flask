# drowsy-detection-flask
Drowsiness detection based on their eye movements. Paired with a web server using Python's Flask to transmit the video (TODO: audio as well) stream over a network. 

Based on drowsiness detection implementation by msindev\
https://github.com/msindev/Driver-Drowsiness-Detector

To run the python script with the device default IP over port 8000, use the following command:

```python drowsy_detect.py --ip 0.0.0.0 --port 8000 --shape-predictor models/shape_predictor_68_face_landmarks.dat```
