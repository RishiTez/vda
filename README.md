### Virtual Air Canvas with AI Object Recognition

Team Members:
SE22UCSE323 - Rishi Tez Reddy Dhava
SE23LCSE002 - Aryan Santosh Jakka

Requirements:
- Python 3.7 or higher
- A webcam
- Internet connection (for the first run only, to download AI models)

Running the code:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 main.py
```

Note: On the very first run, the application will take a few moments to download the YOLOv3-Tiny model files.

The application uses Hand Gestures to switch modes:
- Gesture
- Action
- Description
- Index Finger Up
- Draw / Write
- Paint on the canvas with the selected color.

Instructions:
-     Index + Middle Up
-     Hover / Select
-     Move the cursor without drawing. Use this to click buttons.
-     Hover over Colors
-     Change Color
-     Hover over Blue, Green, Red, or Yellow boxes.
-     Hover over "SCAN"
-     Scan Mode activates the AI Object Recognition tool.
-     Hover over "CLEAR"
-     Reset wipes the entire canvas clean.
-     Raise Index + Middle fingers and hover over the purple SCAN button until it turns grey.
-     Switch to Index Finger only.
-     Draw a square or circle around an object visible in your webcam (e.g., hold up a pen or bottle).
-     Lift your finger to finish the shape.
-     The AI will analyze the area inside your drawing and display the name of the object (e.g., "Cell Phone 90%").

Classes: It can recognize 80 different types of objects, including people, bicycles, cars, ties, bottles, cups, forks, scissors, laptops, cell phones, books etc.
# vda
