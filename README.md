# 🚑 AI-Accident-Detection-System
AI-Based Accident Detection and Emergency Alert System
📌 Overview
AI-Accident-Detection-System is a real-time AI-powered system that detects road accidents from video input and automatically triggers emergency alerts.

The system uses computer vision to monitor traffic footage and identify potential collisions, reducing the delay in emergency response.

💡 Problem Statement
Road accidents often go unreported for crucial minutes due to lack of real-time monitoring.

Manual CCTV monitoring is unreliable
Emergency response is delayed
Lack of instant accident reporting systems
🚀 Solution
AI-Accident-Detection-System detects accidents in real-time and automatically sends alerts without human intervention.

Detects vehicle collisions using AI
Triggers emergency alerts instantly
Provides a live dashboard for monitoring
🛠️ Tech Stack
Object Detection: YOLOv8
Tracking: DeepSORT
Backend: Flask + SocketIO
Computer Vision: OpenCV
Alerts: Twilio (Emergency Call System)
Frontend: HTML, CSS, JavaScript
⚙️ How It Works
Video input is captured (webcam/video file)
YOLOv8 detects vehicles in each frame
DeepSORT tracks vehicles across frames
System analyzes movement patterns for collision detection
Accident is detected based on multiple conditions
Emergency alert (call) is triggered using Twilio
Dashboard updates in real-time
🎯 Features
Real-time accident detection
Live dashboard with UI
Automated emergency call alerts
Vehicle detection and tracking
Works with video or live camera
▶️ How to Run
1. Clone the repository
git clone cd AI-Accident-Detection-System

2. Install dependencies
pip install -r requirements.txt

3. Setup environment variables
Create a .env file and add:

TWILIO_ACCOUNT_SID=your_sid TWILIO_AUTH_TOKEN=your_token TWILIO_PHONE_NUMBER=your_number

4. Run the application
python app.py

5. Open in browser
http://localhost:5000

📂 Project Structure
AI-Accident-Detection-System/ ├── app.py ├── detection_engine.py ├── requirements.txt ├── install.bat ├── alerts/ ├── static/ ├── templates/

👥 Team
Sarthaki Wanjari
Mayuri Rathod
Aadesh Bhatkar
Tilak Pathak
Prerna Nichat
🏆 Achievement
🥈 2nd Runner-Up at a hackathon at Sipna College of Engineering

📌 Note
This project was developed during a hackathon with limited time, focusing on building a functional prototype.

🔮 Future Improvements
SMS + Email alert integration
GPS-based live location tracking
Higher accuracy detection model
Deployment on edge devices
✨ Final Thought
Built with the idea that faster detection can save lives. 🔥
