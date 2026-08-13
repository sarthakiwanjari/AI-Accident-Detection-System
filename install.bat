@echo off
echo ============================================
echo  AccidentWatch v3 — GPU Install Script
echo ============================================

echo.
echo [1/4] Upgrading pip...
python -m pip install --upgrade pip

echo.
echo [2/4] Installing PyTorch with CUDA 11.8 GPU support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo.
echo [3/4] Installing Ultralytics YOLOv8...
pip install ultralytics==8.2.0

echo.
echo [4/4] Installing remaining packages...
pip install flask==3.0.0 flask-socketio==5.3.6 python-dotenv==1.0.0 eventlet==0.35.1 twilio==8.10.0 opencv-python==4.9.0.80 numpy==1.26.4 Pillow==10.2.0 scipy==1.12.0 deep-sort-realtime==1.3.2 requests==2.31.0 geopy==2.4.1 pandas==2.2.0

echo.
echo [CHECK] Verifying GPU detection...
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT FOUND')"

echo.
echo [CHECK] Verifying YOLOv8...
python -c "from ultralytics import YOLO; m=YOLO('yolov8n.pt'); print('YOLOv8 ready')"

echo.
echo ============================================
echo  Installation complete!
echo  Run: python app.py
echo ============================================
pause
