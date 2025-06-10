#!/usr/bin/env python3
# test_server.py - Server'ı test etmek için basit script

import sys
import os

print("Python versiyonu:", sys.version)
print("Çalışma dizini:", os.getcwd())
print("-" * 50)

# 1. Import testleri
print("1. Kütüphaneler kontrol ediliyor...")
try:
    import socket
    print("   ✓ socket")
except ImportError:
    print("   ✗ socket BULUNAMADI!")

try:
    import threading
    print("   ✓ threading")
except ImportError:
    print("   ✗ threading BULUNAMADI!")

try:
    import json
    print("   ✓ json")
except ImportError:
    print("   ✗ json BULUNAMADI!")

try:
    import base64
    print("   ✓ base64")
except ImportError:
    print("   ✗ base64 BULUNAMADI!")

try:
    import cv2
    print("   ✓ cv2 (OpenCV)")
    print(f"     OpenCV versiyon: {cv2.__version__}")
except ImportError as e:
    print(f"   ✗ cv2 (OpenCV) BULUNAMADI! Hata: {e}")
    print("     Yüklemek için: pip install opencv-python")

try:
    import numpy as np
    print("   ✓ numpy")
    print(f"     NumPy versiyon: {np.__version__}")
except ImportError as e:
    print(f"   ✗ numpy BULUNAMADI! Hata: {e}")
    print("     Yüklemek için: pip install numpy")

try:
    import mediapipe as mp
    print("   ✓ mediapipe")
    print(f"     MediaPipe versiyon: {mp.__version__}")
except ImportError as e:
    print(f"   ✗ mediapipe BULUNAMADI! Hata: {e}")
    print("     Yüklemek için: pip install mediapipe")

print("-" * 50)

# 2. Port kontrolü
print("2. Port durumu kontrol ediliyor...")
import socket

def check_port(host, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)
    result = sock.connect_ex((host, port))
    sock.close()
    return result == 0

if check_port('localhost', 5000):
    print("   ⚠️  Port 5000 zaten kullanımda!")
    print("   Başka bir port deneyin veya mevcut işlemi durdurun.")
else:
    print("   ✓ Port 5000 kullanılabilir")

print("-" * 50)

# 3. Basit socket testi
print("3. Socket bağlantı testi...")
try:
    test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    test_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    test_socket.bind(('0.0.0.0', 5555))
    test_socket.close()
    print("   ✓ Socket oluşturma başarılı")
except Exception as e:
    print(f"   ✗ Socket hatası: {e}")

print("-" * 50)

# 4. MediaPipe testi
print("4. MediaPipe Pose testi...")
try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    print("   ✓ MediaPipe Pose başarıyla oluşturuldu")
    pose.close()
except Exception as e:
    print(f"   ✗ MediaPipe hatası: {e}")
    import traceback
    traceback.print_exc()

print("-" * 50)
print("\nTüm testler tamamlandı!")
print("\nEğer tüm testler başarılıysa, server'ı çalıştırmayı deneyin:")
print("python mediapipe_server.py")