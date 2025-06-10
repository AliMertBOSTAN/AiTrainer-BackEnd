import socket
import threading
import json
import base64
import cv2
import numpy as np
import mediapipe as mp
from io import BytesIO
import struct
import time
import math

# MediaPipe setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class ExerciseTracker:
    def __init__(self):
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=True,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75
        )
        self.counters = {}
        self.states = {}
        self.directions = {}
        
    def calculate_angle(self, a, b, c):
        """3 nokta arasındaki açıyı hesapla"""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0])
        angle = math.degrees(radians)
        
        if angle < 0:
            angle = angle + 360
            
        return angle
    
    def draw_muscle_groups(self, image, landmarks, exercise_type):
        """Egzersiz tipine göre ilgili kas gruplarını çiz"""
        h, w, _ = image.shape
        
        def get_point(landmark_id):
            lm = landmarks[landmark_id]
            return (int(lm.x * w), int(lm.y * h))
        
        if exercise_type == "biceps_curl":
            # Biceps kas grubu - kollar
            points = [
                get_point(mp_pose.PoseLandmark.LEFT_SHOULDER.value),
                get_point(mp_pose.PoseLandmark.LEFT_ELBOW.value),
                get_point(mp_pose.PoseLandmark.LEFT_WRIST.value),
                get_point(mp_pose.PoseLandmark.RIGHT_SHOULDER.value),
                get_point(mp_pose.PoseLandmark.RIGHT_ELBOW.value),
                get_point(mp_pose.PoseLandmark.RIGHT_WRIST.value)
            ]
            # Kolları vurgula
            cv2.line(image, points[0], points[1], (0, 255, 255), 8)
            cv2.line(image, points[1], points[2], (0, 255, 255), 8)
            cv2.line(image, points[3], points[4], (0, 255, 255), 8)
            cv2.line(image, points[4], points[5], (0, 255, 255), 8)
            
        elif exercise_type == "squat":
            # Bacak kasları
            points = [
                get_point(mp_pose.PoseLandmark.LEFT_HIP.value),
                get_point(mp_pose.PoseLandmark.LEFT_KNEE.value),
                get_point(mp_pose.PoseLandmark.LEFT_ANKLE.value),
                get_point(mp_pose.PoseLandmark.RIGHT_HIP.value),
                get_point(mp_pose.PoseLandmark.RIGHT_KNEE.value),
                get_point(mp_pose.PoseLandmark.RIGHT_ANKLE.value)
            ]
            # Bacakları vurgula
            cv2.line(image, points[0], points[1], (255, 0, 255), 8)
            cv2.line(image, points[1], points[2], (255, 0, 255), 8)
            cv2.line(image, points[3], points[4], (255, 0, 255), 8)
            cv2.line(image, points[4], points[5], (255, 0, 255), 8)
            
        elif exercise_type == "shoulder_press":
            # Omuz kasları
            points = [
                get_point(mp_pose.PoseLandmark.LEFT_SHOULDER.value),
                get_point(mp_pose.PoseLandmark.LEFT_ELBOW.value),
                get_point(mp_pose.PoseLandmark.LEFT_WRIST.value),
                get_point(mp_pose.PoseLandmark.RIGHT_SHOULDER.value),
                get_point(mp_pose.PoseLandmark.RIGHT_ELBOW.value),
                get_point(mp_pose.PoseLandmark.RIGHT_WRIST.value),
                get_point(mp_pose.PoseLandmark.LEFT_HIP.value),
                get_point(mp_pose.PoseLandmark.RIGHT_HIP.value)
            ]
            # Omuzları ve kolları vurgula
            cv2.line(image, points[0], points[1], (255, 255, 0), 8)
            cv2.line(image, points[1], points[2], (255, 255, 0), 8)
            cv2.line(image, points[3], points[4], (255, 255, 0), 8)
            cv2.line(image, points[4], points[5], (255, 255, 0), 8)
            cv2.line(image, points[0], points[3], (255, 255, 0), 6)
            
        elif exercise_type == "plank":
            # Core kasları
            core_points = [
                get_point(mp_pose.PoseLandmark.LEFT_SHOULDER.value),
                get_point(mp_pose.PoseLandmark.RIGHT_SHOULDER.value),
                get_point(mp_pose.PoseLandmark.LEFT_HIP.value),
                get_point(mp_pose.PoseLandmark.RIGHT_HIP.value)
            ]
            # Core bölgesini dikdörtgen olarak çiz
            cv2.line(image, core_points[0], core_points[1], (0, 255, 0), 6)
            cv2.line(image, core_points[1], core_points[3], (0, 255, 0), 6)
            cv2.line(image, core_points[3], core_points[2], (0, 255, 0), 6)
            cv2.line(image, core_points[2], core_points[0], (0, 255, 0), 6)
            
        elif exercise_type == "lunges":
            # Bacak ve kalça kasları
            points = [
                get_point(mp_pose.PoseLandmark.LEFT_HIP.value),
                get_point(mp_pose.PoseLandmark.LEFT_KNEE.value),
                get_point(mp_pose.PoseLandmark.LEFT_ANKLE.value),
                get_point(mp_pose.PoseLandmark.RIGHT_HIP.value),
                get_point(mp_pose.PoseLandmark.RIGHT_KNEE.value),
                get_point(mp_pose.PoseLandmark.RIGHT_ANKLE.value)
            ]
            # Bacakları vurgula
            cv2.line(image, points[0], points[1], (128, 0, 255), 8)
            cv2.line(image, points[1], points[2], (128, 0, 255), 8)
            cv2.line(image, points[3], points[4], (128, 0, 255), 8)
            cv2.line(image, points[4], points[5], (128, 0, 255), 8)
    
    def draw_angle(self, image, p1, p2, p3, angle):
        """Açı çizimi"""
        cv2.line(image, p1, p2, (255, 255, 255), 3)
        cv2.line(image, p3, p2, (255, 255, 255), 3)
        cv2.circle(image, p1, 3, (0, 0, 255), cv2.FILLED)
        cv2.circle(image, p1, 6, (255, 0, 0), 2)
        cv2.circle(image, p2, 3, (0, 0, 255), cv2.FILLED)
        cv2.circle(image, p2, 6, (255, 0, 0), 2)
        cv2.circle(image, p3, 3, (0, 0, 255), cv2.FILLED)
        cv2.circle(image, p3, 6, (255, 0, 0), 2)
        cv2.putText(image, str(int(angle)), (p2[0]-20, p2[1]+50), 
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)
    
    def process_biceps_curl(self, image, landmarks, exercise_id):
        """Biceps curl hareketini takip et"""
        if exercise_id not in self.counters:
            self.counters[exercise_id] = 0
            self.directions[exercise_id] = 0
        
        h, w, _ = image.shape
        
        # Sol kol noktaları
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * w,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * h]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * w,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * h]
        
        # Sağ kol noktaları
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w,
                         landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h]
        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * w,
                      landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * h]
        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * w,
                      landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * h]
        
        # Kalça noktaları (form kontrolü için)
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * w,
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * h]
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * w,
                    landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * h]
        
        # Açıları hesapla ve çiz
        angle_left = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
        angle_right = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
        
        self.draw_angle(image, tuple(map(int, left_shoulder)), tuple(map(int, left_elbow)), 
                       tuple(map(int, left_wrist)), angle_left)
        self.draw_angle(image, tuple(map(int, right_shoulder)), tuple(map(int, right_elbow)), 
                       tuple(map(int, right_wrist)), angle_right)
        
        # Form kontrolü için açılar
        angle_left_form = self.calculate_angle(left_elbow, left_shoulder, left_hip)
        angle_right_form = self.calculate_angle(right_elbow, right_shoulder, right_hip)
        
        # Hareket sayımı (sol kol üzerinden)
        per = np.interp(angle_left, (200, 270), (0, 100))
        
        if per == 100:
            if self.directions[exercise_id] == 0:
                self.counters[exercise_id] += 0.5
                self.directions[exercise_id] = 1
        if per == 0:
            if self.directions[exercise_id] == 1:
                self.counters[exercise_id] += 0.5
                self.directions[exercise_id] = 0
        
        # Form kontrolü
        correct_form = True
        feedback = ""
        
        if angle_left_form > 30:
            correct_form = False
            feedback = "Sol kol çok ayrık"
        elif angle_right_form < 320:
            correct_form = False
            feedback = "Sağ kol çok ayrık"
        elif angle_left_form <= 30 and angle_right_form >= 320:
            feedback = "Duruş doğru"
        
        return int(self.counters[exercise_id]), correct_form, feedback
    
    def process_squat(self, image, landmarks, exercise_id):
        """Squat hareketini takip et"""
        if exercise_id not in self.counters:
            self.counters[exercise_id] = 0
            self.directions[exercise_id] = 0
        
        h, w, _ = image.shape
        
        # Sol bacak noktaları
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * w,
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * h]
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * w,
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * h]
        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * w,
                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * h]
        
        # Sağ bacak noktaları
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * w,
                    landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * h]
        right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x * w,
                     landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y * h]
        right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * w,
                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * h]
        
        # Omuz noktaları (form kontrolü için)
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h]
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w,
                         landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h]
        
        # Açıları hesapla ve çiz
        angle_left = self.calculate_angle(left_hip, left_knee, left_ankle)
        angle_right = self.calculate_angle(right_hip, right_knee, right_ankle)
        
        self.draw_angle(image, tuple(map(int, left_hip)), tuple(map(int, left_knee)), 
                       tuple(map(int, left_ankle)), angle_left)
        self.draw_angle(image, tuple(map(int, right_hip)), tuple(map(int, right_knee)), 
                       tuple(map(int, right_ankle)), angle_right)
        
        # Form kontrolü için açılar
        angle_left_form = self.calculate_angle(left_knee, left_hip, left_shoulder)
        angle_right_form = self.calculate_angle(right_knee, right_hip, right_shoulder)
        
        # Hareket sayımı
        per = np.interp(angle_left, (195, 287), (0, 100))
        
        if per == 100:
            if self.directions[exercise_id] == 0:
                self.counters[exercise_id] += 0.5
                self.directions[exercise_id] = 1
        if per == 0:
            if self.directions[exercise_id] == 1:
                self.counters[exercise_id] += 0.5
                self.directions[exercise_id] = 0
        
        # Form kontrolü
        correct_form = True
        feedback = ""
        
        if 70 <= angle_left_form <= 100:
            feedback = "Duruş doğru"
        else:
            correct_form = False
            feedback = "Duruş yanlış - Sırtınızı dik tutun"
        
        if angle_left < 90:
            correct_form = False
            feedback = "Çok fazla eğiliyorsunuz"
        
        return int(self.counters[exercise_id]), correct_form, feedback
    
    def process_shoulder_press(self, image, landmarks, exercise_id):
        """Shoulder press hareketini takip et"""
        if exercise_id not in self.counters:
            self.counters[exercise_id] = 0
            self.directions[exercise_id] = 0
        
        h, w, _ = image.shape
        
        # Sol kol noktaları
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * w,
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * h]
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * w,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * h]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * w,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * h]
        
        # Sağ kol noktaları
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * w,
                    landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * h]
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w,
                         landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h]
        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * w,
                      landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * h]
        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * w,
                      landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * h]
        
        # Açıları hesapla ve çiz
        angle_left = self.calculate_angle(left_hip, left_shoulder, left_elbow)
        angle_right = self.calculate_angle(right_hip, right_shoulder, right_elbow)
        
        self.draw_angle(image, tuple(map(int, left_hip)), tuple(map(int, left_shoulder)), 
                       tuple(map(int, left_elbow)), angle_left)
        self.draw_angle(image, tuple(map(int, right_hip)), tuple(map(int, right_shoulder)), 
                       tuple(map(int, right_elbow)), angle_right)
        
        # Form kontrolü için dirsek açıları
        angle_left_elbow = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
        angle_right_elbow = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
        
        # Hareket sayımı
        per = np.interp(angle_left, (230, 290), (0, 100))
        
        if per == 100:
            if self.directions[exercise_id] == 0:
                self.counters[exercise_id] += 0.5
                self.directions[exercise_id] = 1
        if per == 0:
            if self.directions[exercise_id] == 1:
                self.counters[exercise_id] += 0.5
                self.directions[exercise_id] = 0
        
        # Form kontrolü
        correct_form = True
        feedback = ""
        
        if angle_left_elbow > 80:
            correct_form = False
            feedback = "Sol kol çok açık"
        elif angle_left_elbow < 40:
            correct_form = False
            feedback = "Sol kol çok kapalı"
        elif angle_right_elbow > 310:
            correct_form = False
            feedback = "Sağ kol çok kapalı"
        elif angle_right_elbow < 270:
            correct_form = False
            feedback = "Sağ kol çok açık"
        else:
            feedback = "Duruş doğru"
        
        return int(self.counters[exercise_id]), correct_form, feedback
    
    def process_jumping_jack(self, image, landmarks, exercise_id):
        """Jumping jack hareketini takip et"""
        if exercise_id not in self.counters:
            self.counters[exercise_id] = 0
            self.states[exercise_id] = "down"
        
        h, w, _ = image.shape
        
        # Noktaları al
        left_wrist_y = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * h
        right_wrist_y = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * h
        left_shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h
        
        left_ankle_x = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * w
        right_ankle_x = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * w
        left_hip_x = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * w
        right_hip_x = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * w
        
        # Ayaklar arası mesafe
        feet_distance = abs(left_ankle_x - right_ankle_x)
        hip_distance = abs(left_hip_x - right_hip_x)
        
        # Hareket kontrolü
        correct_form = True
        feedback = ""
        
        # Eller yukarıda ve ayaklar açık mı?
        if (left_wrist_y < left_shoulder_y and right_wrist_y < left_shoulder_y and 
            feet_distance > hip_distance * 1.5):
            if self.states[exercise_id] == "down":
                self.counters[exercise_id] += 1
            self.states[exercise_id] = "up"
            feedback = "Harika!"
        else:
            self.states[exercise_id] = "down"
            if left_wrist_y >= left_shoulder_y:
                feedback = "Elleri yukarı kaldırın"
            elif feet_distance <= hip_distance * 1.5:
                feedback = "Ayakları daha açın"
        
        return int(self.counters[exercise_id]), correct_form, feedback
    
    def process_plank(self, image, landmarks, exercise_id):
        """Plank pozisyonunu kontrol et"""
        if exercise_id not in self.counters:
            self.counters[exercise_id] = 0
            self.states[exercise_id] = {"start_time": None, "accumulated_time": 0}
        
        h, w, _ = image.shape
        
        # Noktaları al
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h]
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * w,
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * h]
        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * w,
                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * h]
        
        # Vücut hizası açısını hesapla
        body_angle = self.calculate_angle(left_shoulder, left_hip, left_ankle)
        
        # Form kontrolü
        correct_form = True
        feedback = ""
        
        if 160 <= body_angle <= 180:
            feedback = "Mükemmel plank pozisyonu!"
            if self.states[exercise_id]["start_time"] is None:
                self.states[exercise_id]["start_time"] = time.time()
        else:
            correct_form = False
            if body_angle < 160:
                feedback = "Kalçanız çok yüksek"
            else:
                feedback = "Kalçanız çok düşük"
            
            # Pozisyon bozulduysa süreyi kaydet ve sıfırla
            if self.states[exercise_id]["start_time"] is not None:
                self.states[exercise_id]["accumulated_time"] += time.time() - self.states[exercise_id]["start_time"]
                self.states[exercise_id]["start_time"] = None
        
        # Toplam süreyi hesapla
        total_time = self.states[exercise_id]["accumulated_time"]
        if self.states[exercise_id]["start_time"] is not None:
            total_time += time.time() - self.states[exercise_id]["start_time"]
        
        self.counters[exercise_id] = int(total_time)  # Saniye cinsinden
        
        return self.counters[exercise_id], correct_form, f"{feedback} - Süre: {int(total_time)}s"
    
    def process_lunges(self, image, landmarks, exercise_id):
        """Lunges hareketini takip et"""
        if exercise_id not in self.counters:
            self.counters[exercise_id] = 0
            self.directions[exercise_id] = 0
        
        h, w, _ = image.shape
        
        # Sol bacak noktaları
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * w,
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * h]
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * w,
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * h]
        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * w,
                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * h]
        
        # Sağ bacak noktaları
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * w,
                    landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * h]
        right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x * w,
                     landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y * h]
        right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * w,
                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * h]
        
        # Açıları hesapla
        angle_left = self.calculate_angle(left_hip, left_knee, left_ankle)
        angle_right = self.calculate_angle(right_hip, right_knee, right_ankle)
        
        self.draw_angle(image, tuple(map(int, left_hip)), tuple(map(int, left_knee)), 
                       tuple(map(int, left_ankle)), angle_left)
        self.draw_angle(image, tuple(map(int, right_hip)), tuple(map(int, right_knee)), 
                       tuple(map(int, right_ankle)), angle_right)
        
        # Hangi bacak önde olduğunu belirle
        if left_knee[0] < right_knee[0]:  # Sol bacak önde
            front_angle = angle_left
            back_angle = angle_right
        else:  # Sağ bacak önde
            front_angle = angle_right
            back_angle = angle_left
        
        # Form kontrolü
        correct_form = True
        feedback = ""
        
        # Ön bacak 90 derece civarında olmalı
        if 80 <= front_angle <= 100:
            if self.directions[exercise_id] == 0:
                self.counters[exercise_id] += 1
                self.directions[exercise_id] = 1
            feedback = "Harika form!"
        elif front_angle > 100:
            self.directions[exercise_id] = 0
            feedback = "Daha derine inin"
        else:
            correct_form = False
            feedback = "Çok fazla iniyorsunuz"
        
        return int(self.counters[exercise_id]), correct_form, feedback
    
    def process_image(self, image_data, exercise_type, exercise_id):
        """Görüntüyü işle ve egzersiz takibi yap"""
        # Base64'ten görüntüyü decode et
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # RGB'ye çevir
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # MediaPipe ile işle
        results = self.pose.process(image_rgb)
        
        count = 0
        correct_form = False
        feedback = "Poz algılanamadı"
        
        if results.pose_landmarks:
            # İskelet çizimini ekle
            mp_drawing.draw_landmarks(
                image, 
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
            # Kas gruplarını çiz
            self.draw_muscle_groups(image, results.pose_landmarks.landmark, exercise_type)
            
            # Egzersiz tipine göre işle
            if exercise_type == "squat":
                count, correct_form, feedback = self.process_squat(
                    image, results.pose_landmarks.landmark, exercise_id
                )
            elif exercise_type == "pushup":
                count, correct_form, feedback = self.process_pushup(
                    image, results.pose_landmarks.landmark, exercise_id
                )
            elif exercise_type == "jumping_jack":
                count, correct_form, feedback = self.process_jumping_jack(
                    image, results.pose_landmarks.landmark, exercise_id
                )
            elif exercise_type == "biceps_curl":
                count, correct_form, feedback = self.process_biceps_curl(
                    image, results.pose_landmarks.landmark, exercise_id
                )
            elif exercise_type == "shoulder_press":
                count, correct_form, feedback = self.process_shoulder_press(
                    image, results.pose_landmarks.landmark, exercise_id
                )
            elif exercise_type == "plank":
                count, correct_form, feedback = self.process_plank(
                    image, results.pose_landmarks.landmark, exercise_id
                )
            elif exercise_type == "lunges":
                count, correct_form, feedback = self.process_lunges(
                    image, results.pose_landmarks.landmark, exercise_id
                )
            else:
                feedback = "Bilinmeyen egzersiz tipi"
        
        # Durum bilgilerini görüntüye ekle
        cv2.rectangle(image, (0, 0), (image.shape[1], 90), (0, 0, 0), -1)
        
        # Tekrar sayısını göster
        cv2.putText(image, f'Tekrar: {count}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Form durumunu göster
        color = (0, 255, 0) if correct_form else (0, 0, 255)
        cv2.putText(image, feedback, (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # İşlenmiş görüntüyü base64'e çevir
        _, buffer = cv2.imencode('.jpg', image)
        processed_image = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "processed_image": processed_image,
            "count": count,
            "correct_form": correct_form,
            "feedback": feedback,
            "exercise_id": exercise_id
        }
    
    def process_pushup(self, image, landmarks, exercise_id):
        """Push-up hareketini takip et"""
        if exercise_id not in self.counters:
            self.counters[exercise_id] = 0
            self.directions[exercise_id] = 0
        
        h, w, _ = image.shape
        
        # Noktaları al
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * w,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * h]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * w,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * h]
        
        # Dirsek açısını hesapla
        angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
        
        self.draw_angle(image, tuple(map(int, left_shoulder)), tuple(map(int, left_elbow)), 
                       tuple(map(int, left_wrist)), angle)
        
        # Push-up durumunu kontrol et
        correct_form = True
        feedback = ""
        
        if angle > 160:
            if self.directions[exercise_id] == 0:
                self.counters[exercise_id] += 1
                self.directions[exercise_id] = 1
            feedback = "Yukarıda"
        elif angle < 90:
            self.directions[exercise_id] = 0
            if angle < 50:
                correct_form = False
                feedback = "Çok fazla iniyorsunuz"
            else:
                feedback = "Aşağıda - Harika!"
        else:
            feedback = "Devam edin"
        
        return int(self.counters[exercise_id]), correct_form, feedback
    

class SocketServer:
    def __init__(self, host='0.0.0.0', port=5000):
        self.host = host
        self.port = port
        self.server_socket = None
        self.clients = []
        self.exercise_tracker = ExerciseTracker()
        
    def start(self):
        """Server'ı başlat"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        
        print(f"Server {self.host}:{self.port} adresinde dinleniyor...")
        
        while True:
            client_socket, address = self.server_socket.accept()
            print(f"Yeni bağlantı: {address}")
            
            client_thread = threading.Thread(
                target=self.handle_client,
                args=(client_socket, address)
            )
            client_thread.start()
    
    def receive_message(self, client_socket):
        """Socket'ten tam mesajı al"""
        try:
            # Önce mesaj boyutunu al (4 byte)
            raw_msglen = self.recvall(client_socket, 4)
            if not raw_msglen:
                return None
            
            msglen = struct.unpack('>I', raw_msglen)[0]
            
            # Mesajı al
            data = self.recvall(client_socket, msglen)
            if not data:
                return None
                
            return json.loads(data.decode('utf-8'))
        except Exception as e:
            print(f"Mesaj alma hatası: {e}")
            return None
    
    def recvall(self, sock, n):
        """n byte veri al"""
        data = bytearray()
        while len(data) < n:
            packet = sock.recv(n - len(data))
            if not packet:
                return None
            data.extend(packet)
        return data
    
    def send_message(self, client_socket, message):
        """Socket'e mesaj gönder"""
        try:
            msg = json.dumps(message).encode('utf-8')
            # Mesaj boyutunu gönder (4 byte)
            client_socket.send(struct.pack('>I', len(msg)))
            # Mesajı gönder
            client_socket.send(msg)
            return True
        except Exception as e:
            print(f"Mesaj gönderme hatası: {e}")
            return False
    
    def handle_client(self, client_socket, address):
        """Client bağlantısını yönet"""
        try:
            while True:
                # Mesajı al
                message = self.receive_message(client_socket)
                if not message:
                    break
                
                print(f"Gelen mesaj tipi: {message.get('type')}")
                
                # Mesaj tipine göre işle
                if message['type'] == 'exercise':
                    # Görüntüyü işle
                    result = self.exercise_tracker.process_image(
                        message['image'],
                        message['exercise_type'],
                        message.get('exercise_id', 'default')
                    )
                    
                    # Sonucu gönder
                    response = {
                        'type': 'result',
                        'data': result,
                        'timestamp': time.time()
                    }
                    
                    if not self.send_message(client_socket, response):
                        break
                        
                elif message['type'] == 'ping':
                    # Ping-pong
                    response = {'type': 'pong', 'timestamp': time.time()}
                    if not self.send_message(client_socket, response):
                        break
                
        except Exception as e:
            print(f"Client hatası ({address}): {e}")
        finally:
            client_socket.close()
            print(f"Bağlantı kapatıldı: {address}")

# Test client örneği
def test_client():
    """Server'ı test etmek için örnek client"""
    import time
    
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(('localhost', 5000))
    
    # Test görüntüsü oluştur (gerçek uygulamada kameradan gelecek)
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    _, buffer = cv2.imencode('.jpg', test_image)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    
    # Test mesajı gönder
    message = {
        'type': 'exercise',
        'image': image_base64,
        'exercise_type': 'squat',
        'exercise_id': 'user123_squat_1'
    }
    
    # Mesajı gönder
    msg = json.dumps(message).encode('utf-8')
    client.send(struct.pack('>I', len(msg)))
    client.send(msg)
    
    # Cevabı al
    raw_msglen = client.recv(4)
    msglen = struct.unpack('>I', raw_msglen)[0]
    response = json.loads(client.recv(msglen).decode('utf-8'))
    
    print("Server cevabı:", response)
    
    client.close()

if __name__ == "__main__":
    # Server'ı başlat
    server = SocketServer(host='0.0.0.0', port=3000)
    
    # Test için thread'de çalıştır
    # server_thread = threading.Thread(target=server.start)
    # server_thread.daemon = True
    # server_thread.start()
    
    # Test client'ı çalıştır
    # time.sleep(2)
    # test_client()
    
    # Normal kullanım için:
    server.start()