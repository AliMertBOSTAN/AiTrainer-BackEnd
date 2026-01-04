# fitness_server.py - Düzeltilmiş ve optimize edilmiş versiyon
import asyncio
import base64
import json
import logging
import math
import os
import platform
import random
import smtplib
import socket
import ssl
import sys
import traceback  # Hata takibi için eklendi
from datetime import datetime, timedelta
from email.mime.text import MIMEText

import bcrypt
import cv2
import jwt
import mediapipe as mp
import numpy as np
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel

# Logging yapılandırması
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

MONGODB_URI = os.environ.get("MONGODB_URI")
mongo_client = None
users_collection = None

if MONGODB_URI:
    try:
        mongo_client = AsyncIOMotorClient(MONGODB_URI)
        db = mongo_client["aitrainer"]
        users_collection = db["users"]
        logger.info("MongoDB Atlas bağlantısı kuruldu.")
    except Exception as exc:
        logger.error(f"MongoDB Atlas bağlantısı kurulamadı: {exc}")
        users_collection = None
else:
    logger.warning("MONGODB_URI tanımlı değil. Auth API veritabanına bağlanamayacak.")

JWT_SECRET = os.environ.get("JWT_SECRET")
if not JWT_SECRET:
    logger.warning("JWT_SECRET tanımlı değil. JWT üretimi başarısız olacaktır.")

MAILER_EMAIL = os.environ.get("MAILER_EMAIL")
MAILER_APP_PASSWORD = os.environ.get("MAILER_APP_PASSWORD")
if not MAILER_EMAIL or not MAILER_APP_PASSWORD:
    logger.warning("MAILER_EMAIL veya MAILER_APP_PASSWORD tanımlı değil. Doğrulama e-postaları gönderilemeyecek.")

VERIFICATION_CODE_TTL_MINUTES = 15
VERIFICATION_CODE_COOLDOWN_SECONDS = 60

# FastAPI uygulamasını başlat
app = FastAPI(title="AiTrainer API")

# --- AUTH MODELLERİ ---
class RegisterBody(BaseModel):
    name: str
    email: str
    password: str

class VerifyEmailBody(BaseModel):
    email: str
    code: str

class ResendCodeBody(BaseModel):
    email: str

class LoginBody(BaseModel):
    email: str
    password: str

# --- YARDIMCI FONKSİYONLAR ---
def normalize_email(value: str) -> str:
    return value.strip().lower()

def create_token(user_id: str) -> str:
    if not JWT_SECRET:
        raise RuntimeError("JWT_SECRET tanımlı değil.")
    payload = {
        "sub": user_id,
        "exp": datetime.utcnow() + timedelta(days=7),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")

def send_verification_email(to_email: str, code: str, name: str) -> None:
    # Geliştirme aşamasında kodu her zaman konsola yazalım, böylece mail gitmese bile test edilebilir.
    logger.info(f"🔑 [DEBUG] DOĞRULAMA KODU ({to_email}): {code}")

    if not MAILER_EMAIL or not MAILER_APP_PASSWORD:
        logger.warning(f"MAILER ayarları eksik. Kod: {code}")
        return

    display_name = name.strip() or "AiTrainer Kullanıcısı"
    body = f"""Merhaba {display_name},

AiTrainer hesabınızı doğrulamak için kodunuz: {code}

Kod 15 dakika geçerli."""
    msg = MIMEText(body)
    msg["Subject"] = "AiTrainer Hesap Doğrulama Kodunuz"
    msg["From"] = MAILER_EMAIL
    msg["To"] = to_email

    # Bağlantı denemesi
    try:
        # Önce standart 587 portunu deneyelim (IPv4 zorlamalı)
        gmail_host = "smtp.gmail.com"
        
        try:
            # Port 587 Denemesi
            context = ssl.create_default_context()
            context.check_hostname = False
            
            # IPv4 çözümleme
            addr_info = socket.getaddrinfo(gmail_host, 587, socket.AF_INET, socket.SOCK_STREAM)
            gmail_ip = addr_info[0][4][0]
            
            with smtplib.SMTP(gmail_ip, 587, timeout=10) as smtp:
                smtp.starttls(context=context)
                smtp.login(MAILER_EMAIL, MAILER_APP_PASSWORD)
                smtp.send_message(msg)
                logger.info(f"Mail başarıyla gönderildi (Port 587): {to_email}")
                return
                
        except Exception as e_587:
            logger.warning(f"Port 587 üzerinden gönderim başarısız ({e_587}). Port 465 deneniyor...")

        # Eğer 587 başarısız olursa 465 (SSL) deneyelim
        try:
            context = ssl.create_default_context()
            context.check_hostname = False
            
            # IPv4 çözümleme
            addr_info = socket.getaddrinfo(gmail_host, 465, socket.AF_INET, socket.SOCK_STREAM)
            gmail_ip = addr_info[0][4][0]
            
            with smtplib.SMTP_SSL(gmail_ip, 465, context=context, timeout=10) as smtp:
                smtp.login(MAILER_EMAIL, MAILER_APP_PASSWORD)
                smtp.send_message(msg)
                logger.info(f"Mail başarıyla gönderildi (Port 465): {to_email}")
                return
                
        except Exception as e_465:
            logger.error(f"Port 465 üzerinden de gönderim başarısız: {e_465}")
            raise e_465

    except Exception as e:
        logger.error(f"❌ Mail gönderilemedi: {e}")
        logger.error("⚠️ DigitalOcean kullanıyorsanız, SMTP portları (25, 465, 587) hesabınızda engelli olabilir.")
        logger.error("💡 ÇÖZÜM: Konsoldaki '[DEBUG] DOĞRULAMA KODU' satırındaki kodu kullanarak testinize devam edebilirsiniz.")
        # Hata fırlatmıyoruz, böylece API 500 hatası vermez ve kullanıcı konsoldaki kodla devam edebilir.
        # raise e 

def get_users_collection():
    if users_collection is None:
        logger.error("MongoDB bağlantısı yapılandırılmadı.")
        raise HTTPException(status_code=500, detail="Veritabanı yapılandırılmadı.")
    return users_collection

async def dispatch_verification_email(to_email: str, code: str, name: str) -> None:
    try:
        await asyncio.to_thread(send_verification_email, to_email, code, name)
    except Exception as exc:
        logger.error(f"Doğrulama e-postası gönderilemedi: {exc}")
        # Mail hatası akışı bozmasın, loglayıp devam edelim
        # raise HTTPException(status_code=500, detail="Doğrulama e-postası gönderilemedi.")

def get_local_ip():
    """Yerel IP adresini otomatik bul"""
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        if local_ip.startswith('127.'):
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                s.connect(('8.8.8.8', 80))
                local_ip = s.getsockname()[0]
            except Exception:
                local_ip = '127.0.0.1'
            finally:
                s.close()
        return local_ip
    except Exception as e:
        logger.error(f"IP adresi bulunamadı: {e}")
        return '127.0.0.1'

# --- MEDIAPIPE & EGZERSİZ ANALİZİ ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class PoseDetector:
    def __init__(self):
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.results = None
        self.lm_list = []
        
    def find_pose(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_rgb)
        
        if self.results.pose_landmarks and draw:
            mp_drawing.draw_landmarks(
                img, 
                self.results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                mp_drawing_styles.get_default_pose_landmarks_style()
            )
        return img
    
    def find_position(self, img):
        self.lm_list = []
        if self.results and self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lm_list.append([id, cx, cy, lm.visibility])
        return self.lm_list
    
    def find_angle(self, img, p1, p2, p3, draw=True):
        if len(self.lm_list) > max(p1, p2, p3):
            x1, y1 = self.lm_list[p1][1:3]
            x2, y2 = self.lm_list[p2][1:3]
            x3, y3 = self.lm_list[p3][1:3]
            
            angle = math.degrees(math.atan2(y3-y2, x3-x2) - math.atan2(y1-y2, x1-x2))
            if angle < 0:
                angle += 360
            if angle > 180:
                angle = 360 - angle
                
            if draw:
                cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
                cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
                cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
                cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
                cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
                cv2.putText(img, str(int(angle)), (x2-50, y2+50), 
                           cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            return angle
        return 0

class ExerciseAnalyzer:
    def __init__(self):
        self.detector = PoseDetector()
        self.exercises = {
            'pushup': self.analyze_pushup,
            'push_up': self.analyze_pushup,
            'squat': self.analyze_squat,
            'biceps_curl': self.analyze_biceps_curl,
            'bicep_curl': self.analyze_biceps_curl,
            'shoulder_press': self.analyze_shoulder_press,
            'plank': self.analyze_plank,
            'lunges': self.analyze_lunge,
            'lunge': self.analyze_lunge,
            'jumping_jack': self.analyze_jumping_jack,
            'jumping_jacks': self.analyze_jumping_jack,
            'situp': self.analyze_situp,
            'sit_up': self.analyze_situp,
            'sit-up': self.analyze_situp
        }
        # Her hareket için önemli olan eklem noktaları (MediaPipe ID'leri)
        self.relevant_landmarks_map = {
            'pushup': [11, 12, 13, 14, 15, 16, 23, 24], # Omuzlar, Dirsekler, Bilekler, Kalçalar
            'squat': [11, 12, 23, 24, 25, 26, 27, 28], # Omuzlar, Kalçalar, Dizler, Ayak Bilekleri
            'biceps_curl': [11, 12, 13, 14, 15, 16], # Omuzlar, Dirsekler, Bilekler
            'shoulder_press': [11, 12, 13, 14, 15, 16, 23, 24],
            'plank': [11, 23, 25, 27, 7], # Omuz, Kalça, Diz, Ayak Bileği, Kulak
            'lunges': [23, 24, 25, 26, 27, 28, 11, 12],
            'jumping_jack': [11, 12, 13, 14, 23, 24, 27, 28],
            'situp': [11, 23, 25, 27, 7],
        }
        self.exercise_state = {}
        
    def analyze(self, img, exercise_type, exercise_id):
        try:
            output_img = img.copy()
            output_img = self.detector.find_pose(output_img)
            lm_list = self.detector.find_position(output_img)
            
            exercise_type = exercise_type.lower().replace('-', '_').replace(' ', '_')
            
            # Sonuç objesini hazırla
            result = {
                'count': 0,
                'correct_form': False,
                'feedback': '',
                'landmarks': []
            }

            # İlgili hareketin önemli noktalarını al
            relevant_ids = self.relevant_landmarks_map.get(exercise_type, [])

            if exercise_type in self.exercises and lm_list:
                if exercise_id not in self.exercise_state:
                    self.exercise_state[exercise_id] = {
                        'count': 0,
                        'direction': 0,
                        'form_errors': []
                    }
                
                # Egzersiz analizini çalıştır
                analysis_result = self.exercises[exercise_type](output_img, lm_list, exercise_id)
                result.update(analysis_result)
                
                # Landmarkları (noktaları) sonuca ekle
                # lm_list formatı: [id, cx, cy, visibility]
                result['landmarks'] = [
                    {
                        'id': lm[0], 
                        'x': lm[1], 
                        'y': lm[2], 
                        'visibility': lm[3],
                        'is_relevant': lm[0] in relevant_ids # Bu nokta bu hareket için önemli mi?
                    } 
                    for lm in lm_list
                ]
                
                return output_img, result
            
            result['feedback'] = f'Egzersiz tipi tanınamadı ({exercise_type}) veya vücut tespit edilemedi'
            return output_img, result
            
        except Exception as e:
            logger.error(f"Analiz sırasında hata: {e}")
            traceback.print_exc()
            return img, {'count': 0, 'correct_form': False, 'feedback': 'Analiz hatası', 'landmarks': []}
    
    # --- Egzersiz Analiz Fonksiyonları (Kısaltıldı, mantık aynı) ---
    def analyze_pushup(self, img, lm_list, exercise_id):
        state = self.exercise_state[exercise_id]
        feedback = []
        correct_form = True
        left_arm_angle = self.detector.find_angle(img, 11, 13, 15)
        right_arm_angle = self.detector.find_angle(img, 12, 14, 16)
        body_angle = self.detector.find_angle(img, 11, 23, 25)
        
        if abs(body_angle - 180) > 30:
            feedback.append("Vücudunuzu düz tutun")
            correct_form = False
            
        avg_arm_angle = (left_arm_angle + right_arm_angle) / 2
        if avg_arm_angle > 160:
            if state['direction'] == 0:
                state['count'] += 0.5
                state['direction'] = 1
        elif avg_arm_angle < 90:
            if state['direction'] == 1:
                state['count'] += 0.5
                state['direction'] = 0
                
        if not feedback: feedback.append("Harika gidiyorsunuz!")
        return {'count': int(state['count']), 'correct_form': correct_form, 'feedback': ' - '.join(feedback)}

    def analyze_squat(self, img, lm_list, exercise_id):
        state = self.exercise_state[exercise_id]
        feedback = []
        correct_form = True
        left_knee_angle = self.detector.find_angle(img, 23, 25, 27)
        right_knee_angle = self.detector.find_angle(img, 24, 26, 28)
        back_angle = self.detector.find_angle(img, 11, 23, 25)
        
        if back_angle < 140:
            feedback.append("Sırtınızı dik tutun")
            correct_form = False
            
        avg_knee_angle = (left_knee_angle + right_knee_angle) / 2
        if avg_knee_angle > 170:
            if state['direction'] == 0:
                state['count'] += 0.5
                state['direction'] = 1
        elif avg_knee_angle < 90:
            if state['direction'] == 1:
                state['count'] += 0.5
                state['direction'] = 0
                
        if not feedback: feedback.append("Mükemmel form!")
        return {'count': int(state['count']), 'correct_form': correct_form, 'feedback': ' - '.join(feedback)}

    def analyze_biceps_curl(self, img, lm_list, exercise_id):
        state = self.exercise_state[exercise_id]
        feedback = []
        correct_form = True
        left_arm_angle = self.detector.find_angle(img, 11, 13, 15)
        right_arm_angle = self.detector.find_angle(img, 12, 14, 16)
        left_elbow_angle = self.detector.find_angle(img, 13, 11, 23)
        right_elbow_angle = self.detector.find_angle(img, 14, 12, 24)
        
        if left_elbow_angle > 30:
            feedback.append("Sol dirseğinizi vücudunuza yakın tutun")
            correct_form = False
        if right_elbow_angle < 330:
            feedback.append("Sağ dirseğinizi vücudunuza yakın tutun")
            correct_form = False
            
        avg_angle = (left_arm_angle + right_arm_angle) / 2
        if avg_angle > 160:
            if state['direction'] == 0:
                state['count'] += 0.5
                state['direction'] = 1
        elif avg_angle < 50:
            if state['direction'] == 1:
                state['count'] += 0.5
                state['direction'] = 0
                
        if not feedback: feedback.append("Doğru form!")
        return {'count': int(state['count']), 'correct_form': correct_form, 'feedback': ' - '.join(feedback)}

    def analyze_shoulder_press(self, img, lm_list, exercise_id):
        state = self.exercise_state[exercise_id]
        feedback = []
        correct_form = True
        left_shoulder_angle = self.detector.find_angle(img, 23, 11, 13)
        right_shoulder_angle = self.detector.find_angle(img, 24, 12, 14)
        left_elbow_angle = self.detector.find_angle(img, 11, 13, 15)
        
        if left_elbow_angle < 70 or left_elbow_angle > 110:
            feedback.append("Sol dirseğinizi 90 derece açıda tutun")
            correct_form = False
            
        avg_shoulder_angle = (left_shoulder_angle + right_shoulder_angle) / 2
        if avg_shoulder_angle > 160:
            if state['direction'] == 0:
                state['count'] += 0.5
                state['direction'] = 1
        elif avg_shoulder_angle < 90:
            if state['direction'] == 1:
                state['count'] += 0.5
                state['direction'] = 0
                
        if not feedback: feedback.append("Harika form!")
        return {'count': int(state['count']), 'correct_form': correct_form, 'feedback': ' - '.join(feedback)}

    def analyze_plank(self, img, lm_list, exercise_id):
        state = self.exercise_state[exercise_id]
        feedback = []
        correct_form = True
        body_angle = self.detector.find_angle(img, 11, 23, 27)
        hip_angle = self.detector.find_angle(img, 11, 23, 25)
        neck_angle = self.detector.find_angle(img, 7, 11, 23) if len(lm_list) > 7 else 160
        
        if abs(body_angle - 180) > 25:
            feedback.append("Vücudunuzu düz bir çizgi halinde tutun")
            correct_form = False
        if hip_angle < 150:
            feedback.append("Kalçanız çok aşağıda")
            correct_form = False
        elif hip_angle > 210:
            feedback.append("Kalçanız çok yukarıda")
            correct_form = False
        if neck_angle < 140:
            feedback.append("Başınızı nötr pozisyonda tutun")
            correct_form = False
            
        state['count'] += 1/30
        if not feedback: feedback.append(f"Mükemmel form! {int(state['count'])} saniye")
        return {'count': int(state['count']), 'correct_form': correct_form, 'feedback': ' - '.join(feedback)}

    def analyze_situp(self, img, lm_list, exercise_id):
        state = self.exercise_state[exercise_id]
        feedback = []
        correct_form = True
        torso_angle = self.detector.find_angle(img, 11, 23, 25)
        leg_angle = self.detector.find_angle(img, 23, 25, 27)
        neck_angle = self.detector.find_angle(img, 7, 11, 23) if len(lm_list) > 7 else 150
        
        if leg_angle < 80 or leg_angle > 120:
            feedback.append("Dizlerinizi 90 derece açıda tutun")
            correct_form = False
        if neck_angle < 120:
            feedback.append("Boynunuzu zorlamayın, nötr tutun")
            correct_form = False
            
        if torso_angle < 100:
            if state['direction'] == 0:
                state['count'] += 0.5
                state['direction'] = 1
        elif torso_angle > 150:
            if state['direction'] == 1:
                state['count'] += 0.5
                state['direction'] = 0
                
        if not feedback: feedback.append("Harika form!")
        return {'count': int(state['count']), 'correct_form': correct_form, 'feedback': ' - '.join(feedback)}

    def analyze_jumping_jack(self, img, lm_list, exercise_id):
        state = self.exercise_state[exercise_id]
        feedback = []
        correct_form = True
        left_arm_angle = self.detector.find_angle(img, 23, 11, 13)
        right_arm_angle = self.detector.find_angle(img, 24, 12, 14)
        hip_width = abs(lm_list[23][1] - lm_list[24][1])
        ankle_width = abs(lm_list[27][1] - lm_list[28][1])
        body_straightness = self.detector.find_angle(img, 11, 23, 27)
        
        if abs(body_straightness - 180) > 20:
            feedback.append("Vücudunuzu dik tutun")
            correct_form = False
            
        avg_arm_angle = (left_arm_angle + right_arm_angle) / 2
        if avg_arm_angle > 160 and ankle_width > hip_width * 2:
            if state['direction'] == 0:
                state['count'] += 0.5
                state['direction'] = 1
        elif avg_arm_angle < 30 and ankle_width < hip_width * 1.2:
            if state['direction'] == 1:
                state['count'] += 0.5
                state['direction'] = 0
                
        if not feedback: feedback.append("Mükemmel ritim!")
        return {'count': int(state['count']), 'correct_form': correct_form, 'feedback': ' - '.join(feedback)}

    def analyze_lunge(self, img, lm_list, exercise_id):
        state = self.exercise_state[exercise_id]
        feedback = []
        correct_form = True
        left_knee_y = lm_list[25][2]
        right_knee_y = lm_list[26][2]
        
        if left_knee_y > right_knee_y:
            front_knee_angle = self.detector.find_angle(img, 23, 25, 27)
            back_knee_angle = self.detector.find_angle(img, 24, 26, 28)
        else:
            front_knee_angle = self.detector.find_angle(img, 24, 26, 28)
            back_knee_angle = self.detector.find_angle(img, 23, 25, 27)
        
        torso_angle = self.detector.find_angle(img, 11, 23, 24)
        
        if front_knee_angle < 70:
            feedback.append("Ön diziniz çok öne gitti")
            correct_form = False
        elif front_knee_angle > 110 and state['direction'] == 1:
            feedback.append("Ön dizinizi daha fazla bükün")
            correct_form = False
        if back_knee_angle < 80 and state['direction'] == 1:
            feedback.append("Arka diziniz yere çok yakın")
            correct_form = False
        if abs(torso_angle - 180) > 25:
            feedback.append("Gövdenizi dik tutun")
            correct_form = False
            
        if front_knee_angle < 100:
            if state['direction'] == 0:
                state['count'] += 0.5
                state['direction'] = 1
        elif front_knee_angle > 160:
            if state['direction'] == 1:
                state['count'] += 0.5
                state['direction'] = 0
                
        if not feedback: feedback.append("Mükemmel lunge!")
        return {'count': int(state['count']), 'correct_form': correct_form, 'feedback': ' - '.join(feedback)}

# --- AUTH ENDPOINTS ---
@app.post("/api/auth/register")
async def register(body: RegisterBody):
    users = get_users_collection()
    email = normalize_email(body.email)
    name = body.name.strip()
    password = body.password

    if not email or not name or not password:
        raise HTTPException(status_code=400, detail="Tüm alanlar gerekli.")

    existing = await users.find_one({"email": email})
    if existing and existing.get("isVerified"):
        raise HTTPException(status_code=409, detail="Bu e-posta zaten kayıtlı.")

    code = f"{random.randint(0, 999999):06d}"
    hashed_code = bcrypt.hashpw(code.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    password_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    now = datetime.utcnow()
    expires = now + timedelta(minutes=VERIFICATION_CODE_TTL_MINUTES)

    user_doc = {
        "name": name,
        "email": email,
        "passwordHash": password_hash,
        "isVerified": False,
        "verificationCodeHash": hashed_code,
        "verificationCodeExpiresAt": expires,
        "lastVerificationEmailSentAt": now,
        "updatedAt": now,
    }

    if existing:
        await users.update_one({"_id": existing["_id"]}, {"$set": user_doc})
    else:
        user_doc["createdAt"] = now
        await users.insert_one(user_doc)

    await dispatch_verification_email(email, code, name)
    return {"message": "Doğrulama kodu e-posta adresinize gönderildi."}

@app.post("/api/auth/verify-email")
async def verify_email(body: VerifyEmailBody):
    users = get_users_collection()
    email = normalize_email(body.email)
    code = body.code.strip()

    existing = await users.find_one({"email": email})
    if not existing:
        raise HTTPException(status_code=404, detail="Kullanıcı bulunamadı.")
    if existing.get("isVerified"):
        return {"message": "E-posta zaten doğrulanmış."}

    stored_hash = existing.get("verificationCodeHash")
    expires_at = existing.get("verificationCodeExpiresAt")
    
    if not stored_hash or not expires_at or expires_at < datetime.utcnow():
        raise HTTPException(status_code=400, detail="Kod geçersiz veya süresi dolmuş.")

    if not bcrypt.checkpw(code.encode("utf-8"), stored_hash.encode("utf-8")):
        raise HTTPException(status_code=400, detail="Doğrulama kodu geçersiz.")

    await users.update_one(
        {"_id": existing["_id"]},
        {"$set": {"isVerified": True, "verificationCodeHash": None, "verificationCodeExpiresAt": None, "verifiedAt": datetime.utcnow()}}
    )
    return {"message": "E-posta başarıyla doğrulandı."}

@app.post("/api/auth/resend-code")
async def resend_code(body: ResendCodeBody):
    users = get_users_collection()
    email = normalize_email(body.email)
    existing = await users.find_one({"email": email})
    
    if not existing:
        raise HTTPException(status_code=404, detail="Kullanıcı bulunamadı.")
    if existing.get("isVerified"):
        raise HTTPException(status_code=400, detail="E-posta zaten doğrulanmış.")

    code = f"{random.randint(0, 999999):06d}"
    hashed_code = bcrypt.hashpw(code.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    expires = datetime.utcnow() + timedelta(minutes=VERIFICATION_CODE_TTL_MINUTES)

    await users.update_one(
        {"_id": existing["_id"]},
        {"$set": {"verificationCodeHash": hashed_code, "verificationCodeExpiresAt": expires, "lastVerificationEmailSentAt": datetime.utcnow()}}
    )
    await dispatch_verification_email(email, code, existing.get("name", ""))
    return {"message": "Yeni doğrulama kodu gönderildi."}

@app.post("/api/auth/login")
async def login(body: LoginBody):
    users = get_users_collection()
    email = normalize_email(body.email)
    password = body.password

    existing = await users.find_one({"email": email})
    if not existing or not existing.get("passwordHash"):
        raise HTTPException(status_code=401, detail="E-posta veya şifre yanlış.")
    if not existing.get("isVerified"):
        raise HTTPException(status_code=403, detail="Hesabınız henüz doğrulanmamış.")

    if not bcrypt.checkpw(password.encode("utf-8"), existing["passwordHash"].encode("utf-8")):
        raise HTTPException(status_code=401, detail="E-posta veya şifre yanlış.")

    token = create_token(str(existing["_id"]))
    return {
        "token": token,
        "user": {"id": str(existing["_id"]), "name": existing.get("name"), "email": existing.get("email")}
    }

# --- WEBSOCKET ENDPOINT ---
analyzer = ExerciseAnalyzer()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    client_ip = websocket.client.host
    logger.info(f"WebSocket bağlantısı: {client_ip}")
    
    try:
        # Hoşgeldin mesajı
        welcome = {
            'type': 'connected',
            'message': 'Fitness AI Server\'a hoş geldiniz!',
            'supported_exercises': list(analyzer.exercises.keys()),
            'timestamp': datetime.now().timestamp()
        }
        await websocket.send_text(json.dumps(welcome))
        
        while True:
            try:
                data_str = await websocket.receive_text()
                data = json.loads(data_str)
                
                if data.get('type') == 'exercise':
                    # Görüntü işleme
                    image_data = base64.b64decode(data['image'])
                    nparr = np.frombuffer(image_data, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if img is None:
                        logger.warning("Görüntü decode edilemedi (img is None).")
                        continue
                    
                    img = cv2.resize(img, (640, 480))
                    exercise_type = data.get('exercise_type', 'unknown')
                    exercise_id = data.get('exercise_id', 'default')
                    
                    # Analiz ve çizim
                    processed_img, result = analyzer.analyze(img, exercise_type, exercise_id)
                    
                    # Geri gönderme
                    _, buffer = cv2.imencode('.jpg', processed_img, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    processed_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    response = {
                        'type': 'result',
                        'data': {
                            'processed_image': processed_base64,
                            'count': result['count'],
                            'correct_form': result['correct_form'],
                            'feedback': result['feedback'],
                            'landmarks': result.get('landmarks', []), # Landmark verisi eklendi
                            'exercise_id': exercise_id
                        },
                        'timestamp': datetime.now().timestamp()
                    }
                    await websocket.send_text(json.dumps(response))
            
            except WebSocketDisconnect:
                logger.info(f"WebSocket bağlantısı kesildi (Client): {client_ip}")
                break
            except json.JSONDecodeError:
                logger.error("Geçersiz JSON verisi.")
            except Exception as e:
                logger.error(f"WebSocket döngüsü içinde hata: {e}")
                traceback.print_exc() # Hatayı konsola detaylı yazdır
                
    except Exception as e:
        logger.error(f"WebSocket ana hatası: {e}")
        traceback.print_exc()

def main():
    port = 8000
    # Check for port argument
    for arg in sys.argv[1:]:
        if arg.isdigit():
            port = int(arg)
            
    host = get_local_ip()
    
    # Dinamik URL'leri oluştur
    base_url = f"http://{host}:{port}"
    ws_url = f"ws://{host}:{port}/ws"
    auth_url = f"{base_url}/api/auth"
    
    print("\n" + "="*60)
    print("🚀 FITNESS AI SERVER BAŞLATILDI")
    print("="*60)
    print(f"📍 Sunucu IP Adresi : {host}")
    print(f"🔌 Port Numarası    : {port}")
    print("-" * 60)
    print("📱 REACT NATIVE BAĞLANTI BİLGİLERİ:")
    print(f"   • API Base URL   : {auth_url}")
    print(f"   • WebSocket URL  : {ws_url}")
    print("-" * 60)
    print("🔑 KULLANILABİLİR ENDPOINTLER:")
    print(f"   • Kayıt Ol       : POST {auth_url}/register")
    print(f"   • Giriş Yap      : POST {auth_url}/login")
    print(f"   • Mail Doğrula   : POST {auth_url}/verify-email")
    print(f"   • Kod Tekrar     : POST {auth_url}/resend-code")
    print("-" * 60)
    print(f"📄 API Dokümantasyonu: {base_url}/docs")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
