import asyncio
import websockets
import json
import base64
import cv2
import numpy as np
import mediapipe as mp
import math
from datetime import datetime
import io
from PIL import Image
import logging

# Logging yapılandırması
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PoseDetector:
    def __init__(self, mode=False, upper_body=False, smooth_landmarks=True, 
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mode = mode
        self.upper_body = upper_body
        self.smooth_landmarks = smooth_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=self.mode,
            model_complexity=1,
            smooth_landmarks=self.smooth_landmarks,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        
    def find_pose(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_rgb)
        
        if self.results.pose_landmarks and draw:
            self.mp_draw.draw_landmarks(
                img, self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                self.mp_draw.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )
        return img
    
    def find_position(self, img, draw=True):
        self.lm_list = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lm_list.append([id, cx, cy, lm.visibility])
                
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lm_list
    
    def find_angle(self, img, p1, p2, p3, draw=True):
        # Nokta pozisyonlarını al
        if len(self.lm_list) > max(p1, p2, p3):
            x1, y1 = self.lm_list[p1][1:3]
            x2, y2 = self.lm_list[p2][1:3]
            x3, y3 = self.lm_list[p3][1:3]
            
            # Açıyı hesapla
            angle = math.degrees(math.atan2(y3-y2, x3-x2) - math.atan2(y1-y2, x1-x2))
            if angle < 0:
                angle += 360
            if angle > 180:
                angle = 360 - angle
                
            # Görselleştirme
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
            'squat': self.analyze_squat,
            'biceps_curl': self.analyze_biceps_curl,
            'shoulder_press': self.analyze_shoulder_press,
            'plank': self.analyze_plank,
            'lunges': self.analyze_lunges,
            'jumping_jack': self.analyze_jumping_jack
        }
        self.exercise_state = {}  # Her egzersiz için durum bilgisi
        
    def analyze(self, img, exercise_type, exercise_id):
        # Resmi kopyala
        output_img = img.copy()
        
        # Pose tespiti
        output_img = self.detector.find_pose(output_img)
        lm_list = self.detector.find_position(output_img, draw=False)
        
        # Egzersiz tipine göre analiz
        if exercise_type in self.exercises and lm_list:
            if exercise_id not in self.exercise_state:
                self.exercise_state[exercise_id] = {
                    'count': 0,
                    'direction': 0,
                    'form_errors': []
                }
            
            result = self.exercises[exercise_type](output_img, lm_list, exercise_id)
            return output_img, result
        
        return output_img, {
            'count': 0,
            'correct_form': False,
            'feedback': 'Egzersiz tipi tanınamadı veya vücut tespit edilemedi'
        }
    
    def analyze_pushup(self, img, lm_list, exercise_id):
        state = self.exercise_state[exercise_id]
        feedback = []
        correct_form = True
        
        # Sol kol açısı (11-13-15: omuz-dirsek-bilek)
        left_arm_angle = self.detector.find_angle(img, 11, 13, 15)
        # Sağ kol açısı (12-14-16)
        right_arm_angle = self.detector.find_angle(img, 12, 14, 16)
        
        # Vücut düzlüğü kontrolü (11-23-25: omuz-kalça-diz)
        body_angle = self.detector.find_angle(img, 11, 23, 25)
        
        # Form kontrolü
        if abs(body_angle - 180) > 30:
            feedback.append("Vücudunuzu düz tutun")
            correct_form = False
            
        # Push-up sayımı (kol açısına göre)
        avg_arm_angle = (left_arm_angle + right_arm_angle) / 2
        
        if avg_arm_angle > 160:
            if state['direction'] == 0:
                state['count'] += 0.5
                state['direction'] = 1
        elif avg_arm_angle < 90:
            if state['direction'] == 1:
                state['count'] += 0.5
                state['direction'] = 0
                
        # Feedback oluştur
        if not feedback:
            feedback.append("Harika gidiyorsunuz!")
            
        return {
            'count': int(state['count']),
            'correct_form': correct_form,
            'feedback': ' - '.join(feedback)
        }
    
    def analyze_squat(self, img, lm_list, exercise_id):
        state = self.exercise_state[exercise_id]
        feedback = []
        correct_form = True
        
        # Bacak açıları
        left_knee_angle = self.detector.find_angle(img, 23, 25, 27)  # kalça-diz-ayak bileği
        right_knee_angle = self.detector.find_angle(img, 24, 26, 28)
        
        # Sırt düzlüğü
        back_angle = self.detector.find_angle(img, 11, 23, 25)  # omuz-kalça-diz
        
        # Form kontrolü
        if back_angle < 140:
            feedback.append("Sırtınızı dik tutun")
            correct_form = False
            
        # Squat sayımı
        avg_knee_angle = (left_knee_angle + right_knee_angle) / 2
        
        if avg_knee_angle > 170:
            if state['direction'] == 0:
                state['count'] += 0.5
                state['direction'] = 1
        elif avg_knee_angle < 90:
            if state['direction'] == 1:
                state['count'] += 0.5
                state['direction'] = 0
                
        if not feedback:
            feedback.append("Mükemmel form!")
            
        return {
            'count': int(state['count']),
            'correct_form': correct_form,
            'feedback': ' - '.join(feedback)
        }
    
    def analyze_biceps_curl(self, img, lm_list, exercise_id):
        state = self.exercise_state[exercise_id]
        feedback = []
        correct_form = True
        
        # Kol açıları
        left_arm_angle = self.detector.find_angle(img, 11, 13, 15)
        right_arm_angle = self.detector.find_angle(img, 12, 14, 16)
        
        # Dirsek pozisyonu kontrolü
        left_elbow_angle = self.detector.find_angle(img, 13, 11, 23)
        right_elbow_angle = self.detector.find_angle(img, 14, 12, 24)
        
        if left_elbow_angle > 30:
            feedback.append("Sol dirseğinizi vücudunuza yakın tutun")
            correct_form = False
        if right_elbow_angle < 330:
            feedback.append("Sağ dirseğinizi vücudunuza yakın tutun")
            correct_form = False
            
        # Biceps curl sayımı
        avg_angle = (left_arm_angle + right_arm_angle) / 2
        
        if avg_angle > 160:
            if state['direction'] == 0:
                state['count'] += 0.5
                state['direction'] = 1
        elif avg_angle < 50:
            if state['direction'] == 1:
                state['count'] += 0.5
                state['direction'] = 0
                
        if not feedback:
            feedback.append("Doğru form!")
            
        return {
            'count': int(state['count']),
            'correct_form': correct_form,
            'feedback': ' - '.join(feedback)
        }
    
    def analyze_shoulder_press(self, img, lm_list, exercise_id):
        state = self.exercise_state[exercise_id]
        feedback = []
        correct_form = True
        
        # Omuz açıları
        left_shoulder_angle = self.detector.find_angle(img, 23, 11, 13)
        right_shoulder_angle = self.detector.find_angle(img, 24, 12, 14)
        
        # Dirsek açıları
        left_elbow_angle = self.detector.find_angle(img, 11, 13, 15)
        right_elbow_angle = self.detector.find_angle(img, 12, 14, 16)
        
        # Form kontrolü
        if left_elbow_angle < 70 or left_elbow_angle > 110:
            feedback.append("Sol dirseğinizi 90 derece açıda tutun")
            correct_form = False
            
        # Shoulder press sayımı
        avg_shoulder_angle = (left_shoulder_angle + right_shoulder_angle) / 2
        
        if avg_shoulder_angle > 160:
            if state['direction'] == 0:
                state['count'] += 0.5
                state['direction'] = 1
        elif avg_shoulder_angle < 90:
            if state['direction'] == 1:
                state['count'] += 0.5
                state['direction'] = 0
                
        if not feedback:
            feedback.append("Harika form!")
            
        return {
            'count': int(state['count']),
            'correct_form': correct_form,
            'feedback': ' - '.join(feedback)
        }
    
    def analyze_plank(self, img, lm_list, exercise_id):
        state = self.exercise_state[exercise_id]
        feedback = []
        correct_form = True
        
        # Vücut düzlüğü
        body_angle = self.detector.find_angle(img, 11, 23, 27)  # omuz-kalça-ayak bileği
        
        # Kalça pozisyonu
        hip_angle = self.detector.find_angle(img, 11, 23, 25)  # omuz-kalça-diz
        
        if abs(body_angle - 180) > 20:
            feedback.append("Vücudunuzu düz bir çizgi halinde tutun")
            correct_form = False
            
        if hip_angle < 160:
            feedback.append("Kalçanızı düşürün")
            correct_form = False
        elif hip_angle > 200:
            feedback.append("Kalçanızı yükseltin")
            correct_form = False
            
        # Plank süresi (basit sayaç)
        state['count'] += 1/30  # 30 FPS varsayımı
        
        if not feedback:
            feedback.append(f"Harika! {int(state['count'])} saniye")
            
        return {
            'count': int(state['count']),
            'correct_form': correct_form,
            'feedback': ' - '.join(feedback)
        }
    
    def analyze_lunges(self, img, lm_list, exercise_id):
        state = self.exercise_state[exercise_id]
        feedback = []
        correct_form = True
        
        # Ön bacak açısı
        front_knee_angle = self.detector.find_angle(img, 23, 25, 27)
        # Arka bacak açısı
        back_knee_angle = self.detector.find_angle(img, 24, 26, 28)
        
        # Form kontrolü
        if front_knee_angle < 80:
            feedback.append("Ön diziniz ayak parmağınızı geçmesin")
            correct_form = False
            
        # Lunge sayımı
        if front_knee_angle < 100:
            if state['direction'] == 0:
                state['count'] += 0.5
                state['direction'] = 1
        elif front_knee_angle > 160:
            if state['direction'] == 1:
                state['count'] += 0.5
                state['direction'] = 0
                
        if not feedback:
            feedback.append("Mükemmel lunge!")
            
        return {
            'count': int(state['count']),
            'correct_form': correct_form,
            'feedback': ' - '.join(feedback)
        }
    
    def analyze_jumping_jack(self, img, lm_list, exercise_id):
        state = self.exercise_state[exercise_id]
        feedback = []
        correct_form = True
        
        # Kol pozisyonu
        left_arm_angle = self.detector.find_angle(img, 23, 11, 13)
        right_arm_angle = self.detector.find_angle(img, 24, 12, 14)
        
        # Bacak açıklığı (kalça genişliği)
        hip_distance = abs(lm_list[23][1] - lm_list[24][1])
        
        # Jumping jack sayımı
        avg_arm_angle = (left_arm_angle + right_arm_angle) / 2
        
        if avg_arm_angle > 150 and hip_distance > 100:
            if state['direction'] == 0:
                state['count'] += 0.5
                state['direction'] = 1
        elif avg_arm_angle < 90 and hip_distance < 50:
            if state['direction'] == 1:
                state['count'] += 0.5
                state['direction'] = 0
                
        if not feedback:
            feedback.append("Harika tempo!")
            
        return {
            'count': int(state['count']),
            'correct_form': correct_form,
            'feedback': ' - '.join(feedback)
        }

class WebSocketServer:
    def __init__(self):
        self.analyzer = ExerciseAnalyzer()
        self.clients = set()
        
    async def process_frame(self, websocket, message):
        try:
            data = json.loads(message)
            
            if data.get('type') == 'exercise':
                # Base64 görüntüyü decode et
                image_data = base64.b64decode(data['image'])
                nparr = np.frombuffer(image_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is None:
                    raise ValueError("Görüntü decode edilemedi")
                
                # Egzersiz analizi
                exercise_type = data.get('exercise_type', 'unknown')
                exercise_id = data.get('exercise_id', 'default')
                
                processed_img, result = self.analyzer.analyze(img, exercise_type, exercise_id)
                
                # İşlenmiş görüntüyü base64'e çevir
                _, buffer = cv2.imencode('.jpg', processed_img)
                processed_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # Sonucu gönder
                response = {
                    'type': 'result',
                    'data': {
                        'processed_image': processed_base64,
                        'count': result['count'],
                        'correct_form': result['correct_form'],
                        'feedback': result['feedback'],
                        'exercise_id': exercise_id
                    },
                    'timestamp': datetime.now().timestamp()
                }
                
                await websocket.send(json.dumps(response))
                
        except Exception as e:
            logger.error(f"Frame işleme hatası: {str(e)}")
            error_response = {
                'type': 'error',
                'message': str(e),
                'timestamp': datetime.now().timestamp()
            }
            await websocket.send(json.dumps(error_response))
    
    async def handle_client(self, websocket, path):
        # Yeni client ekleme
        self.clients.add(websocket)
        logger.info(f"Yeni client bağlandı: {websocket.remote_address}")
        
        try:
            # Hoşgeldin mesajı
            welcome = {
                'type': 'connected',
                'message': 'Pose estimation server\'a bağlandınız',
                'timestamp': datetime.now().timestamp()
            }
            await websocket.send(json.dumps(welcome))
            
            # Mesajları dinle
            async for message in websocket:
                await self.process_frame(websocket, message)
                
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client bağlantısı kapandı: {websocket.remote_address}")
        except Exception as e:
            logger.error(f"Client hatası: {str(e)}")
        finally:
            # Client'ı kaldır
            self.clients.remove(websocket)
    
    async def start(self, host='0.0.0.0', port=3000):
        logger.info(f"WebSocket server başlatılıyor: {host}:{port}")
        async with websockets.serve(self.handle_client, host, port):
            logger.info("Server hazır, bağlantı bekleniyor...")
            await asyncio.Future()  # Sonsuza kadar çalış

# HTTP API (Alternatif)
from aiohttp import web

class HTTPServer:
    def __init__(self):
        self.analyzer = ExerciseAnalyzer()
        
    async def process_request(self, request):
        try:
            data = await request.json()
            
            # Base64 görüntüyü decode et
            image_data = base64.b64decode(data['image'])
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Egzersiz analizi
            exercise_type = data.get('exercise_type', 'unknown')
            exercise_id = data.get('exercise_id', 'default')
            
            processed_img, result = self.analyzer.analyze(img, exercise_type, exercise_id)
            
            # İşlenmiş görüntüyü base64'e çevir
            _, buffer = cv2.imencode('.jpg', processed_img)
            processed_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Sonucu döndür
            response = {
                'type': 'result',
                'data': {
                    'processed_image': processed_base64,
                    'count': result['count'],
                    'correct_form': result['correct_form'],
                    'feedback': result['feedback'],
                    'exercise_id': exercise_id
                },
                'timestamp': datetime.now().timestamp()
            }
            
            return web.json_response(response)
            
        except Exception as e:
            logger.error(f"HTTP işleme hatası: {str(e)}")
            return web.json_response({
                'type': 'error',
                'message': str(e)
            }, status=500)
    
    async def start_http(self, host='0.0.0.0', port=3000):
        app = web.Application()
        app.router.add_post('/process', self.process_request)
        
        # CORS için middleware
        async def cors_middleware(app, handler):
            async def middleware_handler(request):
                if request.method == 'OPTIONS':
                    response = web.Response()
                else:
                    response = await handler(request)
                response.headers['Access-Control-Allow-Origin'] = '*'
                response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
                response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
                return response
            return middleware_handler
        
        app.middlewares.append(cors_middleware)
        
        logger.info(f"HTTP server başlatılıyor: {host}:{port}")
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        await site.start()
        logger.info("HTTP server hazır")
        
        await asyncio.Future()  # Sonsuza kadar çalış

async def main():
    # WebSocket server kullan
    server = WebSocketServer()
    await server.start()
    
    # Veya HTTP server kullan
    # http_server = HTTPServer()
    # await http_server.start_http()

if __name__ == "__main__":
    asyncio.run(main())