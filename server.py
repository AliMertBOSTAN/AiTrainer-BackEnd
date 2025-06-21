# fitness_server.py - Düzeltilmiş ve optimize edilmiş versiyon
import asyncio
import websockets
import json
import base64
import cv2
import numpy as np
import mediapipe as mp
import math
from datetime import datetime
import logging
import socket
import sys
import platform

# Logging yapılandırması
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_local_ip():
    """Yerel IP adresini otomatik bul"""
    try:
        # Önce hostname üzerinden dene
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        
        # 127.0.0.1 ise gerçek IP'yi bul
        if local_ip.startswith('127.'):
            # Geçici bir socket oluştur
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                # Google DNS'e bağlan (gerçekten bağlanmaz, sadece route bulur)
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

def find_free_port(start_port=8765, max_attempts=10):
    """Boş bir port bul"""
    for i in range(max_attempts):
        port = start_port + i
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(('', port))
            sock.close()
            return port
        except OSError:
            continue
    raise RuntimeError(f"Boş port bulunamadı ({start_port}-{start_port + max_attempts})")

def get_network_interfaces():
    """Tüm network interface'leri listele"""
    interfaces = []
    
    if platform.system() == 'Darwin':  # macOS
        import subprocess
        try:
            # macOS için ifconfig kullan
            result = subprocess.run(['ifconfig'], capture_output=True, text=True)
            lines = result.stdout.split('\n')
            current_interface = None
            
            for line in lines:
                if line and not line.startswith('\t') and not line.startswith(' '):
                    current_interface = line.split(':')[0]
                elif 'inet ' in line and current_interface:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        ip = parts[1]
                        if not ip.startswith('127.') and not ip.startswith('169.254.'):
                            interfaces.append({
                                'interface': current_interface,
                                'ip': ip
                            })
        except Exception as e:
            logger.error(f"Interface bilgisi alınamadı: {e}")
    
    return interfaces

# MediaPipe setup
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
        self.exercise_state = {}
        
    def analyze(self, img, exercise_type, exercise_id):
        # Resmi kopyala
        output_img = img.copy()
        
        # Pose tespiti
        output_img = self.detector.find_pose(output_img)
        lm_list = self.detector.find_position(output_img)
        
        # Egzersiz tipini normalize et
        exercise_type = exercise_type.lower().replace('-', '_').replace(' ', '_')
        
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
            'feedback': f'Egzersiz tipi tanınamadı ({exercise_type}) veya vücut tespit edilemedi'
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
        left_knee_angle = self.detector.find_angle(img, 23, 25, 27)
        right_knee_angle = self.detector.find_angle(img, 24, 26, 28)
        
        # Sırt düzlüğü
        back_angle = self.detector.find_angle(img, 11, 23, 25)
        
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
        
        # Vücut düzlüğü kontrolü
        body_angle = self.detector.find_angle(img, 11, 23, 27)
        
        # Kalça yüksekliği kontrolü
        hip_angle = self.detector.find_angle(img, 11, 23, 25)
        
        # Baş pozisyonu kontrolü
        if len(lm_list) > 7:
            neck_angle = self.detector.find_angle(img, 7, 11, 23)
        else:
            neck_angle = 160  # Varsayılan değer
        
        # Form kontrolleri
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
            
        # Plank süresi sayacı (saniye)
        state['count'] += 1/30  # 30 FPS varsayımı
        
        if not feedback:
            feedback.append(f"Mükemmel form! {int(state['count'])} saniye")
            
        return {
            'count': int(state['count']),
            'correct_form': correct_form,
            'feedback': ' - '.join(feedback)
        }
    
    def analyze_situp(self, img, lm_list, exercise_id):
        state = self.exercise_state[exercise_id]
        feedback = []
        correct_form = True
        
        # Gövde açısı
        torso_angle = self.detector.find_angle(img, 11, 23, 25)
        
        # Bacak açısı
        leg_angle = self.detector.find_angle(img, 23, 25, 27)
        
        # Boyun pozisyonu
        if len(lm_list) > 7:
            neck_angle = self.detector.find_angle(img, 7, 11, 23)
        else:
            neck_angle = 150  # Varsayılan değer
        
        # Form kontrolleri
        if leg_angle < 80 or leg_angle > 120:
            feedback.append("Dizlerinizi 90 derece açıda tutun")
            correct_form = False
            
        if neck_angle < 120:
            feedback.append("Boynunuzu zorlamayın, nötr tutun")
            correct_form = False
            
        # Sit-up sayımı
        if torso_angle < 100:  # Yukarı pozisyon
            if state['direction'] == 0:
                state['count'] += 0.5
                state['direction'] = 1
        elif torso_angle > 150:  # Aşağı pozisyon
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
    
    def analyze_jumping_jack(self, img, lm_list, exercise_id):
        state = self.exercise_state[exercise_id]
        feedback = []
        correct_form = True
        
        # Kol açıları
        left_arm_angle = self.detector.find_angle(img, 23, 11, 13)
        right_arm_angle = self.detector.find_angle(img, 24, 12, 14)
        
        # Bacak genişliği kontrolü
        hip_width = abs(lm_list[23][1] - lm_list[24][1])
        ankle_width = abs(lm_list[27][1] - lm_list[28][1])
        
        # Vücut dikliği
        body_straightness = self.detector.find_angle(img, 11, 23, 27)
        
        # Form kontrolleri
        if abs(body_straightness - 180) > 20:
            feedback.append("Vücudunuzu dik tutun")
            correct_form = False
            
        # Jumping jack sayımı
        avg_arm_angle = (left_arm_angle + right_arm_angle) / 2
        
        # Açık pozisyon: Kollar yukarıda, bacaklar açık
        if avg_arm_angle > 160 and ankle_width > hip_width * 2:
            if state['direction'] == 0:
                state['count'] += 0.5
                state['direction'] = 1
        # Kapalı pozisyon: Kollar aşağıda, bacaklar kapalı
        elif avg_arm_angle < 30 and ankle_width < hip_width * 1.2:
            if state['direction'] == 1:
                state['count'] += 0.5
                state['direction'] = 0
                
        if not feedback:
            feedback.append("Mükemmel ritim!")
            
        return {
            'count': int(state['count']),
            'correct_form': correct_form,
            'feedback': ' - '.join(feedback)
        }
    
    def analyze_lunge(self, img, lm_list, exercise_id):
        state = self.exercise_state[exercise_id]
        feedback = []
        correct_form = True
        
        # Sol ve sağ bacak pozisyonlarını belirle
        left_hip_y = lm_list[23][2]
        right_hip_y = lm_list[24][2]
        left_knee_y = lm_list[25][2]
        right_knee_y = lm_list[26][2]
        
        # Hangi bacağın önde olduğunu belirle
        if left_knee_y > right_knee_y:  # Sol bacak önde
            front_knee_angle = self.detector.find_angle(img, 23, 25, 27)
            back_knee_angle = self.detector.find_angle(img, 24, 26, 28)
            hip_angle = self.detector.find_angle(img, 11, 23, 25)
        else:  # Sağ bacak önde
            front_knee_angle = self.detector.find_angle(img, 24, 26, 28)
            back_knee_angle = self.detector.find_angle(img, 23, 25, 27)
            hip_angle = self.detector.find_angle(img, 12, 24, 26)
        
        # Gövde dikliği
        torso_angle = self.detector.find_angle(img, 11, 23, 24)
        
        # Form kontrolleri
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
            
        # Lunge sayımı
        if front_knee_angle < 100:  # Aşağı pozisyon
            if state['direction'] == 0:
                state['count'] += 0.5
                state['direction'] = 1
        elif front_knee_angle > 160:  # Yukarı pozisyon
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

class WebSocketServer:
    def __init__(self):
        self.analyzer = ExerciseAnalyzer()
        self.clients = set()
        self.server_info = {}
        
    async def handle_client(self, websocket):
        self.clients.add(websocket)
        client_address = websocket.remote_address if hasattr(websocket, 'remote_address') else 'Unknown'
        logger.info(f"Yeni client bağlandı: {client_address}")
        
        try:
            # Hoşgeldin mesajı ve server bilgileri
            welcome = {
                'type': 'connected',
                'message': 'Fitness AI Server\'a hoş geldiniz!',
                'server_info': self.server_info,
                'supported_exercises': list(self.analyzer.exercises.keys()),
                'timestamp': datetime.now().timestamp()
            }
            await websocket.send(json.dumps(welcome))
            
            async for message in websocket:
                try:
                    data = json.loads(message)
                    
                    if data.get('type') == 'exercise':
                        # Base64 görüntüyü decode et
                        image_data = base64.b64decode(data['image'])
                        nparr = np.frombuffer(image_data, np.uint8)
                        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        
                        if img is None:
                            raise ValueError("Görüntü decode edilemedi")
                        
                        # Görüntüyü yeniden boyutlandır
                        img = cv2.resize(img, (640, 480))
                        
                        # Egzersiz analizi
                        exercise_type = data.get('exercise_type', 'unknown')
                        exercise_id = data.get('exercise_id', 'default')
                        
                        logger.info(f"Egzersiz analizi: {exercise_type} - {exercise_id}")
                        
                        processed_img, result = self.analyzer.analyze(img, exercise_type, exercise_id)
                        
                        # İşlenmiş görüntüyü base64'e çevir
                        _, buffer = cv2.imencode('.jpg', processed_img, [cv2.IMWRITE_JPEG_QUALITY, 70])
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
                        
                except json.JSONDecodeError:
                    logger.error("Geçersiz JSON mesajı")
                    await websocket.send(json.dumps({
                        'type': 'error',
                        'message': 'Geçersiz JSON formatı'
                    }))
                except Exception as e:
                    logger.error(f"Mesaj işleme hatası: {str(e)}")
                    error_response = {
                        'type': 'error',
                        'message': str(e),
                        'timestamp': datetime.now().timestamp()
                    }
                    await websocket.send(json.dumps(error_response))
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client bağlantısı kapandı: {client_address}")
        except Exception as e:
            logger.error(f"Client hatası: {str(e)}")
        finally:
            self.clients.remove(websocket)
    
    async def start(self, host=None, port=None):
        # Otomatik IP ve port bul
        if host is None:
            host = get_local_ip()
        if port is None:
            port = find_free_port()
        
        self.server_info = {
            'host': host,
            'port': port,
            'type': 'websocket',
            'url': f'ws://{host}:{port}'
        }
        
        logger.info(f"WebSocket server başlatılıyor: ws://{host}:{port}")
        print("\n" + "="*50)
        print("FITNESS AI SERVER")
        print("="*50)
        print(f"WebSocket URL: ws://{host}:{port}")
        print(f"Yerel IP: {host}")
        print(f"Port: {port}")
        
        # Diğer network interface'leri göster
        interfaces = get_network_interfaces()
        if interfaces:
            print("\nDiğer ağ arayüzleri:")
            for iface in interfaces:
                print(f"  {iface['interface']}: {iface['ip']}")
        
        print("\nReact Native uygulamanızda şu URL'yi kullanın:")
        print(f"  const wsUrl = 'ws://{host}:{port}';")
        print("\nDesteklenen egzersizler:")
        for ex in sorted(set(self.analyzer.exercises.values()), key=lambda x: x.__name__):
            print(f"  - {ex.__name__.replace('analyze_', '')}")
        print("="*50 + "\n")
        
        server = await websockets.serve(self.handle_client, '0.0.0.0', port)
        logger.info("Server hazır! Bağlantı bekleniyor...")
        await asyncio.Future()

# HTTP Alternatif
from aiohttp import web
try:
    import aiohttp_cors
except ImportError:
    logger.warning("aiohttp_cors yüklü değil. HTTP CORS desteği olmayacak.")
    aiohttp_cors = None

class HTTPServer:
    def __init__(self):
        self.analyzer = ExerciseAnalyzer()
        self.server_info = {}
        
    async def process_request(self, request):
        try:
            data = await request.json()
            
            # Base64 görüntüyü decode et
            image_data = base64.b64decode(data['image'])
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                raise ValueError("Görüntü decode edilemedi")
            
            # Görüntüyü yeniden boyutlandır
            img = cv2.resize(img, (640, 480))
            
            # Egzersiz analizi
            exercise_type = data.get('exercise_type', 'unknown')
            exercise_id = data.get('exercise_id', 'default')
            
            processed_img, result = self.analyzer.analyze(img, exercise_type, exercise_id)
            
            # İşlenmiş görüntüyü base64'e çevir
            _, buffer = cv2.imencode('.jpg', processed_img, [cv2.IMWRITE_JPEG_QUALITY, 70])
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
    
    async def info_handler(self, request):
        """Server bilgilerini döndür"""
        info = dict(self.server_info)
        info['supported_exercises'] = list(self.analyzer.exercises.keys())
        return web.json_response(info)
    
    async def start_http(self, host=None, port=None):
        # Otomatik IP ve port bul
        if host is None:
            host = get_local_ip()
        if port is None:
            port = find_free_port(start_port=5000)
        
        self.server_info = {
            'host': host,
            'port': port,
            'type': 'http',
            'url': f'http://{host}:{port}',
            'endpoints': {
                'process': f'http://{host}:{port}/process',
                'info': f'http://{host}:{port}/info'
            }
        }
        
        app = web.Application()
        
        # CORS ayarları
        if aiohttp_cors:
            cors = aiohttp_cors.setup(app, defaults={
                "*": aiohttp_cors.ResourceOptions(
                    allow_credentials=True,
                    expose_headers="*",
                    allow_headers="*",
                    allow_methods="*"
                )
            })
            
            # Route'ları ekle
            app.router.add_get('/info', self.info_handler)
            resource = cors.add(app.router.add_resource("/process"))
            cors.add(resource.add_route("POST", self.process_request))
        else:
            # CORS olmadan route'ları ekle
            app.router.add_get('/info', self.info_handler)
            app.router.add_post('/process', self.process_request)
            
            # Manuel CORS middleware
            async def cors_middleware(app, handler):
                async def middleware_handler(request):
                    if request.method == 'OPTIONS':
                        response = web.Response()
                    else:
                        response = await handler(request)
                    response.headers['Access-Control-Allow-Origin'] = '*'
                    response.headers['Access-Control-Allow-Methods'] = 'POST, GET, OPTIONS'
                    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
                    return response
                return middleware_handler
            
            app.middlewares.append(cors_middleware)
        
        print("\n" + "="*50)
        print("FITNESS AI HTTP SERVER")
        print("="*50)
        print(f"HTTP URL: http://{host}:{port}")
        print(f"Process endpoint: http://{host}:{port}/process")
        print(f"Info endpoint: http://{host}:{port}/info")
        print(f"Yerel IP: {host}")
        print(f"Port: {port}")
        
        # Diğer network interface'leri göster
        interfaces = get_network_interfaces()
        if interfaces:
            print("\nDiğer ağ arayüzleri:")
            for iface in interfaces:
                print(f"  {iface['interface']}: {iface['ip']}")
        
        print("\nReact Native uygulamanızda şu URL'yi kullanın:")
        print(f"  const httpUrl = 'http://{host}:{port}/process';")
        print("="*50 + "\n")
        
        logger.info(f"HTTP server başlatılıyor: http://{host}:{port}")
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', port)
        await site.start()
        logger.info("HTTP server hazır!")
        
        await asyncio.Future()

async def main():
    # Komut satırı argümanlarını kontrol et
    server_type = 'ws'  # varsayılan WebSocket
    custom_port = None
    
    for arg in sys.argv[1:]:
        if arg == 'http':
            server_type = 'http'
        elif arg.isdigit():
            custom_port = int(arg)
    
    print("Fitness AI Server Başlatılıyor...")
    print(f"Server tipi: {server_type.upper()}")
    
    if server_type == 'http':
        server = HTTPServer()
        await server.start_http(port=custom_port)
    else:
        server = WebSocketServer()
        await server.start(port=custom_port)

if __name__ == "__main__":
    # Kullanım:
    # python fitness_server.py          # WebSocket, otomatik port
    # python fitness_server.py http     # HTTP, otomatik port  
    # python fitness_server.py 9000     # WebSocket, port 9000
    # python fitness_server.py http 5500  # HTTP, port 5500
    
    asyncio.run(main())