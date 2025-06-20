# fitness_server.py - Dinamik IP ve Port versiyonu
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
        if self.results.pose_landmarks:
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
        self.exercise_state = {}
        
    def analyze(self, img, exercise_type, exercise_id):
        # Resmi işle
        output_img = img.copy()
        output_img = self.detector.find_pose(output_img)
        lm_list = self.detector.find_position(output_img)
        
        # Egzersiz durumunu başlat
        if exercise_id not in self.exercise_state:
            self.exercise_state[exercise_id] = {
                'count': 0,
                'direction': 0,
                'form_score': 100
            }
        
        # Egzersiz tipine göre analiz
        if lm_list:
            if exercise_type == 'squat':
                return self.analyze_squat(output_img, lm_list, exercise_id)
            elif exercise_type == 'pushup':
                return self.analyze_pushup(output_img, lm_list, exercise_id)
            elif exercise_type == 'biceps_curl':
                return self.analyze_biceps_curl(output_img, lm_list, exercise_id)
            else:
                # Diğer egzersizler için basit sayma
                return self.simple_count(output_img, lm_list, exercise_id)
        
        return output_img, {
            'count': 0,
            'correct_form': False,
            'feedback': 'Vücut tespit edilemedi. Kameranın önünde durun.'
        }
    
    def analyze_squat(self, img, lm_list, exercise_id):
        state = self.exercise_state[exercise_id]
        
        # Açıları hesapla
        left_knee = self.detector.find_angle(img, 23, 25, 27)  # Sol diz
        right_knee = self.detector.find_angle(img, 24, 26, 28)  # Sağ diz
        
        # Form kontrolü
        correct_form = True
        feedback = []
        
        # Squat sayımı
        avg_knee = (left_knee + right_knee) / 2
        
        if avg_knee > 170:
            if state['direction'] == 0:
                state['count'] += 0.5
                state['direction'] = 1
        elif avg_knee < 90:
            if state['direction'] == 1:
                state['count'] += 0.5
                state['direction'] = 0
                
        # UI'da gösterilecek bilgiler
        cv2.putText(img, f"Tekrar: {int(state['count'])}", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if not feedback:
            feedback.append("Harika form!")
            
        return img, {
            'count': int(state['count']),
            'correct_form': correct_form,
            'feedback': ' - '.join(feedback) if feedback else "Harika form!"
        }
    
    def analyze_pushup(self, img, lm_list, exercise_id):
        state = self.exercise_state[exercise_id]
        
        # Kol açıları
        left_arm = self.detector.find_angle(img, 11, 13, 15)
        right_arm = self.detector.find_angle(img, 12, 14, 16)
        
        # Form kontrolü
        correct_form = True
        feedback = []
        
        # Push-up sayımı
        avg_arm = (left_arm + right_arm) / 2
        
        if avg_arm > 160:
            if state['direction'] == 0:
                state['count'] += 0.5
                state['direction'] = 1
        elif avg_arm < 90:
            if state['direction'] == 1:
                state['count'] += 0.5
                state['direction'] = 0
                
        cv2.putText(img, f"Tekrar: {int(state['count'])}", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return img, {
            'count': int(state['count']),
            'correct_form': correct_form,
            'feedback': "Mükemmel!"
        }
    
    def analyze_biceps_curl(self, img, lm_list, exercise_id):
        state = self.exercise_state[exercise_id]
        
        # Kol açıları
        left_arm = self.detector.find_angle(img, 11, 13, 15)
        right_arm = self.detector.find_angle(img, 12, 14, 16)
        
        # Biceps curl sayımı
        avg_angle = (left_arm + right_arm) / 2
        
        if avg_angle > 160:
            if state['direction'] == 0:
                state['count'] += 0.5
                state['direction'] = 1
        elif avg_angle < 50:
            if state['direction'] == 1:
                state['count'] += 0.5
                state['direction'] = 0
                
        cv2.putText(img, f"Tekrar: {int(state['count'])}", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return img, {
            'count': int(state['count']),
            'correct_form': True,
            'feedback': "Harika gidiyorsunuz!"
        }
    
    def simple_count(self, img, lm_list, exercise_id):
        state = self.exercise_state[exercise_id]
        
        # Basit hareket sayımı
        state['count'] += 0.1  # Her frame'de küçük artış
        
        cv2.putText(img, f"Hareket Algılandi: {int(state['count'])}", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return img, {
            'count': int(state['count']),
            'correct_form': True,
            'feedback': "Devam edin!"
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
        print("="*50 + "\n")
        
        server = await websockets.serve(self.handle_client, '0.0.0.0', port)
        logger.info("Server hazır! Bağlantı bekleniyor...")
        await asyncio.Future()

# HTTP Alternatif
from aiohttp import web
import aiohttp_cors

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
        return web.json_response(self.server_info)
    
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