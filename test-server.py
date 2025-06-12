import asyncio
import websockets
import json
import base64
import cv2
import numpy as np
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestClient:
    def __init__(self, server_url='ws://localhost:3000'):
        self.server_url = server_url
        self.running = False
        
    async def send_video_frames(self, video_source=0, exercise_type='pushup'):
        """Video kaynağından frame'leri server'a gönder"""
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            logger.error("Video kaynağı açılamadı")
            return
            
        async with websockets.connect(self.server_url) as websocket:
            logger.info("Server'a bağlandı")
            
            # İlk mesajı al
            welcome = await websocket.recv()
            logger.info(f"Server mesajı: {welcome}")
            
            frame_count = 0
            self.running = True
            
            try:
                while self.running:
                    ret, frame = cap.read()
                    if not ret:
                        logger.info("Video sonu")
                        break
                    
                    # Frame'i resize et (performans için)
                    frame = cv2.resize(frame, (640, 480))
                    
                    # Frame'i base64'e çevir
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    frame_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    # Mesajı hazırla
                    message = {
                        'type': 'exercise',
                        'image': frame_base64,
                        'exercise_type': exercise_type,
                        'exercise_id': f'test_{exercise_type}'
                    }
                    
                    # Gönder
                    await websocket.send(json.dumps(message))
                    frame_count += 1
                    
                    # Cevabı al
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        result = json.loads(response)
                        
                        if result['type'] == 'result':
                            data = result['data']
                            logger.info(f"Frame {frame_count}: Count={data['count']}, "
                                      f"Form={'✓' if data['correct_form'] else '✗'}, "
                                      f"Feedback: {data['feedback']}")
                            
                            # İşlenmiş görüntüyü göster
                            if 'processed_image' in data:
                                processed_data = base64.b64decode(data['processed_image'])
                                nparr = np.frombuffer(processed_data, np.uint8)
                                processed_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                                
                                if processed_img is not None:
                                    # Bilgileri görüntüye ekle
                                    cv2.putText(processed_img, f"Count: {data['count']}", 
                                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                                              (0, 255, 0), 2)
                                    cv2.putText(processed_img, data['feedback'], 
                                              (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                                              (0, 255, 0) if data['correct_form'] else (0, 0, 255), 2)
                                    
                                    cv2.imshow('Processed', processed_img)
                            
                            # Orijinal görüntüyü de göster
                            cv2.imshow('Original', frame)
                            
                    except asyncio.TimeoutError:
                        logger.warning("Cevap zaman aşımı")
                    
                    # ESC tuşuna basılırsa çık
                    if cv2.waitKey(30) == 27:
                        break
                    
                    # FPS kontrolü
                    await asyncio.sleep(0.033)  # ~30 FPS
                    
            finally:
                cap.release()
                cv2.destroyAllWindows()
                self.running = False
    
    async def send_single_image(self, image_path, exercise_type='pushup'):
        """Tek bir görüntü gönder"""
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Görüntü yüklenemedi: {image_path}")
            return
            
        async with websockets.connect(self.server_url) as websocket:
            logger.info("Server'a bağlandı")
            
            # Görüntüyü base64'e çevir
            _, buffer = cv2.imencode('.jpg', img)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Mesajı gönder
            message = {
                'type': 'exercise',
                'image': img_base64,
                'exercise_type': exercise_type,
                'exercise_id': f'test_{exercise_type}_single'
            }
            
            await websocket.send(json.dumps(message))
            
            # Cevabı al
            response = await websocket.recv()
            result = json.loads(response)
            
            if result['type'] == 'result':
                data = result['data']
                logger.info(f"Sonuç: {data}")
                
                # İşlenmiş görüntüyü göster
                if 'processed_image' in data:
                    processed_data = base64.b64decode(data['processed_image'])
                    nparr = np.frombuffer(processed_data, np.uint8)
                    processed_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if processed_img is not None:
                        cv2.imshow('Processed', processed_img)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()

    async def test_http_endpoint(self, image_path, exercise_type='pushup'):
        """HTTP endpoint'i test et"""
        import a