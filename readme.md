# Pose Estimation WebSocket Server

MediaPipe kullanarak gerçek zamanlı egzersiz takibi yapan WebSocket/HTTP server.

## Özellikler

- **WebSocket ve HTTP desteği**: React Native uygulamanız her iki protokolü de kullanabilir
- **Gerçek zamanlı pose tespiti**: MediaPipe ile hızlı ve doğru vücut noktası tespiti
- **Çoklu egzersiz desteği**:
  - Push-up
  - Squat
  - Biceps Curl
  - Shoulder Press
  - Plank
  - Lunges
  - Jumping Jacks
- **Form analizi**: Egzersiz formunun doğruluğunu kontrol eder
- **Otomatik sayım**: Tekrar sayısını otomatik hesaplar
- **Görsel geri bildirim**: İşlenmiş görüntüde vücut noktaları ve açıları gösterir

## Kurulum

1. **Python 3.8+ yüklü olduğundan emin olun**

2. **Gerekli paketleri yükleyin**:
```bash
pip install -r requirements.txt
```

3. **Server'ı başlatın**:
```bash
python pose_server.py
```

## React Native Uygulaması İçin Yapılandırma

1. **IP Adresini Güncelleme**:
   - `WorkoutScreen.js` dosyasında `192.168.1.100` yerine bilgisayarınızın IP adresini yazın
   - WebSocket için: `ws://YOUR_IP:3000`
   - HTTP için: `http://YOUR_IP:3000/process`

2. **Ağ İzinleri**:
   - iOS: `Info.plist` dosyasına local network usage description ekleyin
   - Android: `AndroidManifest.xml` dosyasına internet permission ekleyin

## Kullanım

### WebSocket Protokolü

**Gönderilecek Mesaj Formatı**: