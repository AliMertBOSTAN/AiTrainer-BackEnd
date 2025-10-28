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

## Ortam Değişkenleri

Backend, `.env` dosyasından aşağıdaki değişkenleri okur:

- `MONGODB_URI`: MongoDB Atlas connection string
- `JWT_SECRET`: JWT token'ları için gizli anahtar
- `MAILER_EMAIL`: Gmail App Password sahibi hesap
- `MAILER_APP_PASSWORD`: Gmail App Password değeri

`.env` dosyasını kök dizinde oluşturun ve uygulamayı çalıştırmadan önce `python-dotenv` ile yüklenmesini sağlayın.

## Kimlik Doğrulama API'si

`server.py` içinde FastAPI tabanlı yeni bir kimlik doğrulama servisi bulunuyor. Servisi çalıştırmak için:

```bash
uvicorn server:auth_app --host 0.0.0.0 --port 8000 --reload
```

### Uç Noktalar

- `POST /api/auth/register`: Yeni bir kullanıcı oluşturur veya doğrulanmamış kaydı güncelleyip doğrulama kodu gönderir.
- `POST /api/auth/verify-email`: Kullanıcının e-postasını doğrular.
- `POST /api/auth/resend-code`: 60 saniyelik cooldown'a uyarak yeni doğrulama kodu gönderir.
- `POST /api/auth/login`: Doğrulanmış kullanıcı için JWT döndürür.

## Expo Auth Yapılandırması

Mobil uygulamada kimlik doğrulama isteklerini bu backend'e yönlendirmek için Expo projenizde aşağıdaki env değerini ekleyin:

```bash
EXPO_PUBLIC_AUTH_API_URL=http://<local-ip>:8000/api/auth
```

`constants/AuthConfig.ts` dosyasında `process.env.EXPO_PUBLIC_AUTH_API_URL` değerinin kullanıldığından emin olun.

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
