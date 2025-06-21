import asyncio
import websockets
import json
import base64

# Örnek bir küçük JPEG dosyasını base64'e çeviriyoruz (gerçek görüntü yerine sabit dummy veri kullanalım)
dummy_base64_image = base64.b64encode(b'\xff\xd8\xff\xe0' + b'\x00' * 100 + b'\xff\xd9').decode('utf-8')

async def handler(websocket):
    print("İstemci bağlandı.")
    try:
        async for message in websocket:
            print(f"Gelen mesaj: {message}")

            # Sabit bir yanıt oluştur
            response = {
                "type": "result",
                "data": {
                    "processed_image": dummy_base64_image,
                    "count": 3,
                    "correct_form": True,
                    "feedback": "Hareket formun doğru. Tebrikler!"
                }
            }

            # JSON olarak gönder
            await websocket.send(json.dumps(response))
            print("Sabit cevap gönderildi.")
    except websockets.exceptions.ConnectionClosed:
        print("Bağlantı kapandı.")

async def main():
    async with websockets.serve(handler, "0.0.0.0", 8765):
        print("Sunucu çalışıyor. ws://0.0.0.0:8765")
        await asyncio.Future()  # Sonsuz döngü

if __name__ == "__main__":
    asyncio.run(main())
