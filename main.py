import airsim
import os
import json
import cv2  # Нужно для загрузки изображений, но можно и pillow
import numpy as np
from find_location import find_location  # Предполагается, что find_location в отдельном файле

# Настройки
ALTITUDE = -35
OUTPUT_DIR = "current_photo" # Каталог для текущей фотографии
PHOTO_MAP_DIR = "output_photo_map" # Каталог с картой местности
JSON_FILENAME = "photo_map.json"

# Подключение к AirSim
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
client.takeoffAsync().join()

# Создаем каталог для текущей фотографии
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    # Делаем снимок
    responses = client.simGetImages([
        airsim.ImageRequest("3", airsim.ImageType.Scene, False, False)])

    if len(responses) > 0:
        response = responses[0]
        # Сохраняем текущий снимок
        current_image_path = os.path.join(OUTPUT_DIR, "current_image.png")

        img_rgb = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        img_rgb = img_rgb.reshape(response.height, response.width, 3)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR) # AirSim выдает RGB, OpenCV ждет BGR
        cv2.imwrite(current_image_path, img_bgr)

        print(f"Saved current image: {current_image_path}")

        # Загружаем "карту местности" из JSON-файла
        with open(os.path.join(PHOTO_MAP_DIR, JSON_FILENAME), 'r') as jsonfile:
            photo_data = json.load(jsonfile)

        # Определяем местоположение
        best_match = find_location(current_image_path, photo_data, "matches")

        if best_match:
            print(f"Дрон находится в: x={best_match['x']}, y={best_match['y']}")
        else:
            print("Не удалось определить местоположение дрона.")

    else:
        print("Error: No image retrieved")

finally:
    # После завершения полета, приземляемся
    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)