import airsim
import os
import time
import json
import cv2  # Используем OpenCV
import numpy as np

# Настройки
GRID_SIZE = 10
ALTITUDE = -35
VELOCITY = 7
OVERLAP = 0.5
OUTPUT_DIR = "output_photo_map"
JSON_FILENAME = "photo_map.json"

# Подключение к AirSim
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
client.takeoffAsync().join()

# Вычисляем координаты углов области съемки
min_x = -100
min_y = -16.5
max_x = -12.2
max_y = 15.7

# Вычисляем интервал между снимками с учетом перекрытия
x_interval = (max_x - min_x) / (GRID_SIZE - 1) # * (1 - OVERLAP)
y_interval = (max_y - min_y) / (GRID_SIZE - 1) # * (1 - OVERLAP)

# Создаем папку для сохранения снимков, если она не существует
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Создаем список для хранения данных
photo_data = []

# Летаем по сетке и делаем снимки
for i in range(GRID_SIZE):
    for j in range(GRID_SIZE):
        x = min_x + i * x_interval
        y = min_y + j * y_interval

        # Перемещаемся в точку съемки
        client.moveToPositionAsync(x, y, ALTITUDE, VELOCITY).join()  # Исправлено!
        time.sleep(0.5)  # Даем время стабилизироваться

        # Делаем снимок
        responses = client.simGetImages([
            airsim.ImageRequest("3", airsim.ImageType.Scene, False, False)])
        if len(responses) > 0:
            response = responses[0]
            if len(response.image_data_uint8) == 0:
                print("Error: Image data is empty!")
                continue  # Перейти к следующей итерации цикла

            # Преобразуем данные изображения в массив NumPy и сохраняем с помощью OpenCV
            img_rgb = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            img_rgb = img_rgb.reshape(response.height, response.width, 3)  #Исправлено
            filename = os.path.join(OUTPUT_DIR, f"img_{i}_{j}.png")
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filename, img_bgr)  #Исправлено
            print(f"Saved image: {filename}")

            # Получаем состояние дрона для ориентации
            state = client.getMultirotorState()
            orientation = state.kinematics_estimated.orientation

            # Сохраняем метаданные в словарь
            photo_info = {
                'filename': filename,
                'x': x,
                'y': y,
                'orientation': {
                    'w': orientation.w_val,
                    'x': orientation.x_val,
                    'y': orientation.y_val,
                    'z': orientation.z_val
                }
            }
            photo_data.append(photo_info)
        else:
            print("Error: No image retrieved")

# После завершения полета, приземляемся
client.landAsync().join()
client.armDisarm(False)
client.enableApiControl(False)

# Записываем данные в JSON файл *ОДИН РАЗ* после цикла
with open(os.path.join(OUTPUT_DIR, JSON_FILENAME), 'w') as jsonfile:
    json.dump(photo_data, jsonfile, indent=4)

print("Done!")