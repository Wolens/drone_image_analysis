import airsim
import os
import time
import json
import cv2  # Используем OpenCV
import numpy as np  # Для работы с массивами

# --- НАСТРОЙКИ ---
GRID_SIZE = 20  # Увеличиваем количество фотографий
ALTITUDE = -35
VELOCITY = 7
OVERLAP = 0.5
OUTPUT_DIR = "output_photo_map"
JSON_FILENAME = "photo_map.json"

# Расширяем территорию в 2 раза
INITIAL_MIN_X = -100
INITIAL_MIN_Y = -16.5
INITIAL_MAX_X = -12.2
INITIAL_MAX_Y = 15.7

EXPANSION_FACTOR = 2
CENTER_X = (INITIAL_MIN_X + INITIAL_MAX_X) / 2
CENTER_Y = (INITIAL_MIN_Y + INITIAL_MAX_Y) / 2
HALF_WIDTH = (INITIAL_MAX_X - INITIAL_MIN_X) / 2 * EXPANSION_FACTOR
HALF_HEIGHT = (INITIAL_MAX_Y - INITIAL_MIN_Y) / 2 * EXPANSION_FACTOR

min_x = CENTER_X - HALF_WIDTH
min_y = CENTER_Y - HALF_HEIGHT
max_x = CENTER_X + HALF_WIDTH
max_y = CENTER_Y + HALF_HEIGHT

print(f"Границы области съемки: min_x={min_x}, min_y={min_y}, max_x={max_x}, max_y={max_y}")

# --- ПОДКЛЮЧЕНИЕ К AIRSIM ---
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
client.takeoffAsync().join()

# --- ВЗЛЕТ НА ЗАДАННУЮ ВЫСОТУ ---
print(f"Взлет на высоту {ALTITUDE} метров...")
client.moveToPositionAsync(0, 0, ALTITUDE, VELOCITY).join()  # Взлетаем в точке (0, 0) на нужную высоту
time.sleep(2) # Даем дрону время стабилизироваться после взлета

# --- ВЫЧИСЛЕНИЕ ИНТЕРВАЛОВ ---
x_interval = (max_x - min_x) / (GRID_SIZE - 1) * (1 - OVERLAP)
y_interval = (max_y - min_y) / (GRID_SIZE - 1) * (1 - OVERLAP)

# --- СОЗДАНИЕ ДИРЕКТОРИИ ---
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- ОСНОВНОЙ ЦИКЛ СЪЕМКИ ---
photo_data = []  # Список для хранения метаданных
try:
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            x = min_x + i * x_interval
            y = min_y + j * y_interval

            print(f"Перемещение в точку: x={x}, y={y}")
            client.moveToPositionAsync(x, y, ALTITUDE, VELOCITY).join() # Летим на высоте к нужной точке
            time.sleep(0.5)  # Даем дрону время стабилизироваться

            # --- ЗАПРОС ИЗОБРАЖЕНИЯ ---
            responses = client.simGetImages([
                airsim.ImageRequest("3", airsim.ImageType.Scene, False, False)
            ])

            if responses and len(responses) > 0:
                response = responses[0]
                if len(response.image_data_uint8) == 0:
                    print("Ошибка: данные изображения пусты!")
                    continue  # Переходим к следующей итерации

                # --- ОБРАБОТКА ИЗОБРАЖЕНИЯ ---
                img_rgb = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
                img_rgb = img_rgb.reshape(response.height, response.width, 3)
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
                filename = os.path.join(OUTPUT_DIR, f"img_{i}_{j}.png")
                cv2.imwrite(filename, img_bgr)  # Save the BGR image
                print(f"Сохранено изображение: {filename}")

                # --- ПОЛУЧЕНИЕ ДАННЫХ ОРИЕНТАЦИИ ---
                state = client.getMultirotorState()
                orientation = state.kinematics_estimated.orientation

                # --- СОХРАНЕНИЕ МЕТАДАННЫХ ---
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
                print("Ошибка: изображение не получено")

except Exception as e:
    print(f"Произошла ошибка во время полета: {e}")

finally:
    # --- ЗАВЕРШЕНИЕ ПОЛЕТА ---
    print("Посадка...")
    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)

    # --- ЗАПИСЬ ДАННЫХ В JSON ---
    json_path = os.path.join(OUTPUT_DIR, JSON_FILENAME)
    with open(json_path, 'w') as jsonfile:
        json.dump(photo_data, jsonfile, indent=4)
    print(f"Метаданные сохранены в: {json_path}")

    print("Готово!")