import airsim
import os
import time
import json
import cv2  # Используем OpenCV
import numpy as np  # Для работы с массивами

# --- НАСТРОЙКИ ---
MIN_X = -144
MIN_Y = -32
MAX_X = 32
MAX_Y = 32
STEP = 1  # Шаг 1 метр
ALTITUDE = -35
VELOCITY = 7
OUTPUT_DIR = "output_photo_map"
JSON_FILENAME = "photo_map.json"

# --- ПРОВЕРКА ДОПУСТИМОСТИ ГРАНИЦ И ШАГА ---
if MAX_X <= MIN_X or MAX_Y <= MIN_Y:
    raise ValueError("MAX_X и MAX_Y должны быть больше, чем MIN_X и MIN_Y соответственно.")

if STEP <= 0:
    raise ValueError("Шаг STEP должен быть положительным числом.")

# --- ПОДКЛЮЧЕНИЕ К AIRSIM ---
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
client.takeoffAsync().join()

# --- ВЗЛЕТ НА ЗАДАННУЮ ВЫСОТУ ---
print(f"Взлет на высоту {ALTITUDE} метров...")
client.moveToPositionAsync(0, 0, ALTITUDE, VELOCITY).join()  # Взлетаем в точке (0, 0) на нужную высоту
time.sleep(2)  # Даем дрону время стабилизироваться после взлета

# --- СОЗДАНИЕ ДИРЕКТОРИИ ---
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- ОСНОВНОЙ ЦИКЛ СЪЕМКИ ---
photo_data = []  # Список для хранения метаданных
try:
    x = MIN_X
    while x <= MAX_X:
        y = MIN_Y
        while y <= MAX_Y:
            print(f"Перемещение в точку: x={x:.2f}, y={y:.2f}")
            client.moveToPositionAsync(x, y, ALTITUDE, VELOCITY).join()  # Летим на высоте к нужной точке
            time.sleep(0.5)  # Даем дрону время стабилизироваться

            # --- ЗАПРОС ИЗОБРАЖЕНИЯ ---
            responses = client.simGetImages([
                airsim.ImageRequest("3", airsim.ImageType.Scene, False, False)
            ])
            if responses and len(responses) > 0:
                response = responses[0]
                if len(response.image_data_uint8) == 0:
                    print("Ошибка: данные изображения пусты!")
                    y += STEP
                    continue  # Переходим к следующей итерации

                # --- ОБРАБОТКА ИЗОБРАЖЕНИЯ ---
                img_rgb = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
                img_rgb = img_rgb.reshape(response.height, response.width, 3)
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)  # Преобразование в BGR для OpenCV
                filename = os.path.join(OUTPUT_DIR, f"img_{x:.2f}_{y:.2f}.png")
                cv2.imwrite(filename, img_bgr)  # Сохраняем изображение в формате BGR
                print(f"Изображение сохранено: {filename}")

                # --- ПОЛУЧЕНИЕ ДАННЫХ ОБ ОРИЕНТАЦИИ ---
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
            y += STEP
        x += STEP

except Exception as e:
    print(f"Во время полета произошла ошибка: {e}")
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