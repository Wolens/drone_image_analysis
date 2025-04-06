import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import json
import random
import airsim
import math

# Настройки
ALTITUDE = -35
OUTPUT_DIR = "current_photo"
PHOTO_MAP_DIR = "output_photo_map"
JSON_FILENAME = "photo_map.json"

# Границы карты местности
min_x = -143.9
min_y = -32.6
max_x = 31.7
max_y = 31.8

# Калибровочные значения
pixels_per_meter_x = 10
pixels_per_meter_y = 10


def is_coordinate_in_range(x, y, min_x, min_y, max_x, max_y):
    """Проверяет, находится ли координата (x, y) в пределах заданного прямоугольника."""
    return min_x <= x <= max_x and min_y <= y <= max_y


def pixel_offset_to_meter_offset(dx, dy):
    """Преобразует смещение в пикселях в смещение в метрах."""
    offset_x = dx / pixels_per_meter_x
    offset_y = dy / pixels_per_meter_y
    return offset_x, offset_y


def find_location(current_image_path, photo_data, output_dir="matches"):
    """Сравнивает текущее изображение с изображениями из photo_data и возвращает наиболее вероятное местоположение."""
    os.makedirs(output_dir, exist_ok=True)

    current_img = cv2.imread(current_image_path, cv2.IMREAD_GRAYSCALE)
    if current_img is None:
        print(f"Error: Could not read image at {current_image_path}")
        return None, 0, 0, 0

    best_match = None
    best_score = 0
    best_match_count = 0
    avg_dx = 0
    avg_dy = 0

    orb = cv2.ORB_create(nfeatures=2000)
    keypoints_current, descriptors_current = orb.detectAndCompute(current_img, None)
    if descriptors_current is None or len(keypoints_current) < 10:
        print(f"Warning: Could not find enough keypoints in {current_image_path}")
        return None, 0, 0, 0

    for data in photo_data:
        map_img = cv2.imread(data['filename'], cv2.IMREAD_GRAYSCALE)
        if map_img is None:
            print(f"Error: Could not read image at {data['filename']}")
            continue

        keypoints_map, descriptors_map = orb.detectAndCompute(map_img, None)
        if descriptors_map is None or len(keypoints_map) < 10:
            continue

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        try:
            matches = bf.match(descriptors_current, descriptors_map)
        except cv2.error as e:
            print(f"Error during matching: {e}")
            continue

        distance_threshold = 50

        good_matches = [m for m in matches if m.distance < distance_threshold]

        new_good_matches = []
        for match in good_matches:
            if match.queryIdx < len(keypoints_current) and match.trainIdx < len(keypoints_map):
                new_good_matches.append(match)

        score = len(new_good_matches)

        if score > best_score:
            best_score = score
            best_match = data
            best_match_count = len(new_good_matches)

            sum_dx = 0
            sum_dy = 0

            for match in new_good_matches:
                query_idx = match.queryIdx
                train_idx = match.trainIdx

                (x1, y1) = keypoints_current[query_idx].pt
                (x2, y2) = keypoints_map[train_idx].pt

                dx = x1 - x2
                dy = y1 - y2

                sum_dx += dx
                sum_dy += dy

            if len(new_good_matches) > 0:
                avg_dx = sum_dx / len(new_good_matches)
                avg_dy = sum_dy / len(new_good_matches)
            else:
                avg_dx = 0
                avg_dy = 0

    if best_match:
        print(f"Найдено лучшее совпадение: {best_match['filename']} со счетом {best_score}")
        print(f"Количество хороших совпадений: {best_match_count}")
        print(f"Среднее смещение: dx={avg_dx:.2f}, dy={avg_dy:.2f}")
        return best_match, best_match_count, avg_dx, avg_dy
    else:
        print("No match found.")
        return None, 0, 0, 0


def get_current_position(client, photo_data, output_dir):
    """Делает снимок, определяет текущее положение дрона и возвращает координаты (x, y) или None."""
    responses = client.simGetImages([
        airsim.ImageRequest("3", airsim.ImageType.Scene, False, False)])

    if len(responses) > 0:
        response = responses[0]
        current_image_path = os.path.join(output_dir, "current_image.png")

        img_rgb = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        img_rgb = img_rgb.reshape(response.height, response.width, 3)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(current_image_path, img_bgr)

        print(f"Saved current image: {current_image_path}")

        best_match, match_count, avg_dx, avg_dy = find_location(current_image_path, photo_data, output_dir="matches")
        if best_match:
            current_x = best_match['x']
            current_y = best_match['y']

            offset_x, offset_y = pixel_offset_to_meter_offset(avg_dx, avg_dy)

            current_x += offset_x
            current_y += offset_y

            print(f"Текущее положение (комп. зрение): x={current_x:.2f}, y={current_y:.2f}")
            print(f"Количество совпадающих точек: {match_count}")
            return current_x, current_y
        else:
            print("Не удалось определить текущее положение.")
            return None
    else:
        print("Error: No image retrieved")
        return None


# Подключение к AirSim
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)


try:
    # Взлетаем
    client.takeoffAsync().join()
    client.moveToPositionAsync(0, 0, ALTITUDE, 5).join()
    time.sleep(2)

    # Получаем целевые координаты от пользователя
    try:
        target_x = float(input("Введите целевую координату X: "))
        target_y = float(input("Введите целевую координату Y: "))
    except ValueError:
        print("Ошибка: Некорректный формат координат. Используйте числа.")
        exit()

    # Проверяем, входят ли координаты в диапазон карты местности
    if not is_coordinate_in_range(target_x, target_y, min_x, min_y, max_x, max_y):
        print("Ошибка: Заданные координаты находятся вне диапазона карты местности.")
        exit()

    # Перемещаем дрон к целевой координате
    print(f"Перемещаемся к цели: x={target_x}, y={target_y}")
    client.moveToPositionAsync(target_x, target_y, ALTITUDE, 5).join()
    time.sleep(5)  # Даем дрону время остановиться

    # Садимся
    client.landAsync().join()
    time.sleep(2)

    # Взлетаем снова для определения координат
    client.takeoffAsync().join()
    client.moveToPositionAsync(target_x, target_y, ALTITUDE, 5).join()
    time.sleep(2)


    # Загружаем "карту местности" из JSON-файла
    with open(os.path.join(PHOTO_MAP_DIR, JSON_FILENAME), 'r') as jsonfile:
        photo_data = json.load(jsonfile)

    # Определяем положение по фото
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    vision_x, vision_y = get_current_position(client, photo_data, OUTPUT_DIR)

    if vision_x is not None and vision_y is not None:
        # Вычисляем RMSE
        rmse = math.sqrt((vision_x - target_x) ** 2 + (vision_y - target_y) ** 2)
        print(f"Реальные координаты (заданные): x={target_x:.2f}, y={target_y:.2f}")
        print(f"Координаты по зрению: x={vision_x:.2f}, y={vision_y:.2f}")
        print(f"RMSE: {rmse:.2f}")
    else:
        print("Не удалось определить положение по фото.")


except Exception as e:
    print(f"Произошла ошибка: {e}")
finally:
    print("Завершение программы...")
    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)