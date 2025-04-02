import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import json
import random
import airsim  # Убедитесь, что airsim установлен
import math

# Настройки
ALTITUDE = -35
VELOCITY = 7
OUTPUT_DIR = "current_photo"
PHOTO_MAP_DIR = "output_photo_map"
JSON_FILENAME = "photo_map.json"

# Границы карты местности
min_x = -143.9
min_y = -32.6
max_x = 31.7
max_y = 31.8

# Калибровочные значения (требуется точная настройка!)
pixels_per_meter_x = 10  # Примерное значение: 10 пикселей на метр по оси X
pixels_per_meter_y = 10  # Примерное значение: 10 пикселей на метр по оси Y


def is_coordinate_in_range(x, y, min_x, min_y, max_x, max_y):
    """
    Проверяет, находится ли координата (x, y) в пределах заданного прямоугольника.
    """
    return min_x <= x <= max_x and min_y <= y <= max_y


def pixel_offset_to_meter_offset(dx, dy):
    """
    Преобразует смещение в пикселях в смещение в метрах.
    (Очень упрощенный пример, требующий калибровки)
    """
    offset_x = dx / pixels_per_meter_x
    offset_y = dy / pixels_per_meter_y
    return offset_x, offset_y


def find_location(current_image_path, photo_data, output_dir="matches"):
    """
    Сравнивает текущее изображение с изображениями из photo_data и возвращает наиболее вероятное местоположение,
    а также оценку смещения в пикселях.
    """
    # Создаем директорию для отладочных изображений
    os.makedirs(output_dir, exist_ok=True)

    current_img = cv2.imread(current_image_path, cv2.IMREAD_GRAYSCALE)
    if current_img is None:
        print(f"Error: Could not read image at {current_image_path}")
        return None, 0, 0, 0

    best_match = None
    best_score = 0
    best_match_count = 0
    avg_dx = 0
    avg_dy = 0  # Инициализируем avg_dx и avg_dy

    # Инициализация ORB детектора
    orb = cv2.ORB_create(nfeatures=2000) #Увеличение количества фич
    #orb = cv2.ORB_create()

    # Находим ключевые точки и дескрипторы для текущего изображения
    keypoints_current, descriptors_current = orb.detectAndCompute(current_img, None)
    if descriptors_current is None or len(keypoints_current) < 10:
        print(f"Warning: Could not find enough keypoints in {current_image_path}")
        return None, 0, 0, 0

    for data in photo_data:
        map_img = cv2.imread(data['filename'], cv2.IMREAD_GRAYSCALE)
        if map_img is None:
            print(f"Error: Could not read image at {data['filename']}")
            continue

        # Находим ключевые точки и дескрипторы для изображения из карты
        keypoints_map, descriptors_map = orb.detectAndCompute(map_img, None)
        if descriptors_map is None or len(keypoints_map) < 10:
            continue

        # Создаем объект BFMatcher (Brute-Force Matcher)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Сопоставляем дескрипторы
        try:
            matches = bf.match(descriptors_current, descriptors_map)
        except cv2.error as e:
            print(f"Error during matching: {e}")
            continue

        # Порог расстояния (экспериментируйте со значением)
        distance_threshold = 50  # Примерное значение.  Подберите подходящее для ваших изображений

        # Отбираем только хорошие совпадения по порогу расстояния
        good_matches = [m for m in matches if m.distance < distance_threshold]

        # Создаем новый список good_matches с правильными индексами
        new_good_matches = []
        for match in good_matches:
            if match.queryIdx < len(keypoints_current) and match.trainIdx < len(keypoints_map):
                new_good_matches.append(match)

        # Оценка качества сопоставления: количество хороших совпадений
        score = len(new_good_matches)

        if score > best_score:
            best_score = score
            best_match = data
            best_match_count = len(new_good_matches)
            # Вычисляем смещение суммируем смещения по x и y
            sum_dx = 0
            sum_dy = 0

            for match in new_good_matches:
                query_idx = match.queryIdx
                train_idx = match.trainIdx

                # Получаем координаты ключевых точек
                (x1, y1) = keypoints_current[query_idx].pt
                (x2, y2) = keypoints_map[train_idx].pt

                # Вычисляем смещения
                dx = x1 - x2
                dy = y1 - y2

                # Суммируем смещения
                sum_dx += dx
                sum_dy += dy

            # Вычисляем среднее смещение
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
        return best_match, best_match_count, avg_dx, avg_dy  # Возвращаем смещение
    else:
        print("No match found.")
        return None, 0, 0, 0  # Возвращаем нулевое смещение


def get_current_position(client, photo_data, output_dir, compare_all=False):
    """
    Делает снимок, определяет текущее положение дрона и возвращает координаты (x, y) или None,
    используя информацию о смещении.
    """
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

        if compare_all:
            print("Сравнение со всеми изображениями:")
            for data in photo_data:
                map_image_path = data['filename']
                _, match_count, _, _ = find_location(current_image_path, [data], "matches") # Передаем [data] чтобы сравнивать с одним изображением
                print(f"  {map_image_path}: {match_count} совпадений")
            return None  # Не возвращаем координаты, если сравниваем со всеми

        else:
            best_match, match_count, avg_dx, avg_dy = find_location(current_image_path, photo_data, output_dir="matches")
            if best_match:
                best_score = 0  # Инициализация best_score
                print(f"Найдено лучшее совпадение: {best_match['filename']} со счетом {best_score}")
                current_x = best_match['x']
                current_y = best_match['y']

                # Преобразуем смещение в пикселях в смещение в метрах
                offset_x, offset_y = pixel_offset_to_meter_offset(avg_dx, avg_dy)

                # Корректируем координаты
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

# Взлетаем на высоту -35
client.takeoffAsync().join()
client.moveToPositionAsync(0, 0, ALTITUDE, 5).join()  # Сначала поднимаемся вверх
time.sleep(2) # Даем время стабилизироваться

# Создаем каталог для текущей фотографии
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Списки для хранения данных графиков
real_x_coords = []
real_y_coords = []
vision_x_coords = []
vision_y_coords = []

# Границы карты местности
min_x = -143.9
min_y = -32.6
max_x = 31.7
max_y = 31.8

try:
    # Запрашиваем у пользователя целевые координаты
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

    # Загружаем "карту местности" из JSON-файла
    with open(os.path.join(PHOTO_MAP_DIR, JSON_FILENAME), 'r') as jsonfile:
        photo_data = json.load(jsonfile)

    # Инициализация:  Перемещаемся в начальную точку
    # Определите начальную точку
    start_x = min_x
    start_y = min_y

    # Переместите дрон в начальную точку
    print(f"Перемещаемся в начальную точку: x={start_x}, y={start_y}")
    client.moveToPositionAsync(start_x, start_y, ALTITUDE, VELOCITY).join()
    time.sleep(2)  # Дайте дрону время стабилизироваться

    # Логирование:  Проверяем, какие изображения соответствуют начальной точке
    print("Сравнение начального снимка со всеми изображениями карты (compare_all=True):")
    get_current_position(client, photo_data, OUTPUT_DIR, compare_all=True) # Сравниваем со всеми

    # Инициализация: Перемещаем дрон к начальной точке.
    position = get_current_position(client, photo_data, OUTPUT_DIR)
    if position is None:
        print("Не удалось определить положение. Прерываем навигацию.")
        exit()
    current_x, current_y = position

    # Цикл навигации
    max_iterations = 50  # Ограничение на количество итераций
    iteration = 0
    estimated_x = current_x  # Начальная оценка
    estimated_y = current_y

    while iteration < max_iterations:
        iteration += 1
        print(f"Итерация {iteration}:")

        # 1. Получаем текущее положение (комп. зрение)
        position = get_current_position(client, photo_data, OUTPUT_DIR)
        if position is None:
            print("Не удалось определить положение. Прерываем навигацию.")
            break
        current_x, current_y = position

        # 2.  Фильтр Калмана
        kalman_gain = 0.5  # Подберите значение (от 0 до 1). Чем больше, тем больше доверия к новым измерениям
        estimated_x = estimated_x + kalman_gain * (current_x - estimated_x)
        estimated_y = estimated_y + kalman_gain * (current_y - estimated_y)


        # 3. Получаем реальное положение из AirSim
        real_position = client.getMultirotorState().kinematics_estimated.position
        real_x = real_position.x_val
        real_y = real_position.y_val
        print(f"Реальное положение: x={real_x:.2f}, y={real_y:.2f}") # Добавлено

        # 4. Добавляем данные в списки для графиков
        real_x_coords.append(real_x)
        real_y_coords.append(real_y)
        vision_x_coords.append(estimated_x) # Use estimated values for the graph
        vision_y_coords.append(estimated_y)

        # 5. Вычисляем расстояние до цели по каждой оси
        delta_x = target_x - estimated_x
        delta_y = target_y - estimated_y
        # distance_to_target = (delta_x**2 + delta_y**2)**0.5  # Больше не нужно общее расстояние

        # 6. Вычисляем отдельные шаги для X и Y
        step_size_x = min(abs(delta_x), 1)  # Максимальный шаг по X = 1
        step_size_y = min(abs(delta_y), 1)  # Максимальный шаг по Y = 1

        # Вычисляем перемещение по X и Y
        move_x = estimated_x + step_size_x * np.sign(delta_x)  # Используем знак delta_x
        move_y = estimated_y + step_size_y * np.sign(delta_y)  # Используем знак delta_y

        print(f"Перемещаемся на: x={move_x:.2f}, y={move_y:.2f}")

        # 7. Отправляем команду перемещения (асинхронно)
        client.moveToPositionAsync(move_x, move_y, ALTITUDE, VELOCITY)  # Убрали .join()

        # 8. Ждем 1 секунду
        time.sleep(1)

        # 9. Вычисляем расстояние до цели
        distance_to_target = (delta_x**2 + delta_y**2)**0.5
        print(f"Расстояние до цели: {distance_to_target:.2f}")  # Добавлено

        if distance_to_target < 1:  # Если расстояние до цели меньше 1, считаем, что достигли
            print("Цель достигнута!")
            break

except Exception as e:
    print(f"Произошла ошибка: {e}")
finally:
    # После завершения полета, приземляемся
    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)

    # Создаем графики
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(real_x_coords, label="Реальная координата X (AirSim)")
    plt.plot(vision_x_coords, label="Координата X (комп. зрение)")
    plt.xlabel("Итерация")
    plt.ylabel("Координата X")
    plt.legend()
    plt.title("Сравнение координаты X")

    plt.subplot(1, 2, 2)
    plt.plot(real_y_coords, label="Реальная координата Y (AirSim)")
    plt.plot(vision_y_coords, label="Координата Y (комп. зрение)")
    plt.xlabel("Итерация")
    plt.ylabel("Координата Y")
    plt.legend()
    plt.title("Сравнение координаты Y")

    plt.tight_layout()  # Чтобы графики не накладывались друг на друга
    plt.show()