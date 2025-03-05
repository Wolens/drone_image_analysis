import airsim
import os
import time
import json
import cv2
import numpy as np
import random
from find_location import find_location
import math
import matplotlib.pyplot as plt  # Добавляем импорт matplotlib

# Настройки
ALTITUDE = -35
VELOCITY = 7
OUTPUT_DIR = "current_photo"
PHOTO_MAP_DIR = "output_photo_map"
JSON_FILENAME = "photo_map.json"

# Границы карты местности
min_x = -100
min_y = -16.5
max_x = -12.2
max_y = 15.7


def is_coordinate_in_range(x, y, min_x, min_y, max_x, max_y):
    """
    Проверяет, находится ли координата (x, y) в пределах заданного прямоугольника.
    """
    return min_x <= x <= max_x and min_y <= y <= max_y


def get_current_position(client, photo_data, output_dir, compare_all=False):
    """
    Делает снимок, определяет текущее положение дрона и возвращает координаты (x, y) или None.
    Если compare_all=True, то сравнивает текущее изображение со всеми изображениями в photo_data
    и выводит количество совпадающих точек для каждого.
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
                _, match_count = find_location(current_image_path, [data], "matches") # Передаем [data] чтобы сравнивать с одним изображением
                print(f"  {map_image_path}: {match_count} совпадений")
            return None  # Не возвращаем координаты, если сравниваем со всеми

        else:
            best_match, match_count = find_location(current_image_path, photo_data, "matches")

            if best_match:
                current_x = best_match['x']
                current_y = best_match['y']
                print(f"Текущее положение (комп. зрение): x={current_x}, y={current_y}")
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
client.takeoffAsync().join()

# Создаем каталог для текущей фотографии
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Списки для хранения данных графиков
real_x_coords = []
real_y_coords = []
vision_x_coords = []
vision_y_coords = []

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

    # Инициализация:  Взлетаем и делаем снимок для сравнения со всеми
    client.moveToPositionAsync(0, 0, -30, 5).join()  # Взлетаем на высоту -30
    time.sleep(2)  # Даем время стабилизироваться

    print("Сравнение начального снимка со всеми изображениями карты:")
    get_current_position(client, photo_data, OUTPUT_DIR, compare_all=True) # Сравниваем со всеми

    # Инициализация: Перемещаем дрон к начальной точке.
    position = get_current_position(client, photo_data, OUTPUT_DIR)
    if position is None:
        print("Не удалось определить положение. Прерываем навигацию.")
        exit()
    current_x, current_y = position
    delta_x = target_x - current_x
    delta_y = target_y - current_y
    distance_to_target = (delta_x**2 + delta_y**2)**0.5
    step_size = min(distance_to_target, 2)  # Размер шага
    move_x = current_x + delta_x / distance_to_target * step_size
    move_y = current_y + delta_y / distance_to_target * step_size

    client.moveToPositionAsync(move_x, move_y, ALTITUDE, VELOCITY)


    # Цикл навигации
    max_iterations = 50  # Ограничение на количество итераций
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        print(f"Итерация {iteration}:")

        # 1. Получаем текущее положение (комп. зрение)
        position = get_current_position(client, photo_data, OUTPUT_DIR)
        if position is None:
            print("Не удалось определить положение. Прерываем навигацию.")
            break
        current_x, current_y = position

        # 2. Получаем реальное положение из AirSim
        real_position = client.getMultirotorState().kinematics_estimated.position
        real_x = real_position.x_val
        real_y = real_position.y_val

        # 3. Добавляем данные в списки для графиков
        real_x_coords.append(real_x)
        real_y_coords.append(real_y)
        vision_x_coords.append(current_x)
        vision_y_coords.append(current_y)

        # 4. Вычисляем расстояние до цели
        delta_x = target_x - current_x
        delta_y = target_y - current_y
        distance_to_target = (delta_x**2 + delta_y**2)**0.5

        if distance_to_target < 1:  # Если расстояние до цели меньше 1, считаем, что достигли
            print("Цель достигнута!")
            break

        # 5. Вычисляем направление движения и перемещаемся
        step_size = min(distance_to_target, 2)  # Размер шага
        move_x = current_x + delta_x / distance_to_target * step_size
        move_y = current_y + delta_y / distance_to_target * step_size

        # 6. Добавляем случайный ветер
        wind_x = random.uniform(-5, 0.5)
        wind_y = random.uniform(-5, 0.5)
        move_x += wind_x
        move_y += wind_y

        print(f"Перемещаемся на: x={move_x:.2f}, y={move_y:.2f}")

        # 7. Отправляем команду перемещения (асинхронно)
        client.moveToPositionAsync(move_x, move_y, ALTITUDE, VELOCITY)  # Убрали .join()

        # 8. Ждем 1 секунду
        time.sleep(1)

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