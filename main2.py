import airsim
import os
import time
import json
import cv2
import numpy as np

class SensorInput:
    """
    Manages the acquisition of data from various sensors installed on the drone (simulated in AirSim).
    """
    def __init__(self, client, camera_name="3"):
        self.client = client
        self.camera_name = camera_name
        self.image_type = airsim.ImageType.Scene # Начинаем с Scene
        self.image_width = 256
        self.image_height = 144

    def get_camera_data(self):
        """
        Capture an image frame from the camera using AirSim API.
        """
        responses = self.client.simGetImages([
            airsim.ImageRequest(self.camera_name, self.image_type)])
        if len(responses) > 0:
            response = responses[0]
            img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            print(f"Image height: {response.height}, width: {response.width}")
            print(f"Data length: {len(img1d)}")
            print(f"Image type: {self.image_type}")
            print(f"Pixel format: uint8")

            # Попытка декодировать изображение с помощью cv2.imdecode
            try:
                img_np = np.frombuffer(response.image_data_uint8, np.uint8)
                img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

                if img is not None:
                    # Flip the image vertically
                    img = np.flipud(img)
                    return img, img.shape[0], img.shape[1]
                else:
                    print("Error decoding image with cv2.imdecode")
                    return None, None, None
            except Exception as e:
                print(f"Error processing image: {e}")
                return None, None, None
        else:
            print("Failed to read from camera")
            return None, None, None

# --- 1. Загрузка базы данных изображений и координат ---
def load_photo_data(json_filename):
    """Загружает данные о снимках из JSON файла."""
    with open(json_filename, 'r') as f:
        data = json.load(f)
    return data

# --- 3. Локализация (сопоставление изображений) ---
def find_location(new_image, height, width, photo_data):
    """Определяет местоположение дрона на основе сопоставления ключевых точек с базой данных."""
    if new_image is None:
        print("No image to find location with.")
        return None

    # 1. Детекция и описание ключевых точек на новом снимке
    orb = cv2.ORB_create()
    keypoints_new, descriptors_new = orb.detectAndCompute(new_image, None)

    if descriptors_new is None or len(keypoints_new) < 5:
        print("Not enough keypoints detected in the new image.")
        return None

    best_match = None
    max_matches = 0

    # 2. Сопоставление нового снимка с каждым снимком в базе данных
    for photo in photo_data:
        # Загрузка изображения из базы данных
        database_image = cv2.imread(photo['filename'], cv2.IMREAD_COLOR)
        if database_image is None:
            print(f"Error loading database image: {photo['filename']}")
            continue

        # Детекция и описание ключевых точек на снимке из базы данных
        keypoints_db, descriptors_db = orb.detectAndCompute(database_image, None)

        if descriptors_db is None or len(keypoints_db) < 5:
            print(f"Not enough keypoints detected in {photo['filename']}") # Исправлено!
            continue

        # Brute-Force Matching
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptors_new, descriptors_db)

        # Отбираем хорошие совпадения
        good_matches = []
        for m in matches:
            if m.distance < 30:  # adjust this threshold
                good_matches.append(m)

        # RANSAC (необязательно, но рекомендуется)
        # ... (добавьте RANSAC для фильтрации ложных соответствий)

        # Оценка количества совпадений
        num_matches = len(good_matches)

        # Обновление наилучшего соответствия
        if num_matches > max_matches:
            max_matches = num_matches
            best_match = photo

    # 3. Возврат координат наилучшего соответствия
    if best_match:
        return float(best_match['x']), float(best_match['y'])
    else:
        return None

# --- 4. Планирование маршрута ---
def plan_route(current_location, target_location):
    """Планирует маршрут от текущего местоположения до целевого (прямая линия)."""
    return [target_location]

# --- 5. Управление дроном ---
def navigate_drone(client, route, altitude, velocity):
    """Перемещает дрон по заданному маршруту."""
    for point in route:
        x, y = point
        client.moveToPositionAsync(x, y, altitude, velocity, airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False, 0)).join()
        print(f"Moving to x={x}, y={y}, altitude={altitude}")

# --- Основная программа ---
if __name__ == "__main__":
    # --- Настройки ---
    JSON_FILENAME = "output_photo_map/photo_map.json"
    TARGET_X = -20
    TARGET_Y = 5
    ALTITUDE = -35
    VELOCITY = 3  # Reduced velocity for better accuracy
    CAMERA_NAME = "3"

    # --- Подключение к AirSim ---
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    client.takeoffAsync().join()

    # --- Взлет на заданную высоту ---
    print(f"Ascending to altitude: {ALTITUDE}")
    client.moveToZAsync(ALTITUDE, VELOCITY).join() # Move to altitude
    print("Reached target altitude.")

    # --- Загрузка базы данных ---
    photo_data = load_photo_data(JSON_FILENAME)

    # --- Инициализация сенсоров ---
    sensor_system = SensorInput(client, CAMERA_NAME)

    # --- Главный цикл ---
    start_time = time.time()  # Add a timeout to prevent infinite loops
    while time.time() - start_time < 60: # Timeout after 60 seconds
        # --- 1. Получение данных с сенсоров ---
        camera_image, height, width = sensor_system.get_camera_data()

        if camera_image is None:
            print("Failed to get camera data. Retrying...")
            time.sleep(1)
            continue

        # --- 2. Локализация ---
        current_location = find_location(camera_image, height, width, photo_data)

        if current_location:
            current_x, current_y = current_location
            print(f"Current location: x={current_x}, y={current_y}")

            # --- 3. Планирование маршрута ---
            target_location = (TARGET_X, TARGET_Y)
            route = plan_route(current_location, target_location)

            # --- 4. Управление дроном ---
            navigate_drone(client, route, ALTITUDE, VELOCITY)
            break  # Successfully navigated, exit the loop

        else:
            print("Could not determine location. Retrying...")
            time.sleep(1)

    # --- Завершение ---
    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)
    print("Done!")