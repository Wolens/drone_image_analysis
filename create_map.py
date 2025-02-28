import airsim
import os
import time
import json  # Импортируем модуль json

# Настройки (как и раньше)
GRID_SIZE = 10
ALTITUDE = -35
VELOCITY = 7
OVERLAP = 0.5
OUTPUT_DIR = "output_photo_map"
JSON_FILENAME = "photo_map.json"  # Имя файла для JSON

# Подключение к AirSim (как и раньше)
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
client.takeoffAsync().join()

# Вычисляем координаты углов области съемки (как и раньше)
min_x = -100
min_y = -16.5
max_x = -12.2
max_y = 15.7

# Вычисляем интервал между снимками (как и раньше)
x_interval = (max_x - min_x) / (GRID_SIZE - 1)
y_interval = (max_y - min_y) / (GRID_SIZE - 1)

# Создаем папку для сохранения снимков, если она не существует (как и раньше)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Создаем список для хранения данных
photo_data = []

# Летаем по сетке и делаем снимки
for i in range(GRID_SIZE):
    for j in range(GRID_SIZE):
        x = min_x + i * x_interval
        y = min_y + j * y_interval

        # Перемещаемся в точку съемки
        client.moveToPositionAsync(x, y, ALTITUDE, VELOCITY).join()
        time.sleep(0.5)  # Даем время стабилизироваться

        # Делаем снимок
        responses = client.simGetImages([
            airsim.ImageRequest("3", airsim.ImageType.DepthVis)])
        if len(responses) > 0:
            response = responses[0]
            filename = os.path.join(OUTPUT_DIR, f"img_{i}_{j}.png")
            airsim.write_file(filename, response.image_data_uint8)
            print(f"Saved image: {filename}")

            # Сохраняем метаданные в словарь
            photo_info = {'filename': filename, 'x': x, 'y': y}
            photo_data.append(photo_info)
        else:
            print("Error: No image retrieved")

# После завершения полета, приземляемся
client.landAsync().join()
client.armDisarm(False)
client.enableApiControl(False)

# Записываем данные в JSON файл *ОДИН РАЗ* после цикла
with open(os.path.join(OUTPUT_DIR, JSON_FILENAME), 'w') as jsonfile:
    json.dump(photo_data, jsonfile, indent=4)  # indent для красивого форматирования

print("Done!")