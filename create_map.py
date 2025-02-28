import airsim
import os
import time

# Настройки
GRID_SIZE = 10  # Размер сетки (количество снимков в каждой строке/столбце)
ALTITUDE = -35  # Высота полета
VELOCITY = 7  # Скорость полёта
OVERLAP = 0.5  # Перекрытие снимков (0.0 - 1.0)
OUTPUT_DIR = "output_photo_map"  # Папка для сохранения снимков

# Подключение к AirSim
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
client.takeoffAsync().join()

# Вычисляем координаты углов области съемки
# (Эти значения нужно подобрать под ваш проект Blocks)
min_x = -100
min_y = -16.5
max_x = -12.2
max_y = 15.7

# Вычисляем интервал между снимками
x_interval = (max_x - min_x) / (GRID_SIZE - 1)
y_interval = (max_y - min_y) / (GRID_SIZE - 1)

# Создаем папку для сохранения снимков, если она не существует
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
            filename = os.path.join(
                OUTPUT_DIR, f"img_{i}_{j}.png")
            airsim.write_file(filename, response.image_data_uint8)
            print(f"Saved image: {filename}")
        else:
            print("Error: No image retrieved")

# После завершения полета, приземляемся
client.landAsync().join()
client.armDisarm(False)
client.enableApiControl(False)
print("Done!")
