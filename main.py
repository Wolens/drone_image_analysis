import airsim
import os
import time
import math

# Параметры полета (измените по своему усмотрению)
TARGET_X = 240.10
TARGET_Y = 63.3
TARGET_Z = -33.60  # Высота (отрицательная, как обычно в AirSim)
VELOCITY = 10  # Скорость движения (метры в секунду) - Снижена
OFFSET_Z = 1.2  # Добавочная высота для съемки
BOTTOM_CAMERA = "3"

# Количество снимков, которые нужно сделать
NUM_SNAPSHOTS = 10

# Подключение к AirSim
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# Взлет
client.takeoffAsync().join()

# Текущая позиция дрона
current_pose = client.simGetVehiclePose()
current_x = current_pose.position.x_val
current_y = current_pose.position.y_val
current_z = current_pose.position.z_val

# Поднимаем дрон на нужную высоту
target_altitude = TARGET_Z + OFFSET_Z
print(f"Moving vertically to altitude: {target_altitude}")
client.moveToPositionAsync(
    current_x, current_y, target_altitude, VELOCITY,
    timeout_sec=10).join()

# Вычисляем расстояние до целевой точки по горизонтали
distance_to_target = math.sqrt(
    (TARGET_X - current_x)**2 + (TARGET_Y - current_y)**2)

# Вычисляем время полета (только горизонтальное движение)
flight_time = distance_to_target / VELOCITY

# Вычисляем интервал времени между снимками
time_interval = flight_time / NUM_SNAPSHOTS

# Создаем директорию "output" в текущей директории скрипта,
# если она не существует
output_dir = os.path.join(os.getcwd(), "output")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Цикл для перемещения и захвата снимков
for i in range(NUM_SNAPSHOTS):
    # Вычисляем целевую позицию для текущего шага
    step_x = current_x + (TARGET_X - current_x) * (i + 1) / NUM_SNAPSHOTS
    step_y = current_y + (TARGET_Y - current_y) * (i + 1) / NUM_SNAPSHOTS
    step_z = target_altitude  # Высота остается постоянной

    # Перемещаем дрон к целевой позиции для текущего шага
    print(f"Moving to: X={step_x}, Y={step_y}, Z={step_z}")
    client.moveToPositionAsync(
        step_x, step_y, step_z, VELOCITY, timeout_sec=10,
        drivetrain=airsim.DrivetrainType.ForwardOnly,
        yaw_mode=airsim.YawMode(False, 0)).join()

    # Запрос изображений
    responses = client.simGetImages([
        airsim.ImageRequest(BOTTOM_CAMERA, airsim.ImageType.DepthVis),
        airsim.ImageRequest("1", airsim.ImageType.DepthPlanar, True)])

    print(f'Retrieved images for snapshot {i+1}: {len(responses)}')

    # Обработка изображений
    for j, response in enumerate(responses):
        if response.image_type == airsim.ImageType.DepthVis:
            print("DepthVis: Type %d, size %d" %
                  (response.image_type, len(response.image_data_uint8)))
            filepath = os.path.join(
                output_dir, f'py_depthvis_{i+1}_{j}.png')
            airsim.write_file(
                os.path.normpath(filepath),
                response.image_data_uint8
            )
        elif response.image_type == airsim.ImageType.DepthPlanar:
            print("DepthPlanar: Type %d, size %d" %
                  (response.image_type, len(response.image_data_float)))
            filepath = os.path.join(
                output_dir, f'py_depthplanar_{i+1}_{j}.pfm')
            airsim.write_pfm(
                os.path.normpath(filepath),
                airsim.get_pfm_array(response)
            )
        else:
            print("Unknown image type %d" % response.image_type)

    # Обновляем текущую позицию дрона (для следующего шага)
    current_x = step_x
    current_y = step_y

# Финальное перемещение в точку назначения (уже на нужной высоте)
print(f"Moving to final target: X={TARGET_X}, Y={TARGET_Y}, "
      f"Z={target_altitude}")
client.moveToPositionAsync(
    TARGET_X, TARGET_Y, target_altitude, VELOCITY, timeout_sec=10,
    drivetrain=airsim.DrivetrainType.ForwardOnly,
    yaw_mode=airsim.YawMode(False, 0)).join()

# Ожидаем несколько секунд, чтобы дрон стабилизировался на месте
time.sleep(5)

# После завершения полета, приземляемся и выключаем API контроль
client.landAsync().join()
client.armDisarm(False)
client.enableApiControl(False)
print("Done!")
