import airsim
import os
import time
import math

# --- Настройки ---
TARGET_X = -32
TARGET_Y = 4
TARGET_Z = -30
VELOCITY = 5
OFFSET_Z = 1.2
BOTTOM_CAMERA = "3"
NUM_SNAPSHOTS = 10


# --- Функции для работы с AirSim ---

def connect_to_airsim():
    """Подключается к AirSim и выполняет базовые настройки."""
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    return client


def takeoff_and_move_to_altitude(client, altitude):
    """Взлетает и поднимает дрон на заданную высоту."""
    client.takeoffAsync().join()
    current_pose = client.simGetVehiclePose()
    current_x = current_pose.position.x_val
    current_y = current_pose.position.y_val
    print(f"Moving vertically to altitude: {altitude}")
    client.moveToPositionAsync(
        current_x, current_y, altitude, VELOCITY,
        timeout_sec=10).join()


def calculate_flight_parameters(client):
    """Вычисляет параметры полета (расстояние, время, интервал)."""
    current_pose = client.simGetVehiclePose()
    current_x = current_pose.position.x_val
    current_y = current_pose.position.y_val
    distance_to_target = math.sqrt(
        (TARGET_X - current_x)**2 + (TARGET_Y - current_y)**2)
    flight_time = distance_to_target / VELOCITY
    time_interval = flight_time / NUM_SNAPSHOTS
    return current_x, current_y, distance_to_target, time_interval


def move_to_snapshot_position(client, current_x, current_y,
                             target_altitude, snapshot_index,
                             num_snapshots):
    """Перемещает дрон к позиции для захвата снимка."""
    step_x = current_x + (TARGET_X - current_x) * (snapshot_index + 1) / num_snapshots
    step_y = current_y + (TARGET_Y - current_y) * (snapshot_index + 1) / num_snapshots
    print(f"Moving to: X={step_x}, Y={step_y}, Z={target_altitude}")
    client.moveToPositionAsync(
        step_x, step_y, target_altitude, VELOCITY,
        timeout_sec=10,
        drivetrain=airsim.DrivetrainType.ForwardOnly,
        yaw_mode=airsim.YawMode(False, 0)).join()
    return step_x, step_y


def capture_and_save_images(client, snapshot_index, output_dir):
    """Захватывает и сохраняет изображения."""
    responses = client.simGetImages([
        airsim.ImageRequest(BOTTOM_CAMERA, airsim.ImageType.DepthVis),
        airsim.ImageRequest("1", airsim.ImageType.DepthPlanar, True)])
    print(f'Retrieved images for snapshot {snapshot_index+1}:'
          f' {len(responses)}')
    for j, response in enumerate(responses):
        if response.image_type == airsim.ImageType.DepthVis:
            print("DepthVis: Type %d, size %d" %
                  (response.image_type, len(response.image_data_uint8)))
            filepath = os.path.join(
                output_dir, f'py_depthvis_{snapshot_index+1}_{j}.png')
            airsim.write_file(
                os.path.normpath(filepath),
                response.image_data_uint8
            )
        elif response.image_type == airsim.ImageType.DepthPlanar:
            print("DepthPlanar: Type %d, size %d" %
                  (response.image_type, len(response.image_data_float)))
            filepath = os.path.join(
                output_dir, f'py_depthplanar_{snapshot_index+1}_{j}.pfm')
            airsim.write_pfm(
                os.path.normpath(filepath),
                airsim.get_pfm_array(response)
            )
        else:
            print("Unknown image type %d" % response.image_type)


def move_to_target(client, target_altitude):
    """Перемещает дрон в финальную целевую точку."""
    print(f"Moving to final target: X={TARGET_X}, Y={TARGET_Y}, "
          f"Z={target_altitude}")
    client.moveToPositionAsync(
        TARGET_X, TARGET_Y, target_altitude, VELOCITY,
        timeout_sec=10,
        drivetrain=airsim.DrivetrainType.ForwardOnly,
        yaw_mode=airsim.YawMode(False, 0)).join()


def land_and_disarm(client):
    """Приземляет дрон и отключает управление."""
    time.sleep(5) # Wait to stablize
    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)
    print("Done!")


def create_output_directory():
    """Создает директорию для сохранения изображений."""
    output_dir = os.path.join(os.getcwd(), "output")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir


# --- Основной код ---
if __name__ == "__main__":
    try:
        client = connect_to_airsim()
        output_dir = create_output_directory()
        target_altitude = TARGET_Z + OFFSET_Z
        takeoff_and_move_to_altitude(client, target_altitude)
        current_x, current_y, distance_to_target, time_interval = (
            calculate_flight_parameters(client))

        for i in range(NUM_SNAPSHOTS):
            current_x, current_y = move_to_snapshot_position(
                client, current_x, current_y, target_altitude, i,
                NUM_SNAPSHOTS)
            capture_and_save_images(client, i, output_dir)
        move_to_target(client, target_altitude)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if 'client' in locals():  # Ensure client is defined before using it
            land_and_disarm(client)