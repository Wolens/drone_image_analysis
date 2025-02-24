import airsim
import os

# Подключение к AirSim
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# Взлет и перемещение
client.takeoffAsync().join()
client.moveToPositionAsync(-25, 23, -50, 5).join()

# Запрос изображений
responses = client.simGetImages([
    airsim.ImageRequest("0", airsim.ImageType.DepthVis),
    airsim.ImageRequest("1", airsim.ImageType.DepthPlanar, True)])
print('Retrieved images: %d', len(responses))

# Создаем директорию "output" в текущей директории скрипта, если она не существует
output_dir = os.path.join(os.getcwd(), "output")  # Use a relative path
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Обработка изображений
for i, response in enumerate(responses):
    if response.image_type == airsim.ImageType.DepthVis:
        print("DepthVis: Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
        filepath = os.path.join(output_dir, f'py_depthvis_{i}.png')
        airsim.write_file(os.path.normpath(filepath), response.image_data_uint8)
    elif response.image_type == airsim.ImageType.DepthPlanar:
        print("DepthPlanar: Type %d, size %d" % (response.image_type, len(response.image_data_float)))
        filepath = os.path.join(output_dir, f'py_depthplanar_{i}.pfm')
        airsim.write_pfm(os.path.normpath(filepath), airsim.get_pfm_array(response))
    else:
        print("Unknown image type %d" % response.image_type)