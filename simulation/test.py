import sim
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import cv2

class RL_car(object):
    def __init__(self):
        # 关闭之前所有的连接
        sim.simxFinish(-1)
        #建立和服务器的连接
        self.clientId = sim.simxStart("127.0.0.1", 19999, True, True, 5000, 5)
        if self.clientId != -1:  #连接成功
            print('connect successfully')
        else:
            print('connect failed')
            sys.exit('Could not connect')

        self.get_handles()

    def get_handles(self):
        # 四个轮子的电机句柄
        _, self.fl_motor_handle = sim.simxGetObjectHandle(self.clientId, 'fl', sim.simx_opmode_blocking)
        _, self.fr_motor_handle = sim.simxGetObjectHandle(self.clientId, 'fr', sim.simx_opmode_blocking)
        _, self.br_motor_handle = sim.simxGetObjectHandle(self.clientId, 'br', sim.simx_opmode_blocking)
        _, self.bl_motor_handle = sim.simxGetObjectHandle(self.clientId, 'bl', sim.simx_opmode_blocking)
        # 视觉相机的句柄
        _, self.vision_sensor_handle = sim.simxGetObjectHandle(self.clientId, 'Vision_sensor', sim.simx_opmode_blocking)
        # 跟踪的球体的句柄
        _, self.track_object_handle = sim.simxGetObjectHandle(self.clientId, 'Sphere', sim.simx_opmode_blocking)

    def destroy(self):
        sim.simxStopSimulation(self.clientId, sim.simx_opmode_blocking)

    def reset(self):
        # 重启仿真
        stop = sim.simxStopSimulation(self.clientId, sim.simx_opmode_blocking)
        start = sim.simxStartSimulation(self.clientId, sim.simx_opmode_blocking)
    
    def step(self, action):
        sim.simxSetJointTargetVelocity(self.clientId, self.fl_motor_handle, action[0], sim.simx_opmode_blocking)
        sim.simxSetJointTargetVelocity(self.clientId, self.fr_motor_handle, action[1], sim.simx_opmode_blocking)
    
    def get_position(self, object_handle, relative_object_handle):
        _, object_position = sim.simxGetObjectPosition(self.clientId, object_handle, relative_object_handle, sim.simx_opmode_blocking)
        return object_position

    def get_image(self):
        _, resolution, image = sim.simxGetVisionSensorImage(self.clientId, self.vision_sensor_handle, 0, sim.simx_opmode_blocking)
        sensorImage = np.array(image).astype(np.uint8)
        sensorImage.resize([resolution[1],resolution[0],3])
        sensorImage = cv2.flip(sensorImage, 0)
        return sensorImage
    
    def sphere_move(self):
        sim.simxSetObjectPosition()
        pass

position_info = []

car = RL_car()
start_time = time.time()
cnt = 0
while time.time() - start_time < 50:
    car.step([1, 1])
    position_each = [str(cnt)]
    object_position = car.get_position(car.track_object_handle, -1)
    for i in range(3):
        position_each.append(object_position[i])
    relative_position = car.get_position(car.track_object_handle, car.vision_sensor_handle)
    for i in range(3):
        position_each.append(relative_position[i])
    sensorImage = car.get_image()
    cv2.imshow('image',sensorImage)
    cv2.waitKey(1)
    cv2.imwrite("./image3/" + str(cnt) + ".png", sensorImage)
    cnt += 1
    position_info.append(position_each)
sim.simxAddStatusbarMessage(car.clientId,'Hello CoppeliaSim!',sim.simx_opmode_oneshot)
sim.simxFinish(car.clientId)

np.save("data3", position_info)
print("end -------------------------")