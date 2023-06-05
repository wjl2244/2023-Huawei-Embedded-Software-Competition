import sys
import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from collections import namedtuple, deque
import numpy as np
import sim
import random
import matplotlib.pyplot as plt
import imutils
import warnings
warnings.filterwarnings('ignore')

class Env():
    def __init__(self):
        sim.simxFinish(-1)
        #建立和服务器的连接
        self.clientId = sim.simxStart("127.0.0.1", 19999, True, True, 5000, 5)

        if self.clientId != -1:  #连接成功
            print('connect successfully')
        else:
            print('connect failed')
            sys.exit('Could not connect')

        # 获取物体的句柄
        # 四个轮子的电机句柄
        _, self.fl_motor_handle = sim.simxGetObjectHandle(self.clientId, 'fl', sim.simx_opmode_blocking)
        _, self.fr_motor_handle = sim.simxGetObjectHandle(self.clientId, 'fr', sim.simx_opmode_blocking)
        _, self.br_motor_handle = sim.simxGetObjectHandle(self.clientId, 'br', sim.simx_opmode_blocking)
        _, self.bl_motor_handle = sim.simxGetObjectHandle(self.clientId, 'bl', sim.simx_opmode_blocking)
        # 视觉相机的句柄
        _, self.vision_sensor_handle = sim.simxGetObjectHandle(self.clientId, 'Vision_sensor', sim.simx_opmode_blocking)
        # 跟踪的球体的句柄
        _, self.track_object_handle = sim.simxGetObjectHandle(self.clientId, 'Sphere', sim.simx_opmode_blocking)

        _, self.car_body_handle = sim.simxGetObjectHandle(self.clientId, 'Cuboid', sim.simx_opmode_blocking)
        # 状态维度，x、y、z相对位置 or  x、y、radius
        self.dim_state = 3
        # 动作维度，左右轮速度
        self.dim_action = 2
        self.action_min = -2.0
        self.action_max = 2.0
        _, self.init_orientation = sim.simxGetObjectOrientation(self.clientId,self.car_body_handle,-1,sim.simx_opmode_blocking)
        _, self.init_position = sim.simxGetObjectPosition(self.clientId,self.car_body_handle,-1,sim.simx_opmode_blocking)
    def destroy(self):
        sim.simxFinish(self.clientId)

    def reset(self):
        sim.simxSetObjectPosition(self.clientId, self.car_body_handle,-1 ,self.init_position, sim.simx_opmode_blocking)
        sim.simxSetObjectOrientation(self.clientId, self.car_body_handle,-1 ,self.init_orientation, sim.simx_opmode_blocking)
        sim.simxStartSimulation(self.clientId, sim.simx_opmode_blocking)
        _, object_position = sim.simxGetObjectPosition(self.clientId, self.track_object_handle, self.vision_sensor_handle, sim.simx_opmode_blocking)
        return object_position
    
    def close(self):
        sim.simxStopSimulation(self.clientId, sim.simx_opmode_blocking)

    def step(self, action, i):
        sim.simxSetJointTargetVelocity(self.clientId, self.fl_motor_handle, action[0], sim.simx_opmode_blocking)
        sim.simxSetJointTargetVelocity(self.clientId, self.fr_motor_handle, action[1], sim.simx_opmode_blocking)
        _, object_position = sim.simxGetObjectPosition(self.clientId, self.track_object_handle, self.vision_sensor_handle, sim.simx_opmode_blocking)
        # 基于相对位置的状态输入
        # dis_x = object_position[0] 
        # dis_z = object_position[2] - 0.4
        # reward = -(dis_x**2 + dis_z**2) - int(i/100) * 0.01
        # if abs(dis_x) < 0.05 and abs(dis_z) < 0.05:
        #     done = True
        #     reward = 100
        # else:
        #     done = False
        # 基于图像中心的状态输入
        x_r, y_r, radius = self.get_image()
        reward = - x_r ** 2 * 0.0001 - abs(radius - 50) ** 2 * 0.01 - i * 0.05
        if x_r < 30 and abs(radius - 50) < 10:
            done = True
            reward = 0
        else:
            done = False
        next_state = object_position
        _,linear_speed_fr,anular_speed_fr = sim.simxGetObjectVelocity(self.clientId, self.fr_motor_handle, sim.simx_opmode_blocking)
        _,linear_speed_fl,anular_speed_fl = sim.simxGetObjectVelocity(self.clientId, self.fl_motor_handle, sim.simx_opmode_blocking)
        speed_fr = np.sqrt(linear_speed_fr[0] ** 2 + linear_speed_fr[1] ** 2 + linear_speed_fr[2] ** 2)
        speed_fl = np.sqrt(linear_speed_fl[0] ** 2 + linear_speed_fl[1] ** 2 + linear_speed_fl[2] ** 2)
        reward = reward - 1.0 * (action[0] -speed_fl)**2 - 1.0 * (action[1] - speed_fr)**2

        return next_state, reward, done, _
    
    def get_image(self):
        _, resolution, image = sim.simxGetVisionSensorImage(self.clientId, self.vision_sensor_handle, 0, sim.simx_opmode_blocking)
        sensorImage = np.array(image).astype(np.uint8)
        sensorImage.resize([resolution[1],resolution[0],3])
        img = cv2.flip(sensorImage, 0)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        greenLower = (29, 86, 6)
        greenUpper = (64, 255, 255)        
        width, height = img.shape[:2]
        blurred = cv2.GaussianBlur(img, (11, 11), 0)        
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)    
        mask = cv2.inRange(hsv, greenLower, greenUpper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)    

        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            return abs(x-width/2), abs(y-height*2/3), radius
        else:
            return width, height, 0
    
    # -1表示世界坐标系的位置
    def get_position(self, object_handle, relative_object_handle):
        _, object_position = sim.simxGetObjectPosition(self.clientId, object_handle, relative_object_handle, sim.simx_opmode_blocking)
        return object_position


class ReplayBeffer():
    def __init__(self, buffer_maxlen):
        self.buffer = deque(maxlen=buffer_maxlen)

    def push(self, data):
        self.buffer.append(data)

    def sample(self, batch_size):
        state_list = []
        action_list = []
        reward_list = []
        next_state_list = []
        done_list = []

        batch = random.sample(self.buffer, batch_size)
        for experience in batch:
            s, a, r, n_s, d = experience
            # state, action, reward, next_state, done

            state_list.append(s)
            action_list.append(a)
            reward_list.append(r)
            next_state_list.append(n_s)
            done_list.append(d)

        return torch.FloatTensor(state_list).to(device), \
               torch.FloatTensor(action_list).to(device), \
               torch.FloatTensor(reward_list).unsqueeze(-1).to(device), \
               torch.FloatTensor(next_state_list).to(device), \
               torch.FloatTensor(done_list).unsqueeze(-1).to(device)

    def buffer_len(self):
        return len(self.buffer)

# Value Net
class ValueNet(nn.Module):
    def __init__(self, state_dim, edge=3e-3):
        super(ValueNet, self).__init__()
        self.linear1 = nn.Linear(state_dim, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 2)

        self.linear3.weight.data.uniform_(-edge, edge)
        self.linear3.bias.data.uniform_(-edge, edge)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x


# Soft Q Net
class SoftQNet(nn.Module):
    def __init__(self, state_dim, action_dim, edge=3e-3):
        super(SoftQNet, self).__init__()
        self.linear1 = nn.Linear(state_dim + action_dim, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 2)

        self.linear3.weight.data.uniform_(-edge, edge)
        self.linear3.bias.data.uniform_(-edge, edge)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x

# Policy Net
class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, log_std_min=-20, log_std_max=2, edge=3e-3):
        super(PolicyNet, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(state_dim, 256)
        self.linear2 = nn.Linear(256, 256)

        self.mean_linear = nn.Linear(256, action_dim)
        self.mean_linear.weight.data.uniform_(-edge, edge)
        self.mean_linear.bias.data.uniform_(-edge, edge)

        self.log_std_linear = nn.Linear(256, action_dim)
        self.log_std_linear.weight.data.uniform_(-edge, edge)
        self.log_std_linear.bias.data.uniform_(-edge, edge)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def action(self, state):
        state = torch.FloatTensor(state).to(device)
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)

        z = normal.sample()
        action = torch.tanh(z).detach().cpu().numpy()

        return action

    # Use re-parameterization tick
    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        noise = Normal(0, 1)

        z = noise.sample()
        action = torch.tanh(mean + std * z.to(device))
        log_prob = normal.log_prob(mean + std * z.to(device)) - torch.log(1 - action.pow(2) + epsilon)

        return action, log_prob

class SAC:
    def __init__(self, env, gamma, tau, buffer_maxlen, value_lr, q_lr, policy_lr):

        self.env = env
        self.state_dim = env.dim_state
        self.action_dim = env.dim_action

        # hyperparameters
        self.gamma = gamma
        self.tau = tau

        # initialize networks
        self.value_net = ValueNet(self.state_dim).to(device)
        self.target_value_net = ValueNet(self.state_dim).to(device)
        self.q1_net = SoftQNet(self.state_dim, self.action_dim).to(device)
        self.q2_net = SoftQNet(self.state_dim, self.action_dim).to(device)
        self.policy_net = PolicyNet(self.state_dim, self.action_dim).to(device)

        # Load the target value network parameters
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

        # Initialize the optimizer
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=value_lr)
        self.q1_optimizer = optim.Adam(self.q1_net.parameters(), lr=q_lr)
        self.q2_optimizer = optim.Adam(self.q2_net.parameters(), lr=q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)

        # Initialize thebuffer
        self.buffer = ReplayBeffer(buffer_maxlen)

    def get_action(self, state):
        action = self.policy_net.action(state)
        return action

    def update(self, batch_size):
        state, action, reward, next_state, done = self.buffer.sample(batch_size)
        new_action, log_prob = self.policy_net.evaluate(state)
        
        # V value loss
        value = self.value_net(state)
        new_q1_value = self.q1_net(state, new_action)
        new_q2_value = self.q2_net(state, new_action)
        next_value = torch.min(new_q1_value, new_q2_value) - log_prob
        value_loss = F.mse_loss(value, next_value.detach())

        # Soft q  loss
        q1_value = self.q1_net(state, action)
        q2_value = self.q2_net(state, action)
        target_value = self.target_value_net(next_state)
        target_q_value = reward + done * self.gamma * target_value
        q1_value_loss = F.mse_loss(q1_value, target_q_value.detach())
        q2_value_loss = F.mse_loss(q2_value, target_q_value.detach())

        # Policy loss
        policy_loss = (log_prob - torch.min(new_q1_value, new_q2_value)).mean()

        # Update Policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Update v
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # Update Soft q
        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        q1_value_loss.backward()
        q2_value_loss.backward()
        self.q1_optimizer.step()
        self.q2_optimizer.step()

        # Update target networks
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

def main(env, agent, Episode, batch_size):
    Return = []
    action_range = [env.action_min, env.action_max]

    for episode in range(Episode):
        score = 0
        state = env.reset()
        for i in range(200):
            action = agent.get_action(state)
            # action output range[-1,1],expand to allowable range
            action_in =  action * (action_range[1] - action_range[0]) / 2.0 +  (action_range[1] + action_range[0]) / 2.0

            next_state, reward, done, _ = env.step(action_in, i)
            done_mask = 0.0 if done else 1.0
            agent.buffer.push((state, action, reward, next_state, done_mask))
            state = next_state

            score += reward
            if done:
                break
            if agent.buffer.buffer_len() > 200:
                agent.update(batch_size)

        print("episode:{}, Return:{}, buffer_capacity:{}".format(episode, score, agent.buffer.buffer_len()))
        Return.append(score)
        score = 0
    env.close()
    plt.plot(Return)
    plt.ylabel('Return')
    plt.xlabel("Episode")
    plt.grid(True)
    plt.show()

    torch.save(agent.policy_net, "policy-center.pkl")

if __name__ == '__main__':
    env = Env()
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # Params
    tau = 0.01
    gamma = 0.99
    q_lr = 3e-3
    value_lr = 3e-3
    policy_lr = 3e-3
    buffer_maxlen = 50000

    Episode = 100
    batch_size = 128

    agent = SAC(env, gamma, tau, buffer_maxlen, value_lr, q_lr, policy_lr)
    main(env, agent, Episode, batch_size)