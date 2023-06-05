#!/usr/bin/env python
# coding: utf-8

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from collections import namedtuple, deque
import numpy as np
import random
import matplotlib.pyplot as plt
import imutils
import time

from jetbot import Robot, Camera, bgr8_to_jpeg
from RGB_Lib import Programing_RGB
from Battery_Vol_Lib import BatteryLevel
import os
import RPi.GPIO as GPIO
import time

class Env():
    def __init__(self, dim_state, dim_action, action_min, action_max, robot, camera, goal):
        # 动作维度，左右轮速度
        self.dim_state = dim_state        
        self.dim_action = dim_action
        self.action_min = action_min
        self.action_max = action_max
        self.robot = robot
        self.camera = camera
        self.goal = goal

    def from_image_get_pos(self):        
        image = self.camera.value
        img = np.array(image).astype(np.uint8)
        img.resize([720, 720, 3])
        # img = cv2.flip(sensorImage, 0)
        greenLower = (29, 86, 6)
        greenUpper = (64, 255, 255)
        width, height = img.shape[:2]
        blurred = cv2.GaussianBlur(img, (11, 11), 0)        
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)        
        mask = cv2.inRange(hsv, greenLower, greenUpper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        # toc = time.time()
        # save_name = 'mask' + str(toc) + '.png'
        # cv2.imwrite(save_name, mask)

        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)        

        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            # return abs(x - width / 2), abs(y - height * 2 / 3), radius   
            if radius > 15:     
                return x, y, radius, width, height
        return -1, -1, 0, width, height

    def close(self):
        self.robot.stop()

    def step(self, action, i, last_action):        
        self.robot.set_motors(float(action[0]), float(action[1]))
        time.sleep(0.2)
        self.robot.stop() 

        x_r, y_r, radius, w, h = self.from_image_get_pos()
        next_state = [x_r,y_r,radius]
        x_dis, r_dis, reach = self.reach_goal(next_state, w, h)

        # reward = - (x_r - w/2) ** 2 * 0.0001 - abs(radius - 50) ** 2 * 0.01 - i * 0.05
        if radius == 0:
            reward_dis = -200
        else:
            reward_dis = -x_dis ** 2 * 0.0001 - r_dis ** 2 * 0.05 if not reach else 100
        reward_action_diff =  (action[0] - last_action[0]) ** 2 - (action[1] - last_action[1]) ** 2
        reward_action =  -action[0]**2 - action[1]**2
        reward_forward = action[0]/abs(action[0]) + action[1]/abs(action[1])
        reward = reward_dis + 0.0 * reward_action_diff + 0.0 * reward_action + 5 * reward_forward
        return next_state, reward, reach

    def reach_goal(self, state, w, h):
        x_dis = abs(state[0] - w/2)
        r_dis = abs(state[2] - self.goal[1])
        if x_dis < self.goal[0] and r_dis < self.goal[2]:
            reach = True
        else:
            reach = False
        return x_dis, r_dis, reach



class ReplayBuffer():
    def __init__(self, buffer_maxlen, device):
        self.buffer = deque(maxlen=buffer_maxlen)
        self.device = device

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

        return torch.FloatTensor(state_list).to(self.device), torch.FloatTensor(action_list).to(self.device), torch.FloatTensor(reward_list).unsqueeze(-1).to(self.device),\
            torch.FloatTensor(next_state_list).to(self.device), torch.FloatTensor(done_list).unsqueeze(-1).to(self.device)

    def buffer_len(self):
        return len(self.buffer)


# Value Net
class ValueNet(nn.Module):
    def __init__(self, state_dim, edge=1e-3):
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
    def __init__(self, state_dim, action_dim, edge=1e-3):
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
    def __init__(self, state_dim, action_dim, device, log_std_min=-20, log_std_max=1, edge=1e-3):
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

        self.device = device

    def forward(self, state):        
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        mean, log_std = self.forward(state)        
        std = log_std.exp()
        normal = Normal(mean, std)        
        z = normal.sample()        
        action = torch.tanh(z).detach().cpu().numpy()        
        print('mean:{}, std:{}, z:{}, action:{}'.format(mean.tolist(), std.tolist(), z.tolist(), action))
        return action

    # Use re-parameterization tick
    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        noise = Normal(0, 1)

        z = noise.sample()
        action = torch.tanh(mean + std * z.to(self.device))
        log_prob = normal.log_prob(mean + std * z.to(self.device)) - torch.log(1 - action.pow(2) + epsilon)

        return action, log_prob


class SAC:
    def __init__(self, env, gamma, tau, buffer_maxlen, value_lr, q_lr, policy_lr, device, load_path=None):

        self.env = env
        self.state_dim = self.env.dim_state
        self.action_dim = self.env.dim_action

        # hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.device = device

        # initialize networks
        if load_path is None:
            self.value_net = ValueNet(self.state_dim).to(self.device)
            self.target_value_net = ValueNet(self.state_dim).to(self.device)
            self.q1_net = SoftQNet(self.state_dim, self.action_dim).to(self.device)
            self.q2_net = SoftQNet(self.state_dim, self.action_dim).to(self.device)
            self.policy_net = PolicyNet(self.state_dim, self.action_dim, self.device).to(self.device)
        else:
            self.value_net = torch.load(os.path.join(load_path, 'value_net_para3.pkl'))
            self.target_value_net = torch.load(os.path.join(load_path, 'target_value_net_para3.pkl'))
            self.q1_net = torch.load(os.path.join(load_path, 'q1_para3.pkl'))
            self.q2_net = torch.load(os.path.join(load_path, 'q2_para3.pkl'))
            self.policy_net = torch.load(os.path.join(load_path, 'policy_net_para3.pkl'))

        # Load the target value network parameters
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

        # Initialize the optimizer
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=value_lr)
        self.q1_optimizer = optim.Adam(self.q1_net.parameters(), lr=q_lr)
        self.q2_optimizer = optim.Adam(self.q2_net.parameters(), lr=q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)

        # Initialize thebuffer
        self.buffer = ReplayBuffer(buffer_maxlen, self.device)

    def get_action(self, state):        
        # if state[2] ==0:
        #     action = np.random.uniform(-1, 1, 2)
        # else:
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




def main(env, agent, Episode, Step, batch_size, RGB, batteryLevel):
    Return = []
    action_range = [env.action_min, env.action_max]    

    for episode in range(Episode):        
        score = 0
        x,y,r,w,h = env.from_image_get_pos() # 重新摆放之后获取目标位置
        state = [x,y,r]
        last_action = [0,0]
        for i in range(Step):                    
            action = agent.get_action(state)
            # action = [min(max(a, env.action_min), env.action_max) for a in action]       
            action = action * env.action_max
            # action[0] = (action[0] / abs(action[0])) * (0.1 + abs(action[0]))
            # action[1] = (action[1] / abs(action[1])) * (0.1 + abs(action[1]))
            next_state, reward, done= env.step(action, i, last_action)
            state_print = ["{:.2f}".format(x) for x in state]
            next_state_print = ["{:.2f}".format(x) for x in next_state]
            print("ep[{}/{}] step[{}/{}] state:{} action:{} next_state:{} reward:{}".format(episode, Episode, i, Step, state_print, action, next_state_print, reward))
            done_mask = 0.0 if done else 1.0
            # if state[2] != 0:
            agent.buffer.push((state, action, reward, next_state, done_mask))
            state = next_state
            last_action = action
            score += reward
            if done:
                break
            if agent.buffer.buffer_len() > 200:
                agent.update(batch_size)
        
        print("                               ")
        print("##################################")
        print("episode:{}, Return:{}, buffer_capacity:{}, reach_goal:{}, step:{}".format(episode, score, agent.buffer.buffer_len(), done, i))
        print("##################################")
        print("                               ")
        Return.append(score)
        RGB.Set_WaterfallLight_RGB()  # 设置灯光
        time.sleep(5.0)  # 等待五秒，重新摆放位置，开始下一轮探索
        RGB.OFF_ALL_RGB()
        if  batteryLevel.Update() == "Battery_Low": 
            RGB.Set_All_RGB(0xFF, 0x00, 0x00)  # 红灯亮起
            print("低电量！")
            # break    
    torch.save(agent.policy_net, "model_weight/policy_net_para3.pkl")
    torch.save(agent.value_net, "model_weight/value_net_para3.pkl")
    torch.save(agent.target_value_net, "model_weight/target_value_net_para3.pkl")
    torch.save(agent.q1_net, "model_weight/q1_para3.pkl")
    torch.save(agent.q2_net, "model_weight/q2_para3.pkl")
    print(Return)
    env.close()


if __name__ == '__main__':            
    dim_state = 3
    dim_action = 2
    action_min = -0.7
    action_max = 0.7
    tau = 0.01
    gamma = 0.99
    q_lr = 1e-3
    value_lr = 1e-3
    policy_lr = 1e-3
    buffer_maxlen = 50000
    goal = [40, 60, 10] # [x坐标与中心的偏差，目标半径，半径的偏差]
    Episode = 2
    Step = 50
    batch_size = 128


    robot = Robot() # 创建机器人
    BEEP_pin = 6 
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(BEEP_pin, GPIO.OUT, initial=GPIO.LOW)
    GPIO.output(BEEP_pin, GPIO.LOW)
    RGB = Programing_RGB() # 设置灯光
    RGB.OFF_ALL_RGB()  # 关闭所有灯光，如果出现流水灯则表示需要重新摆放位置
    batteryLevel = BatteryLevel() # 显示电量
    print(f"电量：{batteryLevel.Update()}")
    camera = Camera.instance(width=720, height=720) # 创建相机实例
    env = Env(dim_state, dim_action, action_min, action_max, robot, camera, goal)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cuda')
    # device = "cpu"
    # Params
    print('device: ', device)    
    
    agent = SAC(env, gamma, tau, buffer_maxlen, value_lr, q_lr, policy_lr, device, load_path='model_weight')
    # agent = SAC(env, gamma, tau, buffer_maxlen, value_lr, q_lr, policy_lr, device)
    main(env, agent, Episode, Step, batch_size, RGB, batteryLevel)