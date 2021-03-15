import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import matplotlib.pyplot as plt
import copy
from SumTree import SumTree
from SumTree import Memory
from torch.autograd import Variable

#带有优先回放的Dueling-DDQN算法
#我只修缮了一下NatureDQN和DQN_with_all两个文件，其余的DQN文件我没有做详细的修缮和注释，但是都大差不差。
#备注：凡是函数后面带_patrol的均为编队移动模型的相关函数，对应GUI界面的Patrol按钮。注意优先采样回放机制，只在编队集合任务上使用了，编队移动任务我并未使用。


NUM = 1
NUM_ACTIONS = 8
NUM_STATES = 15
NUM_Patrol_STATES = 15
NUM_REWARDS = 1
test_flag = False
log_dir = "./model/dqn_eval_net_model.pth"
log_dir_patrol = "./model/dqn_patrol_eval_net_model.pth"


class Net(nn.Module):
    """docstring for Net"""
    def __init__(self):
        # DuelingDQN的核心思想：Dueling网络结构
        super(Net, self).__init__()
        self.fc1_adv = nn.Linear(NUM_STATES, 256)
        self.fc1_val = nn.Linear(NUM_STATES, 256)

        self.fc2_adv = nn.Linear(256, 128)
        self.fc2_val = nn.Linear(256, 128)

        self.out_adv = nn.Linear(128, NUM_ACTIONS)
        self.out_val = nn.Linear(128, 1)
        self.relu = nn.ReLU()

    def forward(self,x):
        adv = F.relu(self.fc1_adv(x))
        val = F.relu(self.fc1_val(x))

        adv = F.relu(self.fc2_adv(adv))
        val = F.relu(self.fc2_val(val))

        adv = self.out_adv(adv)
        val = self.out_val(val).expand(x.size(0), NUM_ACTIONS)

        action_prob = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), NUM_ACTIONS)
        return action_prob

class Net_Patrol(nn.Module):
    """docstring for Net"""
    def __init__(self):
        super(Net_Patrol, self).__init__()
        self.fc1_adv = nn.Linear(NUM_Patrol_STATES, 256)
        self.fc1_val = nn.Linear(NUM_Patrol_STATES, 256)

        self.fc2_adv = nn.Linear(256, 128)
        self.fc2_val = nn.Linear(256, 128)

        self.out_adv = nn.Linear(128, NUM_ACTIONS)
        self.out_val = nn.Linear(128, 1)
        self.relu = nn.ReLU()

    def forward(self,x):
        adv = F.relu(self.fc1_adv(x))
        val = F.relu(self.fc1_val(x))

        adv = F.relu(self.fc2_adv(adv))
        val = F.relu(self.fc2_val(val))

        adv = self.out_adv(adv)
        val = self.out_val(val).expand(x.size(0), NUM_ACTIONS)

        action_prob = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), NUM_ACTIONS)
        return action_prob


class DQN():
    # hyper-parameters
    BATCH_SIZE = 128#提高一点batch-size
    LR = 0.001 #学习率
    LR_patrol = 0.001
    GAMMA = 0.9 #折现因子
    EPISILO = 0.90
    EPISILO_PATROL = 0.999
    MEMORY_CAPACITY = 6000
    Q_NETWORK_ITERATION = 100
    ENV_A_SHAPE = 0
    prioritized = False
    __instance = False
    __Trace = False

    eval_net, target_net = Net(), Net()
    eval_net_patrol, target_net_patrol = Net_Patrol(), Net_Patrol()
    # __new__开启单例模式
    # def __new__(cls, av_ID):
    #     if not DQN.__instance:
    #         DQN.__instance = object.__new__(cls)
    #     return DQN.__instance

    def __init__ (self, av_num) : #初始化函数
        super(DQN,self).__init__()
        self.__agentnum = av_num
        self.loss_buff = []
        self.__numSpace=4  # 0-State 1-Action 2-Reward 3-New State 
        self.__numState = NUM_STATES
        self.__numAction = NUM_ACTIONS
        self.__numReward =  NUM_REWARDS

        # self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = Memory((self.MEMORY_CAPACITY)) #优先回放经验池，具体实现见SumTree文件
        # why the NUM_STATE*2 +2
        # When we store the memory, we put the state, action, reward and next_state in the memory
        # here reward and action is a number, state is a ndarray
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.LR)
        self.loss_func = nn.MSELoss() #均方损失函数 loss_i = (x_i - y_i)^2

        if(test_flag):
            # 加载保存的模型直接进行测试机验证，不进行此模块以后的步骤
            checkpoint = torch.load(log_dir)
            self.eval_net.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            epochs = checkpoint['epoch']
            self.EPISILO = 1


# ————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
        self.__numState_patrol = NUM_Patrol_STATES
        self.__numAction_patrol = NUM_ACTIONS
        self.__numReward_patrol = NUM_REWARDS

        self.learn_step_counter_patrol = 0
        self.memory_counter_patrol = 0
        self.memory_patrol = np.zeros((self.MEMORY_CAPACITY, self.__numState_patrol * 2 + 2)) #测试一下这个二维数组的效果

        # 加载保存的模型直接进行测试机验证，不进行此模块以后的步骤

        self.optimizer_patrol = torch.optim.Adam(self.eval_net_patrol.parameters(), lr=self.LR_patrol)
        self.loss_func_patrol = nn.MSELoss() #均方损失函数 loss_i = (x_i - y_i)^2
        if(test_flag == True):
            checkpoint_patrol = torch.load(log_dir_patrol)
            self.eval_net_patrol.load_state_dict(checkpoint_patrol['model'])
            self.optimizer_patrol.load_state_dict(checkpoint_patrol['optimizer'])

        else:
            checkpoint_patrol = torch.load(log_dir)
            self.eval_net_patrol.load_state_dict(checkpoint_patrol['model'])
            self.optimizer_patrol.load_state_dict(checkpoint_patrol['optimizer'])






    def choose_action(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0) # get a 1D array

        if np.random.randn() <= self.EPISILO:# greedy policy
            action_value = self.eval_net.forward(state)
            return(action_value[0].detach().numpy())

        else: # random policy
            action = np.random.randint(0,self.__numAction)
            ret = np.zeros(self.__numAction)
            ret[action] = 1
            return ret
        # return action


    def store_transition(self, state, action, reward, next_state):
        if(test_flag):
            return
        transition = np.hstack((state, [action, reward], next_state)) #按列水平拼接在一起
        self.memory.store(transition)
        self.memory_counter += 1

    def learn(self):
        if(test_flag):
            return
        #update the parameters
        if self.learn_step_counter % self.Q_NETWORK_ITERATION ==0:
            self.target_net.load_state_dict(self.eval_net.state_dict()) # return weight && bias
        self.learn_step_counter+=1


        tree_idx, batch_memory, ISWeights = self.memory.sample(self.BATCH_SIZE) #从经验池中优先取样

        batch_state = torch.FloatTensor(batch_memory[:, :self.__numState])
        batch_action = torch.LongTensor(batch_memory[:, self.__numState:self.__numState+1].astype(int)) #astype array改变类型
        batch_reward = torch.FloatTensor(batch_memory[:, self.__numState+1:self.__numState+2])
        batch_next_state = torch.FloatTensor(batch_memory[:,-self.__numState:])

        #q_eval
        q_eval = self.eval_net(batch_state).gather(1, batch_action) #gather(input, dim, index, out=None, sparse_grad=False) → Tensor
        q_eval1 = self.eval_net(batch_next_state).detach() #Double DQN的核心思想，用Q网络中最大值的下标去取TargetQ网络中的Q值
        max_index = q_eval1.max(1)[1].squeeze() #eval网络argmax Q的下标
        q_next = self.target_net(batch_next_state).detach() #targetQ网络的动作的Q值

        q_target = batch_reward + self.GAMMA * q_next[torch.arange(0,self.BATCH_SIZE),max_index].view(self.BATCH_SIZE, 1)

        abs_errors = abs(q_target - q_eval).detach()
        self.memory.batch_update(tree_idx, abs_errors)

        loss = self.loss_func(q_eval, q_target)
        k = loss.detach().numpy()
        self.loss_buff.append(float(k))
        self.optimizer.zero_grad()#d_weights = [0] * n 即将梯度初始化为零（因为一个batch的loss关于weight的导数是所有sample的loss关于weight的导数的累加和）
        loss.backward()
        self.optimizer.step()

    def save_model(self, epoch : int):
        state = {'model':self.eval_net.state_dict(), 'optimizer':self.optimizer.state_dict(), 'epoch':epoch}
        # state = {'model':self.eval_net, 'optimizer':self.optimizer, 'epoch':epoch}
        torch.save(state, log_dir)
# ————————————————————————————————————————————————————————————————————————————————————————————————————————————————————

    def choose_action_patrol(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0) # get a 1D array
        if np.random.randn() <= self.EPISILO_PATROL:# greedy policy
            action_value = self.eval_net_patrol.forward(state)
            return(action_value[0].detach().numpy())

        else: # random policy
            action = np.random.randint(0,self.__numAction)
            ret = np.zeros(self.__numAction)
            ret[action] = 1
            return ret

    def store_transition_patrol(self, state, action, reward, next_state):
        transition = np.hstack((state, [action, reward], next_state)) #按列水平拼接在一起
        index = self.memory_counter_patrol % self.MEMORY_CAPACITY
        self.memory_patrol[index, :] = transition
        self.memory_counter_patrol += 1


    def learn_patrol(self):
        #update the parameters
        if self.learn_step_counter_patrol % self.Q_NETWORK_ITERATION ==0:
            self.target_net_patrol.load_state_dict(self.eval_net_patrol.state_dict()) # return weight && bias
        self.learn_step_counter_patrol += 1

        #sample batch from memory
        sample_index = np.random.choice(self.MEMORY_CAPACITY, self.BATCH_SIZE)
        batch_memory = self.memory_patrol[sample_index, :]
        batch_state = torch.FloatTensor(batch_memory[:, :self.__numState_patrol])
        batch_action = torch.LongTensor(batch_memory[:, self.__numState_patrol:self.__numState_patrol+1].astype(int)) #astype array改变类型
        batch_reward = torch.FloatTensor(batch_memory[:, self.__numState_patrol+1:self.__numState_patrol+2])
        batch_next_state = torch.FloatTensor(batch_memory[:,-self.__numState_patrol:])

        #q_eval
        q_eval = self.eval_net_patrol(batch_state).gather(1, batch_action) #gather(input, dim, index, out=None, sparse_grad=False) → Tensor
        q_eval1 = self.eval_net_patrol(batch_next_state).detach()
        max_index = q_eval1.max(1)[1].squeeze() #eval网络argmax Q的下标
        q_next = self.target_net_patrol(batch_next_state).detach() #根据状态s'选择a'tar_net下两个动作的Q值
        q_target = batch_reward + self.GAMMA * q_next[torch.arange(0,self.BATCH_SIZE),max_index].view(self.BATCH_SIZE, 1)
        loss = self.loss_func_patrol(q_eval, q_target)
        k = loss.detach().numpy()
        self.loss_buff.append(float(k))
        self.optimizer_patrol.zero_grad()#d_weights = [0] * n 即将梯度初始化为零（因为一个batch的loss关于weight的导数是所有sample的loss关于weight的导数的累加和）
        loss.backward()
        self.optimizer_patrol.step()

    def save_model_patrol(self, epoch : int):
        state = {'model':self.eval_net_patrol.state_dict(), 'optimizer':self.optimizer_patrol.state_dict(), 'epoch':epoch}
        # state = {'model':self.eval_net_patrol, 'optimizer':self.optimizer.state_dict(), 'epoch':epoch}
        torch.save(state, log_dir_patrol)



