import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import matplotlib.pyplot as plt
import copy

#原始的DQN算法 出处是2015年的Nature
#我只修缮了一下NatureDQN和DQN_with_all两个文件，其余的DQN文件我没有做详细的修缮和注释，但是都大差不大。
#备注：凡是函数后面带_patrol的均为编队移动模型的相关函数，对应GUI界面的Patrol按钮

#超参数
NUM = 1
NUM_ACTIONS = 8
NUM_STATES = 15
NUM_Patrol_STATES = 15
NUM_REWARDS = 1
test_flag = True #False代表重新训练模型 True代表加载已有的模型
log_dir = "./model/dqn_eval_net_model.pth" #编队集合模型的存放地址
log_dir_patrol = "./model/dqn_patrol_eval_net_model.pth" #编队移动模型的存放地址


class Net(nn.Module):
    """docstring for Net"""
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(NUM_STATES, 256) #输入层
        self.fc1.weight.data.normal_(0,0.1) #参数初始化,normal是指正态分布
        self.fc2 = nn.Linear(256,128)  #隐藏层
        self.fc2.weight.data.normal_(0,0.1)
        self.out = nn.Linear(128,NUM_ACTIONS)
        self.out.weight.data.normal_(0,0.1)

    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        action_prob = self.out(x)
        return action_prob

class Net_Patrol(nn.Module):
    """docstring for Net"""
    def __init__(self):
        super(Net_Patrol, self).__init__()
        self.fc1 = nn.Linear(NUM_STATES, 256) #输入层
        self.fc1.weight.data.normal_(0,0.1) #参数初始化,normal是指正态分布
        self.fc2 = nn.Linear(256,128)  #隐藏层
        self.fc2.weight.data.normal_(0,0.1)
        self.out = nn.Linear(128,NUM_ACTIONS)
        self.out.weight.data.normal_(0,0.1)

    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        action_prob = self.out(x)
        return action_prob


class DQN():
    # hyper-parameters
    BATCH_SIZE = 128
    LR = 0.0002 #学习率
    LR_patrol = 0.001
    GAMMA = 0.9 #折现因子
    EPISILO = 0.90 #探索因子
    EPISILO_PATROL = 0.999
    MEMORY_CAPACITY = 6000 #经验池容量
    Q_NETWORK_ITERATION = 100 #每学习100次就更新一个targetQ网络
    __instance = False #用于开启python单例模式

    eval_net, target_net = Net(), Net() #共享的Q网络和TargetQ网络，如果不想共享，把这两行放入__init__函数内即可
    eval_net_patrol, target_net_patrol = Net_Patrol(), Net_Patrol()

    # __new__开启单例模式
    # def __new__(cls, av_ID):
    #     if not DQN.__instance:
    #         DQN.__instance = object.__new__(cls)
    #     return DQN.__instance

    def __init__ (self, av_num) : #初始化函数
        '''
        初始化函数
        参数：av_num 智能体的数量
        返回值：无
        '''
        super(DQN,self).__init__()
        self.__agentnum = av_num
        self.loss_buff = [] #存放每次的损失函数，当然你也可以直接用tensorboardX查看网络的训练情况
        self.__numSpace=4  # 0-State 1-Action 2-Reward 3-New State 
        self.__numState = NUM_STATES
        self.__numAction = NUM_ACTIONS
        self.__numReward =  NUM_REWARDS

        # self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0 #记录当前神经网络学习了几次了
        self.memory_counter = 0 #记录当前经验池的index
        self.memory = np.zeros((self.MEMORY_CAPACITY, self.__numState * 2 + 2)) #经验池
        # why the NUM_STATE*2 +2
        # When we store the memory, we put the state, action, reward and next_state in the memory
        # here reward and action is a number, state is a ndarray

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.LR) #Adam优化器
        self.loss_func = nn.MSELoss() #均方损失函数 loss_i = (x_i - y_i)^2

        if(test_flag):
            # 加载保存的模型直接进行测试机验证，不进行此模块以后的步骤
            checkpoint = torch.load(log_dir)
            self.eval_net.load_state_dict(checkpoint['model']) #加载网络参数
            self.optimizer.load_state_dict(checkpoint['optimizer']) #加载优化器参数
            self.EPISILO = 1

# —————————————————————————编队移动函数的一些初始化———————————————————————————————————————————————————————————————————————————————————————————
        self.__numState_patrol = NUM_Patrol_STATES
        self.__numAction_patrol = NUM_ACTIONS
        self.__numReward_patrol = NUM_REWARDS

        self.learn_step_counter_patrol = 0
        self.memory_counter_patrol = 0
        self.memory_patrol = np.zeros((self.MEMORY_CAPACITY, self.__numState_patrol * 2 + 2)) #测试一下这个二维数组的效果

        # 加载保存的模型直接进行测试机验证，不进行此模块以后的步骤
        self.optimizer_patrol = torch.optim.Adam(self.eval_net_patrol.parameters(), lr=self.LR_patrol)
        self.loss_func_patrol = nn.MSELoss() #均方损失函数 loss_i = (x_i - y_i)^2

        if(test_flag == True): #如果有训练好的编队移动模型就加载编队移动的模型
            checkpoint_patrol = torch.load(log_dir_patrol)
            self.eval_net_patrol.load_state_dict(checkpoint_patrol['model'])
            self.optimizer_patrol.load_state_dict(checkpoint_patrol['optimizer'])

        else: #如果没有，就加载训练好的编队集合的模型的参数作为预训练模型
            checkpoint_patrol = torch.load(log_dir)
            self.eval_net_patrol.load_state_dict(checkpoint_patrol['model'])
            # for name in self.eval_net_patrol.state_dict():
            #     print(name)
            self.optimizer_patrol.load_state_dict(checkpoint_patrol['optimizer'])
        #如果啥模型也没有就先注释掉这一段吧，或者自己修改一下

    def choose_action(self, state):
        '''
        动作选择函数
        参数：state 智能体的状态(np.array)
        返回值：每个动作对应的值函数Q
        '''
        state = torch.unsqueeze(torch.FloatTensor(state), 0) # get a 1D array
        if np.random.randn() <= self.EPISILO:# greedy policy
            action_value = self.eval_net.forward(state)
            # action = torch.max(action_value, 1)[1].data.numpy() #max(input, dim) -> return[input中的最大值, 最大值的下标]
            # action = action[0] if self.ENV_A_SHAPE == 0 else action.reshape(self.ENV_A_SHAPE)
            return(action_value[0].detach().numpy())

        else: # random policy
            action = np.random.randint(0,self.__numAction)
            # action = action if self.ENV_A_SHAPE ==0 else action.reshape(self.ENV_A_SHAPE)
            ret = np.zeros(self.__numAction)
            ret[action] = 1
            return ret
        # return action


    def store_transition(self, state, action, reward, next_state):
        '''
        存储经验片段的函数
        参数：马尔可夫四元组
        返回值：无
        '''
        if(test_flag):
            return
        transition = np.hstack((state, [action, reward], next_state)) #按列水平拼接在一起

        index = self.memory_counter % self.MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1


    def learn(self):
        '''
        神经网络学习函数
        参数：无
        返回值：无
        '''
        if(test_flag): #如果是测试那么就不需要学习了
            return

        #update the parameters
        if self.learn_step_counter % self.Q_NETWORK_ITERATION ==0:
            self.target_net.load_state_dict(self.eval_net.state_dict()) # return weight && bias
        self.learn_step_counter+=1


        #sample batch from memory
        sample_index = np.random.choice(self.MEMORY_CAPACITY, self.BATCH_SIZE)
        batch_memory = self.memory[sample_index, :]
        batch_state = torch.FloatTensor(batch_memory[:, :self.__numState])
        batch_action = torch.LongTensor(batch_memory[:, self.__numState:self.__numState+1].astype(int)) #astype array改变类型
        batch_reward = torch.FloatTensor(batch_memory[:, self.__numState+1:self.__numState+2])
        batch_next_state = torch.FloatTensor(batch_memory[:,-self.__numState:])


        #q_eval
        q_eval = self.eval_net(batch_state).gather(1, batch_action) #gather(input, dim, index, out=None, sparse_grad=False) → Tensor
        q_next = self.target_net(batch_next_state).detach() #根据状态s'选择a'tar_net下两个动作的Q值
        q_target = batch_reward + self.GAMMA * q_next.max(1)[0].view(self.BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target)

        k = loss.detach().numpy()
        self.loss_buff.append(float(k)) #存储损失函数
        self.optimizer.zero_grad() #d_weights = [0] * n 即将梯度初始化为零（因为一个batch的loss关于weight的导数是所有sample的loss关于weight的导数的累加和）
        loss.backward()
        self.optimizer.step()

    def save_model(self, epoch : int):
        '''
        保存模型
        参数：epoch：这个我没用到忽略即可
        返回值：无
        '''
        state = {'model':self.eval_net.state_dict(), 'optimizer':self.optimizer.state_dict(), 'epoch':epoch} #只保存模型参数
        # state = {'model':self.eval_net, 'optimizer':self.optimizer, 'epoch':epoch}#保存整个模型架构
        torch.save(state, log_dir)
# ————————————————————————————————————————————————————————————————————————————————————————————————————————————————————

    def choose_action_patrol(self, state):

        state = torch.unsqueeze(torch.FloatTensor(state), 0) # get a 1D array
        if np.random.randn() <= self.EPISILO_PATROL:# greedy policy
            action_value = self.eval_net_patrol.forward(state)
            # action = torch.max(action_value, 1)[1].data.numpy() #max(input, dim) -> return[input中的最大值, 最大值的下标]
            # action = action[0] if self.ENV_A_SHAPE == 0 else action.reshape(self.ENV_A_SHAPE)
            return(action_value[0].detach().numpy())

        else: # random policy
            action = np.random.randint(0,self.__numAction)
            # action = action if self.ENV_A_SHAPE ==0 else action.reshape(self.ENV_A_SHAPE)
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

        q_eval = self.eval_net(batch_state).gather(1, batch_action) #gather(input, dim, index, out=None, sparse_grad=False) → Tensor
        q_next = self.target_net(batch_next_state).detach() #根据状态s'选择a'tar_net下两个动作的Q值
        q_target = batch_reward + self.GAMMA * q_next.max(1)[0].view(self.BATCH_SIZE, 1)
        loss = self.loss_func_patrol(q_eval, q_target)

        k = loss.detach().numpy()
        self.loss_buff.append(float(k)) #存储损失函数
        self.optimizer_patrol.zero_grad()#d_weights = [0] * n 即将梯度初始化为零（因为一个batch的loss关于weight的导数是所有sample的loss关于weight的导数的累加和）
        loss.backward()
        self.optimizer_patrol.step()
        
    def save_model_patrol(self, epoch : int):
        state = {'model':self.eval_net_patrol.state_dict(), 'optimizer':self.optimizer_patrol.state_dict(), 'epoch':epoch}
        # state = {'model':self.eval_net_patrol, 'optimizer':self.optimizer.state_dict(), 'epoch':epoch}
        torch.save(state, log_dir_patrol)




