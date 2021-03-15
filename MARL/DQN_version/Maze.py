import random
import pandas as pd
import numpy as np
from pprint import pprint
import sys
import copy
from math import pi,sqrt

def gaussian(x, sigma = 0.25, u = 0) -> float:
    x = x / 10
    y = np.exp(-(x - u) ** 2 / (2 * sigma ** 2))
    return y

class Maze():
    size = 16
    def __init__(self, ag_num ):
        '''
        构造函数
        参数：ag_num 智能体的数量
        '''
            #self.size=200
        # self.size = 16  #地图大小,Final修饰的成员变量终身不变，并且必须提前赋值
        self.numMines = 15#雷的数量
        self.binarySonar = False #声纳是否是二进制，历史残留，没卵用
        self.Sound = 5 #智能体的探测范围
        self.stop_reward = 0.0 #用于惩罚，防止智能体一直在地图边缘卡着
        self.rho = 0.8 #初始的挥发系数 ρ大了有利于全局搜索 ρ小了有利于收敛速度，ρ会随着时间的流逝而逐渐减小
        self.rho_min = 0.2 #挥发系数的最小值
        self.Diffuse_rate = 0.2 #扩散系数
        self.__agent_num = ag_num #Agent的数量


        self.__current = [] 
        self.__Attractor_list = [] #
        self.__prev_current = []
        self.__target = []
        self.__target_list = []
        self.__currentBearing = [] #现在的方位
        self.__AttractorBearing = []
        self.__prev_bearing = []#之前的方位
        self.__targetBearing = []
        self.__sonar = [] #声纳的坐标吧(x,y)
        self.__av_sonar = []
        self.__range = []
        self.__mines = []

        self.__avs = []
        self.__Pheromone_map = []
        self.__target_map = []
        self.__end_state = []
        self.__conflict_state = []
        self.refreshMaze( ag_num )
        # self.refreshMaze_patrol( ag_num )

    def get_agentnum(self, agt : int):
        return self.__agent_num

    def set_i_conflict(self, i):
        self.__end_state[i] = True
        self.__conflict_state[i] = True


    def set_conflict(self, i, j):#对 agent i 用agent j 设为发生冲突而停止的 Agent
        self.__avs[self.__current[i][0]][self.__current[i][1]] = 0 #这应该是把这个发生冲突的坐标置为0,表示在这个坐标上没有Agent了（如果有Agent 则avs[x][y]=Agent的编号）
        self.__end_state[i] = True #把 i 和 j 设置为已经停止
        self.__end_state[j] = True
        self.__conflict_state[i] = True #这一行暂时不动，为啥要设置两个 一个 self.__end_state 一个 conflict_state？
        self.__conflict_state[j] = True


    def check_i_conflict(self, i):#检查是否有与Agent i 冲突的Agent

        if  self.__conflict_state[i]: #如果Agent i 已经被标记为停止了，就直接返回True
            return True
        if ( self.__current[i][0] == self.__target_list[i][0]) and (self.__current[i][1] == self.__target_list[i][1]) : #如果Agent i的当前状态已经到达目的地，那么就不会冲突了
            return  False
        if ( self.__current[i][0] < 0 ) or ( self.__current[i][1] < 0 ): #如果Agent i 当前的坐标为负数，那么代表就不会冲突
            return False
        for k in range(self.__agent_num): #遍历所有的 Agent
            if k == i : #自己不会与自己冲突
                continue
            if (self.__current[k][0] == self.__current[i][0] ) and ( self.__current[k][1] == self.__current[i][1]) : #如果两个 Agent 的坐标相等
                self.set_conflict( i, k ) #那么就把 i 和 j 两个Agent设置为冲突
                return True
        return False

    def check_all_conflict(self):
        for i in range(self.__agent_num):
            self.check_i_conflict(i)

    def check_conflict(self, agt, pos, actual ):#这里又重载了一个函数，在pos[0],pos[1]是否有冲突，这个坐标处这里的actual是判断是不是真正移动，是virtual_move or actual move
        k = 0
         #我猜测应该是 Agt这个agent在pos[0]pos[1]这个位置，检查有么有其他的Agent也在这个位置
        for k in range(self.__agent_num):
            if k == agt :
                continue
            if ( self.__current[k][0] == pos[0] ) and ( self.__current[k][1] == pos[1] ) :
                if actual : #如果是真的移动那就意味着两个agent发生撞击了
                    self.set_conflict( agt, k )
                    return True
        return False

    def get_i_wanna(self):
        k = copy.deepcopy(self.__Pheromone_map)
        return(k)

    def refreshMaze(self, agt):
        '''#更新迷宫'''

        k = w = 0
        x = y = 0
        d = 0.0

         # limit the agent number between 1 and 10
        if (agt < 1):
             self.__agent_num = 1
        elif( agt > 100 ):
             self.__agent_num = 100
        else :
             self.__agent_num = agt
        self.buf = []
        self.__current = [([0] * 2) for i in range(self.__agent_num)]  #存储当前Agent的坐标
        self.__Attractor_list = [([0] * 2) for i in range(self.__agent_num)] #存储当前Agent所选择的吸引子的坐标
        self.__target = [([0] * 2) for i in range(self.__agent_num)] #存储当前target的坐标
        self.__target_list = [([0] * 2) for i in range(self.__agent_num)] #存储当前每个智能体要前往的target的目标
        self.__prev_current = [([0] * 2) for i in range(self.__agent_num)] #存储AGent上一个位置的坐标
        self.__currentBearing = [0] * self.__agent_num #所有Agent的当前方向
        self.__AttractorBearing = [0] * self.__agent_num #存储当前AGent所选择吸引子的方位
        self.__prev_bearing = [0] * self.__agent_num #所有Agnet之前的方向
        self.__targetBearing = [0] * self.__agent_num #所有Agent的目标方向
        self.__avs = [([0] * self.size) for i in range(self.size)] #地图1 用于存储每个智能体的位置
        self.__Pheromone_map = [([0] * self.size) for i in range(self.size)] #地图2 用于存储信息素
        self.__target_map = [([0] * self.size) for i in range(self.size)] #地图3 用于存储target
        self.__mines = [([0] * self.size) for i in range(self.size)] #地图4 用于存储雷
        self.__end_state =  [False] * self.__agent_num #判断Agent是否已经停止了
        self.__conflict_state = [False] *  self.__agent_num #判断Agent是否发生了冲突
        self.__sonar =[([0] * 8) for i in range(self.__agent_num)] #声纳信号
        self.__av_sonar = [([0] * 8) for i in range(self.__agent_num)] #av声纳信号
        self.rho = 0.8


        self.set_target() #设置target的坐标

        for k in range(self.__agent_num): #给每个Agent都随机生成一个初始位置
            while True:
                x = random.randint(0, self.size - 1)
                self.__current[k][0] = x
                y = random.randint(0, self.size - 1)
                self.__current[k][1] = y
                if(self.__avs[x][y] == 0 and self.__target_map[x][y] == 0):
                    self.__avs[x][y] = k + 1 #在地图上标出来Agent的位置
                    break

            for w in range(2):
                self.__prev_current[k][w] = self.__current[k][w] #之前的位置状态也标记成这个，反正是初始化无所谓的

            self.__end_state[k] = False
            self.__conflict_state[k] = False

        for i in range(self.numMines):  #初始化雷的位置
            while True:
                x = random.randint(0, self.size - 1)
                y = random.randint(0, self.size - 1)

                if ( self.__avs[x][y] == 0 ) and ( self.__mines[x][y] == 0 ) and ( self.__target_map[x][y] == 0) :
                    self.__mines[x][y] = 1
                    break

        for a in range(self.__agent_num): #每个Agent的初始方向随机
            self.setCurrentBearing(a, random.randint(0, 7))
            # self.setCurrentBearing( a, self.adjustBearing( self.getTargetBearing( a ) ) )
            self.__prev_bearing[a] = self.__currentBearing[a]

        self.Choose_all_target() #按照最近原则为每个智能体选择目标位置，


    def set_target(self):
        '''
        设置target的坐标
        '''
        while True:
            x = random.randint(2, self.size - 3)
            self.__target[0][0] = x
            y = random.randint(2, self.size - 3)
            self.__target[0][1] = y
            if(self.__avs[x][y] == 0):
                self.__target_map[x][y] = 1
                break
        dx,dy = [-1,1,0,0],[0,0,-1,1]
        for i in range(1, self.__agent_num):
            x, y = self.__target[i - 1][0], self.__target[i - 1][1]
            while True:
                t = random.randint(0,3)
                nx, ny = x + dx[t], y + dy[t]
                if(nx < 0 or nx >= self.size or ny < 0 or ny >= self.size):
                    continue
                if self.__avs[nx][ny] == 0 and self.__target_map[nx][ny] == 0:
                    self.__target[i][0], self.__target[i][1] = nx, ny
                    self.__target_map[nx][ny] = 1
                    break

    def adjustBearing(self, old_bearing ):
        '''
        用于调整坦克的方向
        参数：原始方向
        返回值：调整后的方向
        '''
        if( ( old_bearing == 1 ) or ( old_bearing == 7 ) ):
            return 0  #右上左上都归为上
        if( ( old_bearing == 3 ) or ( old_bearing == 5 ) ):
            return 4  #右下左下都归为下
        return old_bearing

    def get_a_b_Bearing(self, a : list, b : list): #获取a在b的方位
        if(a[0] < 0) or (b[0] < 0):
            return 0
        d = [0] * 2
        d[0] = a[0] - b[0]
        d[1] = a[1] - b[1]

        if( d[0] == 0 and d[1] < 0 ): #我是以左上角为坐标系向下向右建立坐标轴，横轴为x，竖轴为y
            return( 0 ) #向上
        if( d[0] > 0 and d[1] < 0 ):
            return( 1 ) #右上
        if( d[0] > 0 and d[1] == 0 ):
            return( 2 ) #右
        if( d[0] > 0 and d[1] > 0 ):
            return( 3 ) #右下
        if( d[0] == 0 and d[1] > 0 ):
            return( 4 ) #下
        if( d[0] < 0 and d[1] > 0 ):
            return( 5 ) #左下
        if( d[0] < 0 and d[1] == 0 ):
            return( 6 ) #左
        if( d[0] < 0 and d[1] < 0 ):
            return( 7 ) #左上
        return( 0 )


    def getTargetBearing(self, i):    #获得目标方位

        if ( self.__current[i][0] < 0 ) or ( self.__current[i][1] < 0 ) :
            return 0
        #  [] d = new [ self.__agent_num]
        d = [0] * 2
        d[0] = self.__target_list[i][0] - self.__current[i][0]
        d[1] = self.__target_list[i][1] - self.__current[i][1]

        if( d[0] == 0 and d[1] < 0 ): #我是以左上角为坐标系向下向右建立坐标轴，横轴为x，竖轴为y
            return( 0 ) #向上
        if( d[0] > 0 and d[1] < 0 ):
            return( 1 ) #右上
        if( d[0] > 0 and d[1] == 0 ):
            return( 2 ) #右
        if( d[0] > 0 and d[1] > 0 ):
            return( 3 ) #右下
        if( d[0] == 0 and d[1] > 0 ):
            return( 4 ) #下
        if( d[0] < 0 and d[1] > 0 ):
            return( 5 ) #左下
        if( d[0] < 0 and d[1] == 0 ):
            return( 6 ) #左
        if( d[0] < 0 and d[1] < 0 ):
            return( 7 ) #左上
        return( 0 )

    def get_all_TargetBearing(self):#又重载了一个获取目标方位的函数，这个是直接返回所有Agent的目标方位

        ret = [0] * self.__agent_num
        k = 0

        for k in range(self.__agent_num):
            ret[k] = self.getTargetBearing( k )
        return ret

    def getCurrentBearing(self, i ): #获取 Agent i的当前方位
        return  self.__currentBearing[i]

    def get_all_CurrentBearing(self):  #获取所有Agent的当前方位
        return ( self.__currentBearing )

    def setCurrentBearing(self, i, b ):#把 Agent i 的方位设成 b
        self.__currentBearing[i] = b

    def set_all_CurrentBearing(self, b): #把所有Agent的方位设为b
        self.__currentBearing = b


    def Choose_Pheromone(self, agt : int):
        """选择一个信息素作为吸引子并返回其位于agt的相对方位，如果没有就返回-1
        选择规则为计算探测区域(以agt为中心的正方形)内权重最大的信息素作为吸引子
        """
        x, y = -1, -1 #记录权重最大的信息素的坐标和权值
        Max_num = -1
        x0, y0 = self.__current[agt][0], self.__current[agt][1]
        x1, x2, y1, y2 = max(x0-self.Sound, 0), min(x0+self.Sound,self.size-1), max(y0-self.Sound,0), min(y0+self.Sound,self.size-1)

        # 四个点为正方形的四个顶点
        for j in range(y1, y2 + 1):#提取探测区域内所有的吸引子
            for i in range(x1, x2 + 1):
                if(i == x0) and (j == y0):
                    continue
                if(self.__Pheromone_map[i][j] > 0):
                    cur = gaussian(self.get_a_b_Range([i,j],[x0,y0])) * self.__Pheromone_map[i][j]
                    if cur > Max_num :
                        Max_num = cur
                        x, y =  i, j
                    elif cur == Max_num:
                        x , y = random.choice([[x,y],[i,j]])
        if Max_num == -1 and x == -1 and y == -1:
            self.__Attractor_list[agt][0], self.__Attractor_list[agt][1] = -1, -1
            self.__AttractorBearing[agt] = self.__currentBearing[agt]
            return 0,self.__AttractorBearing[agt]

        else:
            self.__Attractor_list[agt][0], self.__Attractor_list[agt][1] = x, y
            self.__AttractorBearing[agt] = self.get_a_b_Bearing([x, y], [x0, y0])
            return self.get_a_b_Range(self.__Attractor_list[agt],self.__current[agt]),self.__AttractorBearing[agt]

    def Choose_all_Pheromone(self): #给所有的Agent选择吸引子
        for agt in range(self.__agent_num):
            self.Choose_Pheromone(agt)

    def Choose_target(self, agt : int): #按照最近原则，给Agent选择target
        dist_agt = []
        MAX_INT = sys.maxsize
        for target in self.__target:
            if(self.__target_map[target[0]][target[1]]):
                dist_agt.append((target[0] - self.__current[agt][0]) ** 2 + (target[1] - self.__current[agt][1]) ** 2)
            else:
                dist_agt.append(MAX_INT)

        ret = dist_agt.index(min(dist_agt)) #距离最近的那个target
        self.__target_list[agt][0],self.__target_list[agt][1] = self.__target[ret][0],self.__target[ret][1]
        # print(str(agt) + " : " + str(ret))
        self.__target_map[self.__target[ret][0]][self.__target[ret][1]] = 0
        return self.__target_list[agt][0],self.__target_list[agt][1]

    def Choose_all_target(self): #给所有的Agent选择吸target
        for agt in range(self.__agent_num):
            self.Choose_target(agt)

    def Leave_Pheromone(self, agt : int):
        '''当且仅当Agent到达终点才释放信息素'''
        x0, y0 = self.__current[agt][0], self.__current[agt][1]
        if(x0 == self.__target_list[agt][0] and y0 == self.__target_list[agt][1]):
            if(self.__Pheromone_map[x0][y0] <= self.Sound):
                self.__Pheromone_map[x0][y0] += 1


    def Decay_Pheromone(self):
        '''
        信息素衰减函数，小于0.01就不记录了
        '''
        if(self.rho > self.rho_min):
            self.rho = round((self.rho * 0.95),3)
        for i in range(self.size):
            for j in range(self.size):
                if(self.__Pheromone_map[i][j]):
                    self.__Pheromone_map[i][j] = round((self.__Pheromone_map[i][j] * self.rho),3)
                    if(self.__Pheromone_map[i][j] < 0.01): self.__Pheromone_map[i][j] = 0

    def Diffuse_Pheromone(self):
        '''
        信息素扩散函数
        '''
        dx = [0, 0, -1, 1]
        dy = [-1, 1, 0, 0]
        for i in range(self.size):
            for j in range(self.size):
                if(self.__Pheromone_map[i][j] and self.__mines[i][j] == 0):
                    for k in range(4):
                        x , y = i + dx[k], j + dy[k]
                        if(x >= 0 and x < self.size and y >= 0 and y < self.size and self.__mines[x][y] == 0):
                            self.__Pheromone_map[x][y] =  round((self.__Pheromone_map[x][y] +(self.__Pheromone_map[i][j] * self.Diffuse_rate)),3)

    def getReward(self, agt : int, pos : list):
        #这个函数是获得agt当前的奖励值
        x = pos[0]
        y = pos[1]
        x0 = self.__prev_current[agt][0]
        y0 = self.__prev_current[agt][1]
        next_state = self.getState(agt)
        # 1.抵达奖励
        if ( x == self.__target_list[agt][0] ) and ( y == self.__target_list[agt][1]) : # reach self.__target
            self.__end_state[agt] = True
            return 10.0, next_state #获得奖励10
        if ( x < 0 ) or ( y < 0 )  :# out of field
            return -10.0, next_state

        # 2.撞雷惩罚
        if self.__mines[x][y] == 1 :      # hit self.__mines 碰到雷返回0
            return  -10.0, next_state


        # 3.碰撞惩罚
        if self.isConflict(agt) or self.check_i_conflict(agt) :
            return  -10.0, next_state
        # 4.停止惩罚

        if(self.__prev_current[agt][0] == self.__current[agt][0]) and (self.__prev_current[agt][1] == self.__current[agt][1]):
            self.stop_reward += -0.01
        else:
            self.stop_reward = 0.0

        turn_reward = 0.0
        # 5.拐弯惩罚
        if(self.__currentBearing[agt] != self.__prev_bearing[agt]):
            turn_reward = -0.01

        # 6. 预防式的靠近惩罚
        # close_reward = -max(next_state[:8]) * 0.1
        close_reward = -0.1 * max(next_state[:8]) #( 0 -> -1)
        # print(close_reward)
        # 6.靠近或远离惩罚(应用Reward Shaping方法)
        now_Range = - 0.01 * self.getRange(agt) #(-0.1 -> -1.6)

        # 7.靠近或远离吸引子的奖励与惩罚

        attractor_reaward = 0.0#(0 -> 1)
        if(self.__Attractor_list[agt][0] == -1 and self.__Attractor_list[agt][1] == -1):
            attractor_reaward = 0.0
        else:
            attractor_reaward = 1 / ((self.get_a_b_Range([x,y],self.__Attractor_list[agt])) + 1) 

        # print("now_Range : " + str(now_Range))
        # print("turn_reward : " + str(turn_reward))
        # print("close_reward : " + str(close_reward))
        # print("attractor_reaward : " + str(attractor_reaward))
        # print("stop_reward : " + str(self.stop_reward))
        # print()
        return (now_Range + turn_reward + close_reward + attractor_reaward + self.stop_reward), next_state


    def get_i_Reward(self,i : int):#获取agent i 的奖励
        return( self.getReward( i, self.__current[i]) )


    def get_a_b_Range(self, a : list, b : list ): #两个点 a和 b 返回两者x和y坐标的曼哈顿距离
        Range = 0
        d = [0] * 2

        d[0] = abs( a[0] - b[0] )
        d[1] = abs( a[1] - b[1] )
        Range = d[0] + d[1]
        return( Range )

    def getRange(self,i):  #返回 I 和 self.__target 坐标的曼哈顿距离
       return( self.get_a_b_Range( self.__current[i], self.__target_list[i] ) )

    def get_i_j_Range(self,i,j):#返回 I 和 j 的曼哈顿距离
        return(self.get_a_b_Range( self.__current[i], self.__current[j] ) )

    def get_all_Range(self):#返回所有agent与target的的曼哈顿距离
        k = 0
        all_range = [0] * self.__agent_num
        for k in range(self.__agent_num):
            all_range[k] = self.getRange( k )
        return( all_range )


    def getSonar(self, agt : int, new_sonar : list ): #获取声纳信号
        r = 0
        x = self.__current[agt][0]
        y = self.__current[agt][1]
        if ( x < 0 ) or ( y < 0 ) :
            for k in range(8):
                new_sonar[k] = 0
            return

        aSonar =  [0.0] * 8 #八个方位输入

        r = 0 #r就是当前位置(x,y)距离墙或者雷的距离
        while (y - r >= 0) and (self.__mines[x][y-r] != 1 ) and (r <= self.Sound):    #从(x,y)位置向上摸索，看看有没有雷或者墙
            r = r + 1
        if r == 0 or r > self.Sound: # or y-r<0) #也就是说在（x,y）上有颗雷，这时候显然就不能有输入了
            aSonar[0] = 0.0
        else : #
            aSonar[0] = 1.0 / r

        r = 0
        while (x + r <= self.size - 1) and (y - r >= 0) and (self.__mines[x+r][y-r] != 1) and (r <= self.Sound) :
            r = r + 1
        if r == 0 or r > self.Sound:
            aSonar[1] = 0.0
        else :
            aSonar[1] = 1.0 / r

        r = 0
        while (x + r <= self.size - 1 and self.__mines[x+r][y] != 1) and (r <= self.Sound):
            r = r + 1
        if r == 0 or r > self.Sound:
            aSonar[2] = 0.0
        else :
            aSonar[2] = 1.0 / r

        r = 0
        while (x + r <= self.size - 1 and y + r <= self.size - 1 and self.__mines[x+r][y+r] != 1) and (r <= self.Sound):
            r = r + 1
        if (r == 0) or (r > self.Sound):
            aSonar[3] = 0.0
        else :
            aSonar[3] = 1.0 / r

        r = 0
        while (y + r <= self.size - 1 and self.__mines[x][y+r] != 1) and (r <= self.Sound):
            r = r + 1
        if (r==0)or (r > self.Sound) :
            aSonar[4] = 0.0
        else :
            aSonar[4] = (1.0 / r)

        r=0
        while (x-r>=0 and y+r<=self.size-1 and self.__mines[x-r][y+r]!=1) and (r <= self.Sound):
            r = r + 1
        if (r==0) or (r > self.Sound) :
            aSonar[5] = 0.0
        else :
            aSonar[5] = 1.0 / r

        r=0
        while (x-r>=0 and self.__mines[x-r][y]!=1) and (r <= self.Sound):
            r = r + 1
        if (r==0) or (r > self.Sound) :
            aSonar[6] = 0.0
        else :
            aSonar[6] = 1.0 / r

        r=0
        while (x-r>=0 and y-r>=0 and self.__mines[x-r][y-r]!=1) and (r <= self.Sound):
            r = r + 1
        if (r==0)or (r > self.Sound) :
            aSonar[7] = 0.0
        else :
            aSonar[7] = 1.0 / r

        # self.__currentBearing = self.get_all_CurrentBearing ()

        for k in range(8):
            new_sonar[k] = aSonar[k] #这也太绕了我靠，new_sonar的方位是顺时针，从左方向开始计数，左方向为0 右方向为4，aSonar的方位是从上方向开始的 左侧为6 右侧为2
            if (self.binarySonar):#上面那式子就是做一个转换，把all_Sonar的八个方向的五个方向取过来放到new_sonar中
                if (new_sonar[k] < 1):
                    new_sonar[k] = 0  # binary self.__sonar signal
        return

    def getAVSonar(self, agt : int, new_av_sonar : list): ##获取AV声纳信号

        r = 0
        x = self.__current[agt][0]
        y = self.__current[agt][1]

        if( ( x < 0 ) or ( y < 0 ) ):

            for k in range(8):
                new_av_sonar[k] = 0 #初始化当前Agent的五个感知信号的输入
            return

        aSonar = [0] * 8 #初始化八个方向的探测Agent的声纳信号

        r=0
        while( y-r>=0 and (self.__avs[x][y-r]==(agt+1) or self.__avs[x][y-r]==0) and (r <= self.Sound)): #y-r>=0限制有没有到墙边，程序里的Agent的编号是0-7，实际编号为1-8，这里的Agt+1就是指本身，向上探测，看是否有其余的Agent
            r = r + 1
        if (r==0) or (r > self.Sound) :
            aSonar[0] = 0.0 #
        else :
            aSonar[0] = 1.0 / r

        r=0
        while (x+r<=self.size-1 and y-r>=0 and ( self.__avs[x+r][y-r]==(agt+1) or self.__avs[x+r][y-r]==0 ) and (r <= self.Sound)): #右上
            r = r + 1
        if (r==0) or (r > self.Sound):
            aSonar[1] = 0.0
        else :
            aSonar[1] = 1.0 / r

        r=0
        while (x+r<=self.size-1 and ( self.__avs[x+r][y]==(agt+1) or self.__avs[x+r][y]==0 ) and (r <= self.Sound)):#右侧
            r = r + 1
        if (r==0) or (r > self.Sound):
            aSonar[2] = 0.0
        else :
            aSonar[2] = 1.0 / r

        r=0
        while (x+r<=self.size-1 and y+r<=self.size-1 and ( self.__avs[x+r][y+r]==(agt+1) or self.__avs[x+r][y+r]==0 ) and (r <= self.Sound)): #右下
            r = r + 1
        if (r==0) or (r > self.Sound):
            aSonar[3] = 0.0
        else :
            aSonar[3] = 1.0 / r

        r=0
        while (y+r<=self.size-1 and ( self.__avs[x][y+r]==(agt+1) or self.__avs[x][y+r]==0 ) and (r <= self.Sound)):#下
            r = r + 1
        if (r==0) or (r > self.Sound):
            aSonar[4] = 0.0
        else :
            aSonar[4] = 1.0 / r

        r=0
        while (x-r>=0 and y+r<=self.size-1 and ( self.__avs[x-r][y+r]==(agt+1) or self.__avs[x-r][y+r]==0 ) and (r <= self.Sound)):#左下
            r = r + 1
        if (r==0) or (r > self.Sound):
            aSonar[5] = 0.0
        else :
            aSonar[5] = 1.0 / r

        r=0
        while (x-r>=0 and ( self.__avs[x-r][y]==(agt+1) or self.__avs[x-r][y]==0 )and (r <= self.Sound) ):#左
            r = r + 1
        if (r==0) or (r > self.Sound):
            aSonar[6] = 0.0
        else :
            aSonar[6] = 1.0 / r

        r=0
        while (x-r>=0 and y-r>=0 and ( self.__avs[x-r][y-r]==(agt+1) or self.__avs[x-r][y-r]==0 ) and (r <= self.Sound)):#左上
            r = r + 1
        if (r==0) or (r > self.Sound):
            aSonar[7] = 0.0
        else :
            aSonar[7] = 1.0 / r

        for k in range(8):
            new_av_sonar[k] = aSonar[k] #方向转换，这里的+6要谨慎对待，可以替换为-2，不过不同的编译器对于负数求余的认知不同，还是尽量使用正数求余
            if( self.binarySonar ): #二值化输入的声纳信号，只有0和1
                if( new_av_sonar[k] < 1 ):#
                    new_av_sonar[k] = 0  # binary self.__sonar signal
        return

    def get_all_Sonar(self, agt : int, new_sonar : list): #把两种声纳信号合二为一放入8维探测器中
        r = 0
        x = self.__current[agt][0]
        y = self.__current[agt][1]
        if ( x < 0 ) or ( y < 0 ) or self.is_end(agt):
            for k in range(8):
                new_sonar[k] = 0
            return

        aSonar =  [0] * 8 #八个方位输入

        for r in range(self.Sound + 1):
            nx, ny = x, y - r
            if (nx < 0) or (nx >= self.size) or (ny < 0) or (ny >= self.size):
                aSonar[0] = 1 / r if(r !=0) else 1
                break
            if(self.__mines[nx][ny]):
                aSonar[0] = 1 / r if(r !=0) else 1
                break
            elif(self.__avs[nx][ny] != 0 and self.__avs[nx][ny] != agt + 1):
                aSonar[0] = 1 / r if(r !=0) else 1
                break

        for r in range(self.Sound + 1):
            nx, ny = x + r, y - r
            if (nx < 0) or (nx >= self.size) or (ny < 0) or (ny >= self.size):
                aSonar[1] = 1 / r if(r !=0) else 1
                break
            if(self.__mines[nx][ny]):
                aSonar[1] = 1 / r if(r !=0) else 1
                break
            elif(self.__avs[nx][ny] != 0 and self.__avs[nx][ny] != agt + 1):
                aSonar[1] = 1 / r if(r !=0) else 1
                break

        for r in range(self.Sound + 1):
            nx, ny = x + r, y
            if (nx < 0) or (nx >= self.size) or (ny < 0) or (ny >= self.size):
                aSonar[2] = 1 / r if(r !=0) else 1
                break
            if(self.__mines[nx][ny]):
                aSonar[2] = 1 / r if(r !=0) else 1
                break
            elif(self.__avs[nx][ny] != 0 and self.__avs[nx][ny] != agt + 1):
                aSonar[2] = 1 / r if(r !=0) else 1
                break

        for r in range(self.Sound + 1):
            nx, ny = x + r, y + r
            if (nx < 0) or (nx >= self.size) or (ny < 0) or (ny >= self.size):
                aSonar[3] = 1 / r if(r !=0) else 1
                break
            if(self.__mines[nx][ny]):
                aSonar[3] = 1 / r if(r !=0) else 1
                break
            elif(self.__avs[nx][ny] != 0 and self.__avs[nx][ny] != agt + 1):
                aSonar[3] = 1 / r if(r !=0) else 1
                break

        for r in range(self.Sound + 1):
            nx, ny = x, y + r
            if (nx < 0) or (nx >= self.size) or (ny < 0) or (ny >= self.size):
                aSonar[4] = 1 / r if(r !=0) else 1
                break
            if(self.__mines[nx][ny]):
                aSonar[4] = 1 / r if(r !=0) else 1
                break
            elif(self.__avs[nx][ny] != 0 and self.__avs[nx][ny] != agt + 1):
                aSonar[4] = 1 / r if(r !=0) else 1
                break

        for r in range(self.Sound + 1):
            nx, ny = x - r, y + r
            if (nx < 0) or (nx >= self.size) or (ny < 0) or (ny >= self.size):
                aSonar[5] = 1 / r if(r !=0) else 1
                break
            if(self.__mines[nx][ny]):
                aSonar[5] = 1 / r if(r !=0) else 1
                break
            elif(self.__avs[nx][ny] != 0 and self.__avs[nx][ny] != agt + 1):
                aSonar[5] = 1 / r if(r !=0) else 1
                break

        for r in range(self.Sound + 1):
            nx, ny = x - r, y
            if (nx < 0) or (nx >= self.size) or (ny < 0) or (ny >= self.size):
                aSonar[6] = 1 / r if(r !=0) else 1
                break
            if(self.__mines[nx][ny]):
                aSonar[6] = 1 / r if(r !=0) else 1
                break
            elif(self.__avs[nx][ny] != 0 and self.__avs[nx][ny] != agt + 1):
                aSonar[6] = 1 / r if(r !=0) else 1
                break

        for r in range(self.Sound + 1):
            nx, ny = x - r, y - r
            if (nx < 0) or (nx >= self.size) or (ny < 0) or (ny >= self.size):
                aSonar[7] = 1 / r if(r !=0) else 1
                break
            if(self.__mines[nx][ny]):
                aSonar[7] = 1 / r if(r !=0) else 1
                break
            elif(self.__avs[nx][ny] != 0 and self.__avs[nx][ny] != agt + 1):
                aSonar[7] = 1 / r if(r !=0) else 1
                break
        # print("________________分割线_____________")
        # print(aSonar)
        # print("________________分割线_____________")
        for k in range(8):
            new_sonar[k] = aSonar[k] #这也太绕了我靠，new_sonar的方位是顺时针，从左方向开始计数，左方向为0 右方向为4，aSonar的方位是从上方向开始的 左侧为6 右侧为2
        return

     #这个virtual_move函数用于虚拟执行下一步的行走以计算奖励，不实际改变方向和坐标，而把虚拟行走后的坐标存入res[0]andres[1]中
    def virtual_move(self, agt : int, d : int):#Agent的虚拟行走函数，一次行走1，agt为Agent的编号，d为要前进的方向（d为相对方向，d的取值应为-2 -1 0 1 2），res为Agt对于Agent的坐标

        k = 0
        bearing = d #计算按d行走后的绝对方向
        res = [0] * 2
        res[0] = self.__current[agt][0]
        res[1] = self.__current[agt][1]


        if bearing == 0:
            if( res[1] > 0 ):
                res[1] -= 1

        elif bearing == 1:
            if( ( res[0] < self.size - 1 ) and ( res[1] > 0 ) ):
                res[0] += 1
                res[1] -= 1

        elif bearing == 2:
            if( res[0] < self.size - 1 ):
                res[0] += 1

        elif bearing == 3:
            if( ( res[0] < self.size - 1 ) and ( res[1] < self.size - 1 ) ):

                res[0] += 1
                res[1] += 1
        elif bearing == 4:
            if( res[1] < self.size - 1 ):
                res[1] += 1

        elif bearing == 5:
            if( ( res[0] > 0 ) and ( res[1] < self.size - 1 ) ):
                res[0] -= 1
                res[1] += 1

        elif bearing == 6:
            if( res[0] > 0 ):
                res[0] -= 1

        elif bearing == 7:
            if( ( res[0] > 0 ) and ( res[1] > 0 ) ):
                res[0] -= 1
                res[1] -= 1
        else:
            pass
        #备注！训练好模型后在doStep展示阶段可以开启下列语句，用于防止智能体间碰撞和雷区碰撞，注意在训练阶段不要开启下列语句，否则会大幅增加训练时间
        # if(self.__avs[res[0]][res[1]] == 0 and self.__mines[res[0]][res[1]] == 0):
        #     return True
        # else:
        #     return False
        #备注！训练好模型后在doStep展示阶段可以开启下列语句，用于防止智能体间碰撞，注意在训练阶段不要开启下列语句，否则会大幅增加训练时间
        # if(self.__avs[res[0]][res[1]] == 0):
        #     return True
        # else:
        #     return False

        return True

    def getState(self, agt : int): #获取智能体的状态输入

        this_Sonar = [0] * 8
        that_Sonar = [0] * 8
        this_AVSonar = [0] * 8
        that_AVSonar = [0] * 8

        this_bearing = 0
        this_targetRange = 0.0

        # self.getSonar( agt, that_Sonar )   #获得mines和边界的声纳信息
        # self.getAVSonar( agt, that_AVSonar )   #autonomous vehicle (AV) 获得其他智能体和边界的声纳信息
        self.get_all_Sonar(agt, that_Sonar)
        for i in range(8):
            this_Sonar[i] = that_Sonar[i]

        this_bearing = self.getTargetBearing( agt ) #获得target方位
        this_targetRange = self.getRange( agt ) #获得target距离
        #可以设置为相对坐标也可以设置为极坐标
        # relative_position = [this_targetRange, this_bearing]
        relative_position = [self.__target_list[agt][0] - self.__current[agt][0], self.__target_list[agt][1] - self.__current[agt][1]]

        attractor_range, attractor_bearing = self.Choose_Pheromone(agt)
        #可以设置为相对坐标也可以设置为极坐标
        pho_position = [self.__Attractor_list[agt][0] - self.__current[agt][0], self.__Attractor_list[agt][1] - self.__current[agt][1]]
        # pho_position = [attractor_range, attractor_bearing ]
    
        return  np.hstack((this_Sonar, relative_position ,pho_position, [agt],[0,0])) #最后两个[0,0]为编队移动任务中 距离领航的那个agent的距离和方位（相对坐标）



    def get_all_State(self):
        state = []
        for agt in range(self.__agent_num):
            state = state + (self.getState(agt).tolist())
        return state

    def turn(self, i, d ):#转向
        bearing = self.getCurrentBearing( i )
        bearing = ( bearing + d ) % 8
        self.setCurrentBearing( i, bearing )

    def move(self, i, d ):#这是Agent实际的移动函数,d为方向，移动成功就返回1，移动不成功就返回-1 
        k = 0
        if( ( self.__current[i][0] < 0 ) or ( self.__current[i][1] < 0 ) ):
            return( -1 )

        for k in range(2):
            self.__prev_current[i][k] = self.__current[i][k]

        self.__prev_bearing[i] = self.__currentBearing[i]

        self.__currentBearing[i] = d

        if(self.__currentBearing[i]== 0):
            if (self.__current[i][1] > 0): self.__current[i][1] -= 1
            else:
                return( -1 )

        elif(self.__currentBearing[i]== 1):
            if (self.__current[i][0] < self.size - 1 and self.__current[i][1] > 0):
                self.__current[i][0] += 1
                self.__current[i][1] -= 1

            else :
                return( -1 )

        elif(self.__currentBearing[i]==  2):
            if (self.__current[i][0]<self.size-1): self.__current[i][0] += 1
            else :
                return( -1 )


        elif(self.__currentBearing[i]==  3):
            if (self.__current[i][0]<self.size-1 and self.__current[i][1]<self.size-1):
                self.__current[i][0] += 1
                self.__current[i][1] += 1

            else :
                return( -1 )


        elif(self.__currentBearing[i]==  4):
            if (self.__current[i][1]<self.size-1):
                self.__current[i][1] += 1
            else :
                return( -1 )


        elif(self.__currentBearing[i]==  5):
            if (self.__current[i][0]>0 and self.__current[i][1]<self.size-1):
                self.__current[i][0] -= 1
                self.__current[i][1] += 1

            else :
                return( -1 )


        elif(self.__currentBearing[i]==  6):
            if (self.__current[i][0]>0): self.__current[i][0] -= 1
            else :
                return( -1 )


        elif(self.__currentBearing[i]==  7):
            if (self.__current[i][0]>0 and self.__current[i][1]>0):
                self.__current[i][0] -= 1
                self.__current[i][1] -= 1

            else :
                return( -1 )


        else:
            pass
        self.__avs[self.__prev_current[i][0]][self.__prev_current[i][1]] = 0
        self.__avs[self.__current[i][0]][self.__current[i][1]] = i + 1

        return (1)

   # return True if the move still keeps the agent within the field 检查是否超出边界
    def withinField (self, i, d ):  #检测AGent i 还能否在 相对方向d 上继续移动

        testBearing = d
        if testBearing == 0:
            if (self.__current[i][1]>0):
                return (True)

        elif testBearing == 1:
            if (self.__current[i][0]<self.size-1 and self.__current[i][1]>0):
                return( True )

        elif testBearing == 2:
            if (self.__current[i][0]<self.size-1): return (True)

        elif testBearing == 3:
            if (self.__current[i][0]<self.size-1 and self.__current[i][1]<self.size-1):
                return( True )

        elif testBearing == 4:
            if (self.__current[i][1]<self.size-1):
                return( True )

        elif testBearing == 5:
            if (self.__current[i][0]>0 and self.__current[i][1]<self.size-1):
                return (True)

        elif testBearing == 6:
            if (self.__current[i][0]>0):
                return( True )

        elif testBearing == 7:
            if (self.__current[i][0]>0 and self.__current[i][1]>0):
                return( True )

        else:
            pass

        return (False)


    def move_all( self, d ): #一次移动所有的Agent

        k = 0

        res = [0] * self.__agent_num
        for k in range(self.__agent_num):
            res[k] = self.move( k, d[k] )
        return res

    def undoMove(self): #取消上次的移动
        self.__currentBearing = self.__prev_bearing
        self.__current[0] = self.__prev_current[0]
        self.__current[1] = self.__prev_current[1]


    def endState(self, agt ):#修改并返回agt的end_state 用于判断当前agt是否已经停止

        x = self.__current[agt][0]
        y = self.__current[agt][1]

        if( self.__conflict_state[agt] ):#agt已经发生冲突了

            self.__end_state[agt] = True
            return( self.__end_state[agt] )

        if( ( x < 0 ) or ( y < 0 ) ):#出界了

            self.__end_state[agt] = True
            return( self.__end_state[agt] )

        if( ( x == self.__target_list[agt][0] ) and ( y == self.__target_list[agt][1] ) ):#到达终点了

            self.__end_state[agt] = True
            self.__target_map[x][y] = 0
            return( self.__end_state[agt] )

        if( ( self.__mines[x][y] == 1 ) or ( self.check_i_conflict( agt ) ) or ( self.__end_state[agt] ) ):# 踩雷or检测冲突oragt已经停止

            # self.__avs[x][y] = 0
            self.__end_state[agt] = True

        else :
            self.__end_state[agt] = False
        return( self.__end_state[agt] )

    def is_end(self, agt : int):#用于返回智能体是否已经停止运行了
        return self.__end_state[agt]


    def endState_target_moving(self,target_moving ):
        '''
        这个函数用于检测是否所有的Agent都停止工作了，如果是的话返回True
        参数:target_moving没用，输入True就行了
        返回值：全部都停止运动了就返回True
        '''

        bl = True #这一参数是用来当target是移动的时候，返回Agent k 当前是否已经进入endState
        for k in range(self.__agent_num):

            if( target_moving ):
                if( self.isHitTarget( k ) ): #这里偷懒了。。。只考虑了单个智能体的情况
                    return( True )
                if( not self.endState( k ) ):#如果Agent k 还没有停止
                    bl = False

            else :
                if( not self.endState( k ) ):
                    return( False )

        if( target_moving ):
            return( bl )
        else :
            return( True )


    def isHitMine(self, i ):#判断是否踩雷，踩雷就返回True

        if( ( self.__current[i][0] < 0 ) or ( self.__current[i][1] < 0 ) ):
            return( False )
        if( self.__mines[self.__current[i][0]][self.__current[i][1]] == 1 ):
            return True
        else :
            return False

    def isConflict(self, i): #判断i是否发生冲突

        return( self.__conflict_state[i] )

    def isHitTarget(self, i): #在target_move模式下判断是否已经到达目标

        if( ( self.__current[i][0] == self.__target_list[i][0] ) and ( self.__current[i][1] == self.__target_list[i][1] ) ):
            return True
        else :
            return False

    def test_mines(self, i, j ):#判断坐标(i,j)是否有雷

        if( self.__mines[i][j] == 1 ):
            return( True )
        else :
            return( False )

    def Bearing_gap(self, a : int, b : int): #获取 a 与 b的方向差

        ret = min((self.__currentBearing[a] - self.__currentBearing[b]) % 8, (self.__currentBearing[b] - self.__currentBearing[a]) % 8)
        return ret

    def test_current( self, agt, i, j ):#判断agt当前的坐标是不是(i,j)
        if ( self.__current[agt][0] == i and self.__current[agt][1] == j ):
            return( True )
        else :
            return( False )

    def test_target(self,i,j ): #判断当前target的坐标是不是（i，j）

        if( ( self.__target_list[0] == i ) and ( self.__target_list[1] == j ) ):
            return( True )
        else :
            return( False )

    def getMines(self,i,  j ):#获得当前坐标是否有雷

        return( self.__mines[i][j] )

    def getCurrent(self, agt ): #获取当前agt的坐标

        return( self.__current[agt] )

    def get_all_Current(self):#获取当前所有agt的坐标

        return( self.__current )

    def getCurrent_to_path(self, agt : int, path : list): #把agt的坐标存入path[]

        for k in range(2):
            path[k] = self.__current[agt][k]
        return

    def get_all_Current_to_path(self,path : list):#把所有agt的坐标存入path[][]


        for i in range(self.__agent_num):
            for j in range(2):
                path[i][j] = self.__current[i][j]
        return

    def getPrevCurrent(self, agt):

        return( self.__prev_current[agt] )


    def get_all_PrevCurrent(self):

        return( self.__prev_current )


    def getTarget(self):

        return( self.__target )


# —————————————————————————下面都是编队移动任务的函数—————————————————————————————————————————————————————————————————————

    def set_target_patrol1(self): #直线型坦克
        while True:
            x = random.randint(self.__agent_num // 2 + 1, self.size - self.__agent_num // 2 - 1)
            self.__target[0][0] = x
            y = random.randint(0, self.size - 1)
            self.__target[0][1] = y
            if(self.__avs[x][y] == 0):
                self.__target_map[x][y] = 1
                self.__Pheromone_map[x][y] = 0.5 #更改
                break

        for agt in range(1, self.__agent_num):
            gap = -(agt + 1) // 2 if (agt & 1) else agt // 2
            self.__target[agt][0], self.__target[agt][1] = self.__target[0][0] + gap, self.__target[0][1]
            self.__target_map[self.__target[agt][0]][self.__target[agt][1]] = 1
            self.__Pheromone_map[self.__target[agt][0]][self.__target[agt][1]] = 0.5
            # self.__target[agt + 1][0], self.__target[agt + 1][1] = self.__target[0][0] + gap, self.__target[0][1]
            # self.__target_map[self.__target[agt + 1][0]][self.__target[agt + 1][1]] = 1
            # self.__Pheromone_map[self.__target[agt + 1][0]][self.__target[agt + 1][1]] = 1



    def set_tank_patrol2(self): #方格型坦克
        try:
            if(self.__agent_num != 8):
                raise Exception("智能体的数量只能为8")
        except Exception as e:
            print(e)
        self.__current[0][0],self.__current[0][1] = self.size // 2, self.size - 3
        self.__avs[self.__current[0][0]][self.__current[0][1]] = 1
        
        self.__current[1][0], self.__current[1][1] = self.__current[0][0] + 1, self.__current[0][1]
        self.__avs[self.__current[1][0]][self.__current[1][1]] = 2

        self.__current[2][0], self.__current[2][1] = self.__current[0][0] + 1, self.__current[0][1] + 1
        self.__avs[self.__current[2][0]][self.__current[2][1]] = 3

        self.__current[3][0], self.__current[3][1] = self.__current[0][0] + 1, self.__current[0][1] + 2
        self.__avs[self.__current[3][0]][self.__current[3][1]] = 4

        self.__current[4][0], self.__current[4][1] = self.__current[0][0] , self.__current[0][1] + 2
        self.__avs[self.__current[4][0]][self.__current[4][1]] = 5

        self.__current[5][0], self.__current[5][1] = self.__current[0][0] - 1, self.__current[0][1] + 2
        self.__avs[self.__current[5][0]][self.__current[5][1]] = 6

        self.__current[6][0], self.__current[6][1] = self.__current[0][0] - 1, self.__current[0][1] + 1
        self.__avs[self.__current[6][0]][self.__current[6][1]] = 7

        self.__current[7][0], self.__current[7][1] = self.__current[0][0] - 1, self.__current[0][1]
        self.__avs[self.__current[7][0]][self.__current[7][1]] = 8

    def set_tank_patrol(self): #箭头型坦克
        try:
            if(self.__agent_num != 8):
                raise Exception("智能体的数量只能为8")
        except Exception as e:
            print(e)
        self.__current[0][0],self.__current[0][1] = self.size // 2, self.size - 5
        self.__avs[self.__current[0][0]][self.__current[0][1]] = 1
        
        self.__current[1][0], self.__current[1][1] = self.__current[0][0] + 1, self.__current[0][1] + 1
        self.__avs[self.__current[1][0]][self.__current[1][1]] = 2

        self.__current[2][0], self.__current[2][1] = self.__current[0][0] + 2, self.__current[0][1] + 2
        self.__avs[self.__current[2][0]][self.__current[2][1]] = 3

        self.__current[3][0], self.__current[3][1] = self.__current[0][0] - 1, self.__current[0][1] +1
        self.__avs[self.__current[3][0]][self.__current[3][1]] = 4

        self.__current[4][0], self.__current[4][1] = self.__current[0][0] - 2, self.__current[0][1] + 2
        self.__avs[self.__current[4][0]][self.__current[4][1]] = 5

        self.__current[5][0], self.__current[5][1] = self.__current[0][0] , self.__current[0][1] + 2
        self.__avs[self.__current[5][0]][self.__current[5][1]] = 6

        self.__current[6][0], self.__current[6][1] = self.__current[0][0] , self.__current[0][1] + 3
        self.__avs[self.__current[6][0]][self.__current[6][1]] = 7

        self.__current[7][0], self.__current[7][1] = self.__current[0][0] , self.__current[0][1] + 4
        self.__avs[self.__current[7][0]][self.__current[7][1]] = 8


    def set_target_patrol(self):#箭头型Target
        while True:
            x = random.randint(2, self.size - 3)
            self.__target[0][0] = x
            y = 1
            self.__target[0][1] = y
            if(self.__avs[x][y] == 0):
                self.__target_map[x][y] = 1
                self.__Pheromone_map[x][y] = 0.5 #更改
                break
        
        self.__target[1][0], self.__target[1][1] = self.__target[0][0] + 1, self.__target[0][1] + 1
        self.__target_map[self.__target[1][0]][self.__target[1][1]] = 1
        self.__Pheromone_map[self.__target[1][0]][self.__target[1][1]] = 0.5

        self.__target[2][0], self.__target[2][1] = self.__target[0][0] + 2, self.__target[0][1] + 2
        self.__target_map[self.__target[2][0]][self.__target[2][1]] = 1
        self.__Pheromone_map[self.__target[2][0]][self.__target[2][1]] = 0.5


        self.__target[3][0], self.__target[3][1] = self.__target[0][0] -1, self.__target[0][1] + 1
        self.__target_map[self.__target[3][0]][self.__target[3][1]] = 1
        self.__Pheromone_map[self.__target[3][0]][self.__target[3][1]] = 0.5

        self.__target[4][0], self.__target[4][1] = self.__target[0][0] -2 , self.__target[0][1] + 2
        self.__target_map[self.__target[4][0]][self.__target[4][1]] = 1
        self.__Pheromone_map[self.__target[4][0]][self.__target[4][1]] = 0.5

        self.__target[5][0], self.__target[5][1] = self.__target[0][0] , self.__target[0][1] + 2
        self.__target_map[self.__target[5][0]][self.__target[5][1]] = 1
        self.__Pheromone_map[self.__target[5][0]][self.__target[5][1]] = 0.5

        self.__target[6][0], self.__target[6][1] = self.__target[0][0] , self.__target[0][1] + 3
        self.__target_map[self.__target[6][0]][self.__target[6][1]] = 1
        self.__Pheromone_map[self.__target[6][0]][self.__target[6][1]] = 0.5

        self.__target[7][0], self.__target[7][1] = self.__target[0][0] , self.__target[0][1] + 4
        self.__target_map[self.__target[7][0]][self.__target[7][1]] = 1
        self.__Pheromone_map[self.__target[7][0]][self.__target[7][1]] = 0.5


    def set_target_patrol2(self): #方格型Target
        while True:
            x = random.randint(2, self.size - 2)
            self.__target[0][0] = x
            y = 1
            self.__target[0][1] = y
            if(self.__avs[x][y] == 0):
                self.__target_map[x][y] = 1
                self.__Pheromone_map[x][y] = 0.5 #更改
                break
        
        self.__target[1][0], self.__target[1][1] = self.__target[0][0] + 1, self.__target[0][1]
        self.__target_map[self.__target[1][0]][self.__target[1][1]] = 1
        self.__Pheromone_map[self.__target[1][0]][self.__target[1][1]] = 0.5 #给每个Target的位置放置单位为1的信息素以加速收敛

        self.__target[2][0], self.__target[2][1] = self.__target[0][0] + 1, self.__target[0][1] + 1
        self.__target_map[self.__target[2][0]][self.__target[2][1]] = 1
        self.__Pheromone_map[self.__target[2][0]][self.__target[2][1]] = 0.5


        self.__target[3][0], self.__target[3][1] = self.__target[0][0] + 1, self.__target[0][1] + 2
        self.__target_map[self.__target[3][0]][self.__target[3][1]] = 1
        self.__Pheromone_map[self.__target[3][0]][self.__target[3][1]] = 0.5

        self.__target[4][0], self.__target[4][1] = self.__target[0][0] , self.__target[0][1] + 2
        self.__target_map[self.__target[4][0]][self.__target[4][1]] = 1
        self.__Pheromone_map[self.__target[4][0]][self.__target[4][1]] = 0.5

        self.__target[5][0], self.__target[5][1] = self.__target[0][0] - 1, self.__target[0][1] + 2
        self.__target_map[self.__target[5][0]][self.__target[5][1]] = 1
        self.__Pheromone_map[self.__target[5][0]][self.__target[5][1]] = 0.5

        self.__target[6][0], self.__target[6][1] = self.__target[0][0] - 1, self.__target[0][1] + 1
        self.__target_map[self.__target[6][0]][self.__target[6][1]] = 1
        self.__Pheromone_map[self.__target[6][0]][self.__target[6][1]] = 0.5

        self.__target[7][0], self.__target[7][1] = self.__target[0][0] - 1, self.__target[0][1]
        self.__target_map[self.__target[7][0]][self.__target[7][1]] = 1
        self.__Pheromone_map[self.__target[7][0]][self.__target[7][1]] = 0.5





    def choose_target_patrol(self): #按照编号给每个Agent分配target
        for agt in range(self.__agent_num):
            self.__target_list[agt][0],self.__target_list[agt][1] = self.__target[agt][0],self.__target[agt][1]
            # self.__target_map[self.__target[agt][0]][self.__target[agt][1]] = 0

    def refreshMaze_patrol(self, agt : int):
        '''#更新迷宫'''

        k = w = 0
        x = y = 0
        d = 0.0

         # limit the agent number between 1 and 10
        if (agt < 1):
             self.__agent_num = 1
        elif( agt > 100 ):
             self.__agent_num = 100
        else :
             self.__agent_num = agt
        self.leader = 0 #用于记录领航者，一般默认为0号Agent
        self.__current = [([0] * 2) for i in range(self.__agent_num)]
        self.__Attractor_list = [([0] * 2) for i in range(self.__agent_num)]
        self.__target = [([0] * 2) for i in range(self.__agent_num)]
        self.__target_list = [([0] * 2) for i in range(self.__agent_num)]
        self.__prev_current = [([0] * 2) for i in range(self.__agent_num)]
        self.__currentBearing = [0] * self.__agent_num #所有Agent的当前方向
        self.__AttractorBearing = [0] * self.__agent_num
        self.__prev_bearing = [0] * self.__agent_num #所有Agnet之前的方向
        self.__targetBearing = [0] * self.__agent_num #所有Agent期望的目标方向
        self.__avs = [([0] * self.size) for i in range(self.size)]
        self.__Pheromone_map = [([0] * self.size) for i in range(self.size)]
        self.__target_map = [([0] * self.size) for i in range(self.size)]
        self.__mines = [([0] * self.size) for i in range(self.size)]
        self.__end_state =  [False] * self.__agent_num #判断Agent是否已经停止了
        self.__conflict_state = [False] *  self.__agent_num #判断Agent是否发生了冲突
        self.__sonar =[([0] * 8) for i in range(self.__agent_num)] #先初始化第一维
        self.__av_sonar = [([0] * 8) for i in range(self.__agent_num)]
        self.__outqueue = [0] * self.__agent_num
        self.rho = 1

        for k in range(3):
            d = random.random()   #返回一个随机数，[0.0,1.0]

        self.set_tank_patrol() #设置坦克
        self.set_target_patrol() #设置target
        self.choose_target_patrol() #给每个agent选择target

        for k in range(self.__agent_num): 

            for w in range(2):
                self.__prev_current[k][w] = self.__current[k][w] #之前的位置状态也标记成这个，反正是初始化无所谓的

            self.__end_state[k] = False
            self.__conflict_state[k] = False

        for i in range(self.numMines):  #初始化雷的位置
            while True:
                x = random.randint(0, self.size - 1)
                y = random.randint(0, self.size - 1)

                if ( self.__avs[x][y] == 0 ) and ( self.__mines[x][y] == 0 ) and ( self.__target_map[x][y] == 0) :
                    self.__mines[x][y] = 1
                    break

        for a in range(self.__agent_num):
            # self.setCurrentBearing(a, random.randint(0, 7))
            self.setCurrentBearing( a, self.adjustBearing( self.getTargetBearing( a ) ) ) #调整智能体方向朝向target
            self.__prev_bearing[a] = self.__currentBearing[a]




    def getState_patrol(self, agt : int):
        #这里我都使用的是相对坐标而不是极坐标（距离range,相对bearing），也可以修改为极坐标，修改后记得Reward函数相应处也要修改~
        this_Sonar = [0] * 8
        that_Sonar = [0] * 8
        this_AVSonar = [0] * 8
        that_AVSonar = [0] * 8

        this_bearing = 0
        this_targetRange = 0.0

        # self.getSonar( agt, that_Sonar )   #获得mines和边界的声纳信息
        # self.getAVSonar( agt, that_AVSonar )   #autonomous vehicle (AV) 获得其他智能体和边界的声纳信息
        self.get_all_Sonar(agt, that_Sonar)
        for i in range(8):
            this_Sonar[i] = that_Sonar[i]

        this_bearing = self.getTargetBearing( agt )
        this_targetRange = self.getRange( agt ) #获得目标范围
        # relative_position = [this_targetRange, this_bearing]
        relative_position = [self.__target_list[agt][0] - self.__current[agt][0], self.__target_list[agt][1] - self.__current[agt][1]]



        attractor_range, attractor_bearing = self.Choose_Pheromone(agt)
        # pho_position = [attractor_range, attractor_bearing ]
        pho_position = [self.__Attractor_list[agt][0] - self.__current[agt][0], self.__Attractor_list[agt][1] - self.__current[agt][1]]



        # leader_range = self.get_i_j_Range(agt, self.leader) #获取领航者与智能体agt之间的距离
        # leader_bearing = self.get_a_b_Bearing(self.__current[self.leader], self.__current[agt]) ##获取领航者与智能体agt之间的相对方位
        # leader_position = [leader_range, leader_bearing]
        leader_position = [self.__current[agt][0] - self.__current[self.leader][0], self.__current[agt][1] - self.__current[self.leader][1]]

        return  np.hstack((this_Sonar, relative_position ,pho_position, [agt],leader_position))

    def getReward_patrol(self, agt : int):
        #这个函数应该是获得agt当前的奖励值
        # 返回值有三个 1.总的奖励函数 2.奖励函数中基于队形控制的惩罚项 3.next_state
        x = self.__current[agt][0]
        y = self.__current[agt][1]
        x0 = self.__prev_current[agt][0]
        y0 = self.__prev_current[agt][1]
        next_state = self.getState_patrol(agt)

        # 1.Separation 脱离队形的惩罚（与leader之间的距离和方位）
        # next_state[-2：-1]代表了state的后两个也就是和leader的相对坐标（如果你想使用极坐标（距离和相对方向），那么要修改Seq_reward)
        # 这里的是对应方格队形，按照如下排列
        #701
        #6 2
        #543
        should_range = 0
        Seq_reward = 0
        if(agt == 1):
            if(next_state[-2] == 1 and next_state[-1] == 0):
                Seq_reward = 0.0
            else:
                Seq_reward = - (abs(next_state[-2]) + abs(next_state[-1])) #如果智能体不在它应该在的位置上就给它基于相对坐标的惩罚
        elif(agt == 2):
            if(next_state[-2] == 1 and next_state[-1] == 1):
                Seq_reward = 0.0
            else:
                Seq_reward = - (abs(next_state[-2]) + abs(next_state[-1]))
        elif(agt == 3):
            if(next_state[-2] == 1 and next_state[-1] == 2):
                Seq_reward = 0.0
            else:
                Seq_reward = - (abs(next_state[-2]) + abs(next_state[-1]))
        elif(agt == 4):
            if(next_state[-2] == 0 and next_state[-1] == 2):
                Seq_reward = 0.0
            else:
                Seq_reward = - (abs(next_state[-2]) + abs(next_state[-1]))
        elif(agt == 5):
            if(next_state[-2] == -1 and next_state[-1] == 2):
                Seq_reward = 0.0
            else:
                Seq_reward = - (abs(next_state[-2]) + abs(next_state[-1]))
        elif(agt == 6):
            if(next_state[-2] == -1 and next_state[-1] == 1):
                Seq_reward = 0.0
            else:
                Seq_reward = - (abs(next_state[-2]) + abs(next_state[-1]))
        elif(agt == 7):
            if(next_state[-2] == -1 and next_state[-1] == 0):
                Seq_reward = 0.0
            else:
                Seq_reward = - (abs(next_state[-2]) + abs(next_state[-1]))
        else:
            pass

        #  2.Cohesion 靠近惩罚
        # Coh_reward =  -max(next_state[0:8]) #( 0 -> -0.1)
        Coh_reward = 0 #这是之前按照flocking规则设置的，后来我就没有用就设置为0了


        # 3. Alignment 方向惩罚
        Ali_reward = 0
        ggap = self.Bearing_gap(agt, self.leader) #获取智能体与领航者之间方向的差距
        if(ggap == 0):
            Ali_reward  = 0.0
        else:
            Ali_reward = -0.5 * self.Bearing_gap(agt, self.leader)#(-0.1 -> -0.4)



        # 4.抵达奖励
        if ( x == self.__target_list[agt][0] ) and ( y == self.__target_list[agt][1]) : # reach self.__target
            self.__end_state[agt] = True

            return 10.0 + (Seq_reward + Ali_reward + Coh_reward), (Seq_reward + Ali_reward + Coh_reward), next_state

        if ( x < 0 ) or ( y < 0 )  :# out of field
            return -10.0 + (Seq_reward + Ali_reward + Coh_reward),(Seq_reward + Ali_reward + Coh_reward),  next_state

        # 5.撞雷惩罚
        if self.__mines[x][y] == 1 :      # hit self.__mines 碰到雷返回0
            return  -10.0 + (Seq_reward + Ali_reward + Coh_reward), (Seq_reward + Ali_reward + Coh_reward), next_state


        # 6.碰撞惩罚
        if self.isConflict(agt) or self.check_i_conflict(agt) :
            return  -10.0 + (Seq_reward + Ali_reward + Coh_reward), (Seq_reward + Ali_reward + Coh_reward), next_state


        # 7.靠近或远离惩罚(应用Reward Shaping方法)
        now_Range = - 0.1 * self.getRange(agt) #(-0.1 -> -1.6)

        # 8.靠近或远离吸引子的奖励与惩罚
        attractor_reaward = 0.0#(0 -> 1) 
        # 这个感觉不能太大,智能体会在旗子处来回震荡,因为旗子处在初始化时,放了1个单位的信息素
        # print(self.__Attractor_list[agt])
        # print([x0,y0])
        if(self.__Attractor_list[agt][0] == -1 and self.__Attractor_list[agt][1] == -1):
            attractor_reaward = 0.0
        else:
            attractor_reaward = 1 / ((self.get_a_b_Range([x,y],self.__Attractor_list[agt])) + 1)
        # print("Agent :" + str(agt))
        # print("attractor_reaward :" + str(attractor_reaward))
        # print("now_Range : " + str(now_Range))
        # print("Seq_reward : " + str(Seq_reward))
        # print("Ali_reward : " + str(Ali_reward))
        # print("Coh_reward : " + str(Coh_reward))
        # print()

        return (Seq_reward + Ali_reward + Coh_reward + now_Range + attractor_reaward), (Seq_reward + Ali_reward + Coh_reward), next_state


    def choose_leader(self):
        '''
        这个函数是在领航者停止后，按序号顺序选择一个新的领航者
        '''
        if(self.leader >= self.__agent_num):
                return
        while(self.is_end(self.leader)):
            self.leader = self.leader + 1
            if(self.leader >= self.__agent_num):
                return
if __name__ == "__main__":
    m = Maze(4)
    m.refreshMaze_patrol(4)