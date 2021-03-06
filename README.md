# BioMARL:基于生物启发式算法的多智能体强化学习算法
## 项目介绍：
多智能体系统(MAS)由于具有解决复杂任务的灵活性、可靠性和智能性，已被广泛地应用于不同的应用领域，如计算机网络、机器人和智能电网等。通信是多代理世界保持组织和生产的重要因素。然而，以往的多代理通信研究大多试图预先定义通信协议或采用额外的决策模块进行通信调度，这将造成巨大的通信开销，并且不能直接推广到一个大型代理集合。本文提出了一个轻量级的通信框架:**基于信息素协同机制的分布式深度Q网络算法（Pheromone Collaborative Deep Q-Network, PCDQN）**，它结合了深度Q网络和stigmergy机制。它利用stigmergy机制作为部分可观察环境下独立强化学习代理之间的间接沟通桥梁。我们证明了PCDQN框架的优越性，同时也将我们的模型转移到解决多代理路径搜索问题上。利用PCDQN框架，多代理编队能够学习适当的策略，成功地在雷区导航环境中获得最优路径。
***
## 实验环境
```sh
Windows10操作系统，8GB内存，2核Intel Core i7-6500U
pytorch-1.4.0+cpu
python 3.8
```
***
## 实验平台
**雷区导航模拟器（Minefield Navigation Simulator）**

 <div align=center>
 <img src="https://github.com/Aaricis/BioMARL/blob/main/MARL/images/simulator.png" width="50%"/>
 </div>

***

## 实验原理

协调是分布式MAS的中心问题。智能体很少是独立的系统，通常涉及多个智能体并行工作以实现一个共同的目标。当使用多个智能体来实现一个目标时，需要协调它们的行动，以确保系统的稳定性。分布式体系结构中智能体之间的经典协调技术是通过通信机制。我们不是直接传递信息，而是将**间接交流**灌输到一般的RL架构中，这种交流基于其他代理的观察行为，而不是交流，以及它对环境的影响。这种类型的交流在生物学中被称为**Stigmergy**，它指的是基于环境变化的交流，而不是直接的信息传递。本文提出的**基于信息素协同机制的分布式深度Q网络框架**分别由**改进的DQN体系结构**和**Stigmergy**机制组成。

这两个部分不是独立的，而是互惠互利的。其中，DQN架构扮演着神经系统的角色，引导智能体的策略来适应环境，并从交互中学习以实现他们的目标。而Stigmergy作为独立学习主体间接沟通的桥梁，在协调过程中减少动态环境的负面影响。
    
<div align=center>
<img src="https://github.com/Aaricis/BioMARL/blob/main/MARL/images/4.png" width="60%"/>
</div>

接下来介绍网络架构的细节。

- **神经网络架构**

    网络的输入是一个包含传感器信号、数字信息素信息和agent本身序列号的n维向量。第一隐层是全连接的，由256个整流线性单元(ReLu)组成。接下来是一个由两个流组成的决斗架构，分别评估状态值和每个操作的优势。最后，一个完全连接的线性层为每个有效操作投影输出。如图所示：该架构融合Dueling网络、优先经验回放和双DQN之于原始的DQN网络架构。
    
    <div align=center>
    <img src="https://github.com/Aaricis/BioMARL/blob/main/MARL/images/5.png" width="50%"/>
    </div>

- **生物启发式算法：Stigmergy**

    具体使用了Stigmergy中的**信息素协同机制**
    
    基于信息素的协同机制有如下特点：
    
       （1）信息素由移动中的智能体所释放，且只有当智能体到达目标处时才会释放，只有正常移动中的智能体可以探测信息素；
       （2）每个位置的信息素会随着时间的推移逐渐挥发；
       （3）每个位置的信息素在每个时间步结束后会向四周扩散。
    如图所示：
    
    <div align=center>
    <img src="https://github.com/Aaricis/BioMARL/blob/main/MARL/images/6.png" width="40%"/>
    </div>
    
    在每个时间步，智能体在探测范围内探测地图上的信息素，并按照规则选取其中一处为“吸引子”，并将其极坐标作为状态输入的一部分。
    
***
## 实验结果
- **多智能体雷区导航实验**
    模拟环境中地图大小为16x16，地雷个数为15，智能体数量为4，每轮最大时间步为30，训练轮数为2000;
    
    训练结果如图：
    
    <div align=center>
    <img src="https://github.com/Aaricis/BioMARL/blob/main/MARL/images/9.png" width="50%"/>
    </div>
    
    多智能体导航效果如下所示：
    
    <div align=center>
    <img src="https://github.com/Aaricis/BioMARL/blob/main/MARL/images/11.png" width="50%"/>
    </div>
    
- **集群编队控制**
    
    编队控制路径轨迹实验效果对比图：左边为传统DQN方法，右边为PCDQN（即我们提出的新方法）
    
    <div align=center>
    <img src="https://github.com/Aaricis/BioMARL/blob/main/MARL/images/16_1.png" width="40%"/>
    </div>
    
    <div align=center>
    <img src="https://github.com/Aaricis/BioMARL/blob/main/MARL/images/16_2.png" width="40%"/>
    </div>
