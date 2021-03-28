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

- **神经网络架构**

    网络的输入是一个包含传感器信号、数字信息素信息和agent本身序列号的n维向量。第一隐层是全连接的，由256个整流线性单元(ReLu)组成。接下来是一个由两个流组成的决斗架构，分别评估状态值和每个操作的优势。最后，一个完全连接的线性层为每个有效操作投影输出。如图所示：
    
    <div align=center>
    <img src="https://github.com/Aaricis/BioMARL/blob/main/MARL/images/5.png" width="50%"/>
    </div>

- **生物启发式算法：Stigmergy**

***
## 实验结果
