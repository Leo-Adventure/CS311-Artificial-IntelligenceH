## AI 期末复习

> Author: MQ-Adventure
>
> Duration:  Summer Semester
[TOC]

### 智能体（Intelligence Agents）

#### Agent

Agent: perceiving and acting according to environment

running cycle: (1) perceive (2) think (3) act

**Agent = Architeture + Program**

Agent function: mapping from **percepts** to **actions**

#### Rational Agent

对每个可能的感知序列，根据已知的感知序列提供的证据和 Agent 具有的先验知识，理性 Agent 选择使其性能度量最大化的行动(maximize its expected performance)

#### PEAS

定义理性智能体时，使用 PEAS: 

- Performance（性能）
- Environment（环境）
- Actuators（执行器）
- Sensors（传感器）

#### Environment Types

完全可观测（Fully Observable）

确定性（Deterministic）

连续性（Episodic）

静态（Static）：环境是否会在决策时变化

离散（Discrete）

单/多智能体（Single/ multi）

已知（known）

#### Agent Types

- Simple Reflex Agents（环境必须是完全可观测）

- Model-Based Reflex Agents（根据感知历史序列来处理部分可观测的环境）
- Goal-Based Agents（以目标为行动指引，结合目标以及环境模型考虑未来状态进行决策）
- Utility-Based Agents（在目标之外，还追求效益）

#### Learning Agents

Four conceptual components

1. Learning element（负责做出改进）
2. Performance element（负责选择行动）
3. Critic（评价智能体做得多好）
4. Problem generator（使得智能体进行探索）（牺牲短期性能，使得长期更好）

#### Agent states

- Atomic Representation （原子表示）（只关心智能体位置，不关心内部状态）
- Factored Representation （要素化表示）（每个状态有属性值）
- Structured Representation（结构化表示）（能够表示对象之间的关系）

### Uninformed Search

#### Problem Formulation

- Initial state
- State
- Actions
- Transition model
- Goal test
- Path cost

#### Search

树搜：没有重复的状态

图搜：有重复的状态，需要进行记录

#### Search Strategies

b - 搜索树的最大分支系数

d - 解的深度（最浅深度）

m - 状态空间的最大深度，可能是无限

#### BFS

使用队列维护节点，先扩展最浅的节点

![](picture\BFS.png)

时间复杂度：$O(b^d)$

空间复杂度：$O(b^d)$

#### DFS

使用栈维护节点

时间复杂度：$O(b^m)$

空间复杂度：$O(bm)$

#### DLS(Depth-limited search)

有深度限制的 DFS

![](picture\DFS.png)

#### IDS(Iterative deepening search)

结合了 BFS 和 DFS

DLS with increasing depth limits

#### UCS(Uniform-cost search)

使用边代价为深度估计的 BFS，使用**优先队列**对边权进行排序，每次挑选边权最小的进行扩展

![](picture\UCS.png)

### Informed Search & Local Search

有信息和无信息的区别在于全局信息

#### Greedy Search

维护优先队列，每次挑选与自己相邻的与终点距离最近的往下走

![](picture\Greedy.png)



#### A* Search

除了考虑当前节点距离终点的距离，在启发函数当中还考虑从起始节点到当前节点的历史距离。

![](picture\A_star.png)

和 `Greedy`算法的区别只在于启发函数的不同。

### Local Search

若到达目标的路径不重要，则可以考虑使用局部搜索

并且如果希望找到纯最优解，则也可以考虑使用局部搜索

#### Hill Climbing 爬山算法

也叫做贪心局部搜索，只会看相邻的邻居节点会不会更优。

在到达顶峰的时候终止

优点：

- 不用维护搜索树
- 几乎不用内存
- 常常可以在一个连续或更大的状态空间当中找到一个足够好的解

![](picture\Climb.png)

#### Genetic Algorithm(GA)

后继状态来自于两个父代的结合，而不是来自单一状态的调整

从 `k`个随机产生的状态开始（称为种群），每个状态都是一个个体

目标函数称为 fitness function，更好的状态有着更高的目标函数的值

在当前种群中选择两个父本，进行字符串的杂交之后产生子代，放入到新的种群当中，直到种群中的个体表现很优秀或者时间到了，此时选择种群当中最好的子代返回。

![](picture\GA.png)

#### Simulated Annealing

由爬山算法演变而来，即使是在邻近解比当前解差的情况下，也会有$e^{-\Delta E/T}$的概率移动到邻近解。

![](C:\大二课程\人工智能\picture\SA.png)