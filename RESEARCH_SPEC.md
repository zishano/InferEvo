研究与开发任务：构建 Agent 驱动的自主演进式 LLM 推理架构仿真与探索系统
1. 角色定义
你是一名资深 AI Infrastructure 系统架构师、LLM Serving 研究员、GPU 性能建模专家及自主系统优化研究员。
你具备以下领域的专业能力：
大规模 LLM 推理服务系统设计。
GPU 计算、访存、通信与硬件资源建模。
MoE、MLA、KDA、Hybrid Attention 等模型架构。
TP、DP、EP、ETP、PP、CP、DCP 等并行机制。
PD 分离、AF 分离及其他分离式推理架构。
离散事件仿真、动态 DAG、计算图与硬件图建模。
Design Space Exploration（DSE）与多目标 Pareto 优化。
Agentic Infrastructure、Recursive Self-Improvement（RSI）及自主系统演进。
大型软件系统的模块化设计、依赖管理、扩展控制与独立验证。
你的任务不是简单地执行预定义开发步骤，而是作为整个项目的技术负责人，自主完成现有系统分析、技术路线选择、架构设计、代码开发、实验验证和持续优化。
2. 项目背景
现有 LLM Serving 系统及其仿真工具存在以下局限：
传统推理仿真器通常只能模拟已经实现的 Serving 机制。
新机制往往需要人工修改多个相互耦合的系统组件，扩展成本较高。
现有设计空间搜索通常局限于预定义参数，例如 TP、DP、EP、CP、PP、Batch Size 和 P/D 资源比例。
基于真实系统 Profiling 的仿真方法难以预测尚未实现的新推理机制。
现有 Agent 优化工作通常依赖真实 GPU 或真实 Serving 系统进行性能反馈，难以直接用于未来系统架构探索。
缺少能够同时探索 Serving 系统机制、底层计算通信组织及硬件资源映射的统一自主研究平台。
本项目希望突破上述限制，利用 Simthesizer 的动态 DAG 和快速机制扩展能力，结合 RoofLang 的计算图、硬件图、算子性能预测和设计空间搜索能力，构建一个能够自主发现并演进新型 LLM 推理架构的仿真系统。
本项目不具备可供持续实验的真实 GPU 集群或完整真实 Serving 系统。
因此，所有自主优化、机制扩展和设计空间探索主要在仿真环境中完成。
3. 核心研究目标
构建一个 Agent-driven Self-Evolving LLM Inference Architecture Explorer。
系统需要具备以下完整能力：
根据模型、硬件、工作负载和初始 Serving 架构，执行端到端推理仿真。
自动获取细粒度性能反馈，并分析当前系统的主要瓶颈。
由 Agent 自主提出可能解决瓶颈的新机制、新策略或新架构。
由 Agent 分析新机制涉及的组件、依赖关系及修改范围。
自动扩展相应仿真组件和底层计算通信图。
对新机制进行语义正确性、资源可行性和系统一致性验证。
自动搜索新机制对应的并行策略、资源映射及配置空间。
评估新机制的端到端性能和适用条件。
根据实验结果保留、修改、组合或淘汰候选机制。
将有效机制作为后续研究的基础，持续执行多轮自主优化。
最终目标不是寻找某个已有推理系统的最佳配置，而是：
在没有真实系统的情况下，通过 Agent 自主生成、扩展、组合和验证新的推理机制，探索当前尚未实现的推理架构，并预测下一阶段 LLM Serving 系统可能的演进方向。
4. 基础仓库与技术路线
4.1 核心开发仓库
Simthesizer
仓库：
https://github.com/casys-kaist/Simthesizer
主要负责：
系统级动态 DAG。
请求生命周期。
Workload 建模。
Router 与 Scheduler。
动态 Batch。
KV Cache 与模型状态管理。
Serving 资源分配。
PD、AF 等分离式架构。
新系统机制的快速表达与扩展。
端到端事件仿真。
原则上，将 Simthesizer 作为整个动态 Serving 仿真系统的主体。
但必须先阅读实际源码，确认其组件边界、扩展接口、事件执行语义及现有能力，不得仅根据论文摘要假设其实现。
RoofLang
使用当前项目中已有的 RoofLang 仓库及相关扩展版本。
主要负责：
模型计算图。
算子性能预测。
硬件资源抽象。
Tensor Layout。
算子与 Tensor 放置。
GPU 并行机制。
计算、访存和通信成本。
GPU 拓扑与通信组织。
计算通信重叠。
显存容量及资源可行性。
底层架构设计空间探索。
RoofLang 是本项目唯一的底层算子性能预测后端。
不引入 LLMCompass，不额外维护第二套独立的算子性能预测框架。
如果 RoofLang 缺少必要的算子、硬件能力或通信机制，应优先考虑在其现有抽象基础上进行合理扩展。
4.2 参考仓库
可以调研以下项目，但不要求将其直接集成到运行系统中：
ServingStudio：Agent Skills、自动瓶颈分析与优化工作流。
BLIS：模块化系统架构、成本模型解耦与接口管理。
Frontier：现代 Serving 机制、状态管理、校准与回归测试。
LLMServingSim 2.0：模型目录、分布式推理仿真及公开性能数据。
AISimulate：硬件描述、部署配置搜索及已有系统基线。
ServeTwin：无目标硬件条件下的性能预测方法。
ROCm InferaSim：低成本性能建模与配置搜索。
RSIBench-Infra：自主优化任务组织与独立验证方法。
上述项目主要作为架构设计、机制实现、公开数据和研究方法的参考来源。
避免为了增加功能而直接拼接多个完整仿真器。
4.3 总体设计原则
原则上采用以下职责分工：
Simthesizer 决定系统执行什么、何时执行以及系统状态如何变化。
RoofLang 决定计算任务如何映射到硬件、如何执行以及需要多少资源和时间。
Agent 决定应该分析什么、修改什么、扩展什么以及下一轮探索什么。
独立验证系统负责判断候选机制是否正确、可行和具有可信的预测收益。
允许根据实际源码分析调整具体技术路线，但必须说明调整理由、收益和潜在代价。
4.4 强制要求：建立技术来源、源码参考与系统模块之间的映射关系
本项目不是从零开始构建一个全新的仿真系统，而是在已有开源系统、研究论文和成熟算法的基础上进行组合、扩展与创新。
在设计和开发任何核心模块之前，必须主动调研对应的参考仓库、论文、算法及实现方式。
不能只知道需要开发什么，还必须理解：
为什么需要这个模块？
已有哪些工作解决了类似问题？
对应的技术来源是什么？
参考仓库具体如何实现？
哪些代码、接口、算法或设计思想可以复用？
哪些能力不适合直接复用？
如何适配本项目无真实系统、开放式机制探索和组件级受控演进的要求？
本项目需要在已有工作基础上增加什么能力？
以下技术映射是初始研究指导，不是要求机械照搬参考仓库的实现。Agent 应通过实际源码和论文分析不断完善技术来源及开发方案。
4.4.1 联合仿真执行引擎
核心技术来源：
Simthesizer：https://github.com/casys-kaist/Simthesizer
RoofLang：当前项目已有仓库。
BLIS：https://github.com/inference-sim/inference-sim
AISimulate：https://github.com/ai-dynamo/aisimulate
重点借鉴内容：
Simthesizer：
Unified Dynamic DAG。
动态事件与执行依赖表达。
Serving 机制扩展方式。
计算模拟器注册与调用机制。
请求状态及动态系统行为。
Synthesizer Agent 扩展新机制的方法。
RoofLang：
模型计算图与硬件图。
算子性能预测后端。
Tensor Layout 与设备映射。
并行策略表达。
计算、访存和通信成本建模。
计算通信重叠。
资源可行性与设计空间搜索。
BLIS：
Serving Simulator 与 Latency Kernel 的职责分离。
性能模型接口。
硬件及模型描述管理。
统一配置和数据格式。
AISimulate：
底层性能预测与动态 Serving 仿真的组合方式。
配置搜索与性能查询之间的关系。
性能预测结果的复用。
开发要求：
首先阅读 Simthesizer 和 RoofLang 的实际源码，定位计算模拟器注册、调用、图构建、性能预测及资源建模的入口。
重点研究 BLIS 和 AISimulate 如何解决系统仿真与性能预测之间的接口问题。
在此基础上设计本项目的 Execution IR 和 RoofLang Adapter。
不要默认重新开发 Simthesizer 已经具备的能力，也不要在 Simthesizer 中重新实现一套 RoofLang 算子性能模型。
RoofLang 必须作为唯一底层算子性能预测后端，不引入 LLMCompass。
4.4.2 Research Agent Controller 与自主研究工作流
核心技术来源：
ServingStudio：https://github.com/SyFI-ServingStudio/ServingStudio
Simthesizer：https://github.com/casys-kaist/Simthesizer
KD-DSE：https://arxiv.org/abs/2611.22411
RoofLang 已有 RSI Agent、Harness、Skills 和自主搜索实现。
Recursive Self-Improvement 研究思想。
Plan-and-Execute、ReAct、反思式迭代及状态机工作流。
重点借鉴内容：
ServingStudio：
Agent Skills 的组织方式。
仿真驱动的瓶颈诊断。
优化任务分解。
仿真器扩展与优化实验之间的关系。
Agent 自主制定研究计划的方法。
Simthesizer：
Synthesizer Agent 如何理解已有仿真器。
如何根据目标机制生成扩展代码。
如何组织动态 DAG 相关的机制实现。
RoofLang：
已有自主优化循环。
性能反馈如何提供给 Agent。
候选生成与评估方式。
已有 RSI 工作流、搜索约束和实验管理能力。
KD-DSE：
知识驱动的设计空间探索。
根据性能瓶颈选择优化方向。
利用架构知识减少无效候选。
有限评估预算下的搜索策略。
开发要求：
优先分析并复用当前 RoofLang 和 Simthesizer 已有的 Agent Infrastructure，不要直接从零实现复杂的多 Agent 框架。
研究如何将现有 Agent 能力统一为一个自主研究 Controller。
Controller 应具备分析、设计、规划、实现、搜索、验证、选择和反思等逻辑职责，但不强制每种职责必须对应独立 Agent。
Agent 必须能够根据实际实验结果自主改变研究方向、回退失败机制、扩大搜索空间并继续迭代。
不能将自主研究流程退化为固定顺序执行的一组预定义脚本。
4.4.3 Dense Feedback 与自动瓶颈诊断
核心技术来源：
RoofLang 已有性能分析与逐层反馈能力。
ServingStudio：https://github.com/SyFI-ServingStudio/ServingStudio
KD-DSE：https://arxiv.org/abs/2611.22411
Roofline Model。
Critical Path Analysis。
Resource Utilization Analysis。
Sensitivity Analysis。
CAFEML：https://arxiv.org/abs/2611.26840
重点借鉴内容：
Roofline：分析计算受限与访存受限。
Critical Path：识别真正决定执行时间的依赖路径。
ServingStudio：瓶颈驱动的 Agent 优化流程。
KD-DSE：根据瓶颈类型选择设计空间。
CAFEML：考虑并发算子之间的资源争用，避免理想化重叠模型。
RoofLang：算子、GPU、计算图及通信图的细粒度性能反馈。
开发要求：
首先审计 RoofLang 和 Simthesizer 当前能够输出哪些性能指标，哪些指标可以直接复用，哪些需要补充。
建立统一的跨层性能反馈格式，覆盖：
算子级。
Transformer 层级。
GPU 级。
网络链路级。
Serving Worker 级。
请求级。
Agent Session 级。
瓶颈诊断不能只根据耗时占比。
必须区分：
总耗时较大的操作。
位于关键路径上的操作。
可以被其他操作隐藏的开销。
因资源争用而暴露的开销。
由于负载不均衡产生的等待。
由调度和资源分配引起的系统瓶颈。
必要时通过受控反事实实验验证瓶颈，例如单独提高 HBM 带宽、计算效率或网络带宽，观察端到端性能变化。
Agent 应利用诊断结果选择下一轮优化方向，而不是仅根据最终吞吐随机生成候选。
4.4.4 Mechanism Manager 与组件级受控扩展
核心技术来源：
BLIS：https://github.com/inference-sim/inference-sim
Simthesizer：https://github.com/casys-kaist/Simthesizer
ServingStudio：https://github.com/SyFI-ServingStudio/ServingStudio
Plugin Architecture。
Dependency Injection。
Interface Contract。
Dependency Graph。
Topological Sorting。
Change Impact Analysis。
Git Branch / Git Worktree。
Transactional Update 与版本回退思想。
重点借鉴内容：
BLIS：
系统组件与成本模型之间的模块化解耦。
独立模块的接口管理。
模型、硬件、校准数据及配置的分离。
Simthesizer：
通过动态 DAG 扩展新的系统行为。
将新机制表达为可执行的仿真组件。
减少新增机制对其他组件的侵入。
ServingStudio：
Agent 如何扩展仿真器能力。
如何将研究任务转换为具体代码修改任务。
软件工程方法：
通过组件注册管理可用机制。
通过依赖图识别修改影响范围。
通过接口契约保证组件兼容。
通过拓扑排序安排组件扩展顺序。
通过独立候选版本隔离不同机制。
通过回归测试及版本回退保护稳定系统。
开发要求：
本模块是项目的重点自主研发模块。
不能假设 BLIS 或 Simthesizer 已经完整提供 Agent 受控组件演进能力。
应结合已有框架的真实扩展接口，自主设计：
Component Registry。
Mechanism Specification。
Component Dependency Graph。
Change Impact Analyzer。
Extension Planner。
Candidate Isolation。
Compatibility Checker。
Regression Test Selector。
Mechanism Version Manager。
上述名称仅代表逻辑职责，不要求机械地实现九个独立模块。
Agent 在扩展新机制之前，应先分析已有组件及其依赖，制定修改计划，并确定：
哪些组件可以直接复用。
哪些组件需要新增。
哪些组件需要修改。
哪些接口需要扩展。
哪些组件需要重新验证。
修改顺序及失败回退方式。
优先使用非侵入式扩展，但当现有接口无法表达新机制时，允许 Agent 提出受控的核心引擎修改。
不能因为强调模块化而限制真正的跨层架构创新。
4.4.5 开放式设计空间搜索与机制演进
核心技术来源：
RoofLang 已有 Design Space Exploration。
AISimulate：https://github.com/ai-dynamo/aisimulate
ROCm InferaSim：https://arxiv.org/abs/2611.04032
KD-DSE：https://arxiv.org/abs/2611.22411
Evolutionary Search。
NSGA-II / Multi-objective Optimization。
Pareto Archive。
Hierarchical Search。
Constraint-based Pruning。
Multi-fidelity Optimization。
MAP-Elites / Quality-Diversity Search。
重点借鉴内容：
RoofLang：
已有并行策略及硬件映射搜索。
计算图和通信图的设计空间。
性能目标及资源约束。
候选生成和可行性判断。
AISimulate：
已有 Serving 配置空间的搜索方式。
部署配置及资源分配搜索。
InferaSim：
降低候选评估成本。
利用少量高价值评估指导后续搜索。
分阶段筛选和自动调优思想。
KD-DSE：
根据架构知识和瓶颈反馈缩小搜索空间。
避免无依据的随机探索。
进化搜索与 Quality-Diversity：
保留多个具有不同特征的有效机制。
支持机制变异、组合和进一步扩展。
避免过早收敛到单一局部最优架构。
开发要求：
区分两类本质不同的搜索：
第一类：在已有机制中搜索最优参数与资源配置。
第二类：由 Agent 生成新的机制、新组件和新的设计变量，从而扩大设计空间。
传统 DSE 算法可以用于第一类搜索。
第二类搜索需要 Agent 根据瓶颈、架构知识和历史实验，自主改变设计空间的结构。
不能仅将所有机制提前编码为固定参数，再声称实现了开放式架构探索。
应支持：
动态新增设计变量。
动态新增机制。
不同机制对应不同参数空间。
不同机制之间的组合。
多目标 Pareto 比较。
多候选并行探索。
无效候选剪枝。
搜索预算分配。
历史知识复用。
不要求同时实现所有列出的搜索算法。
Agent 应根据实际设计空间特点、评估成本和实验结果，自主选择适合的搜索方法。
4.4.6 Kimi-K3 模型与 H200/B300 硬件建模
核心技术来源：
Kimi-K3 官方模型配置、技术报告及模型实现。
NVIDIA H200、B300 官方硬件资料。
RoofLang 已有模型及硬件抽象。
LLMServingSim：https://github.com/casys-kaist/LLMServingSim
AISimulate：https://github.com/ai-dynamo/aisimulate
Frontier：https://github.com/NetX-lab/Frontier
SGLang：https://github.com/sgl-project/sglang
vLLM：https://github.com/vllm-project/vllm
Alibaba InferSim：https://github.com/alibaba/InferSim
重点借鉴内容：
LLMServingSim：
模型目录及模型结构描述。
多 GPU 推理仿真中的模型参数管理。
已公开的性能验证数据。
AISimulate：
硬件描述及拓扑参数。
GPU 精度支持和部署配置。
已有推理性能预测参考。
SGLang 与 vLLM：
Kimi-K3 的实际模型执行流程。
KDA、MLA、Latent MoE 的实现。
权重量化及反量化路径。
不同硬件对应的 Kernel 选择。
KV Cache 与 Recurrent State 管理。
TP、DP、EP、DCP 等机制的真实执行语义。
Frontier 与 InferSim：
现代推理机制的性能建模方法。
算子、通信及系统开销表达。
公开性能对齐方法。
开发要求：
所有模型和硬件参数必须有明确来源。
优先从官方配置、模型代码和硬件规格中提取参数，再参考其他仿真器的模型描述。
必须检查：
Kimi-K3 实际模型结构。
Prefill 与 Decode 的算子差异。
KDA State 与 MLA KV Cache 的差异。
H200 与 B300 的低精度执行路径差异。
不同硬件拓扑下的通信原语可行性。
权重、激活、状态、通信缓冲区及运行时显存。
参考 vLLM 和 SGLang 的目的是理解已有机制的数学及执行语义，而不是将它们当前支持的功能作为本项目的机制边界。
允许 Agent 设计超出现有真实框架能力的新机制，但必须给出合理的执行方式、物理资源需求及可实现性依据。
4.4.7 独立验证与仿真可信度
核心技术来源：
RSIBench-Infra：https://arxiv.org/abs/2611.24395
Frontier：https://github.com/NetX-lab/Frontier
ServeTwin：https://arxiv.org/abs/2610.02732
ROCm InferaSim：https://arxiv.org/abs/2611.04032
Simthesizer 的机制扩展验证方法。
Property-based Testing。
Differential Testing。
Metamorphic Testing。
Regression Testing。
Sensitivity Analysis。
Uncertainty Quantification。
重点借鉴内容：
RSIBench-Infra：
优化环境与最终评估环境的隔离。
独立验证契约。
防止 Agent 通过修改评估过程获得虚假收益。
可复现候选及实验记录。
Frontier：
仿真模型校准。
端到端性能误差分析。
不同 Serving 机制的验证。
ServeTwin：
缺少目标硬件时的性能预测。
基于公开 Benchmark 的仿真可信度分析。
硬件参数变化下的架构探索。
InferaSim：
性能模型的低成本校准。
代表性测量数据与其他配置之间的外推思想。
软件测试与数值分析方法：
数学语义验证。
数据布局与通信正确性验证。
状态一致性验证。
资源守恒验证。
回归验证。
不确定性与敏感性分析。
开发要求：
本项目没有真实 GPU 系统，因此不能直接照搬依赖真实硬件持续测量的验证流程。
应建立独立于优化 Agent 的验证体系，包括：
小规模 CPU 数学语义 Reference。
Tensor Layout 与数据依赖检查。
KV Cache 与 Recurrent State 一致性检查。
显存与硬件资源约束检查。
网络拓扑与通信可行性检查。
仿真事件及状态转换检查。
已有机制回归测试。
公开数据校准。
性能模型敏感性分析。
新机制预测收益的稳健性评估。
禁止优化 Agent 在普通机制优化过程中直接修改最终评估标准。
对于新机制需要扩展成本模型的情况，应采用独立变更流程，并重新评估基线及相关候选。
最终结果必须明确区分数学正确性、物理可行性、预测收益和真实硬件验证状态。
4.4.8 知识积累、候选管理与持续自主演进
核心技术来源：
RoofLang 已有 RSI 实验管理能力。
Recursive Self-Improvement。
Evolutionary Search。
Pareto Archive。
MAP-Elites。
Experience Replay。
Experiment Tracking。
Ablation Study。
Git 版本谱系。
重点借鉴内容：
记录候选机制的父子关系。
保留不同优化方向的优秀候选。
保存失败原因及适用条件。
根据历史经验避免重复无效搜索。
支持机制组合和进一步变异。
根据新硬件和新负载重新评估已有机制。
开发要求：
不要只保存每轮性能最高的候选。
应建立机制谱系和实验档案，记录：
机制来源。
父机制。
组件依赖。
代码版本。
适用模型。
适用硬件。
工作负载。
性能结果。
资源约束。
预测不确定性。
失败原因。
已知局限。
可组合机制。
Agent 应能够检索历史实验，判断哪些机制值得迁移、组合、继续优化或重新验证。
最终形成可持续积累的推理架构机制库，而不是一次性优化脚本。
4.4.9 Agentic Workload 与动态请求依赖
核心技术来源：
Simthesizer 的动态 DAG。
AgentServeSim：https://arxiv.org/abs/2606.09613
ServeTwin：https://arxiv.org/abs/2610.02732
Contexture：https://arxiv.org/abs/2611.26980
公开 Coding Agent 执行轨迹。
Closed-loop Workload Simulation。
重点借鉴内容：
AgentServeSim：
以完整 Agent Program 而非独立请求作为仿真对象。
前序模型调用和工具执行对后续请求释放时间的影响。
跨轮状态及 KV 保留策略。
ServeTwin：
闭环 Serving Benchmark。
多轮 Agent 工作负载的系统级性能评估。
Contexture：
Agent 上下文与 Serving Runtime 之间的相互作用。
缓存和上下文管理策略对后续工作负载的影响。
Simthesizer：
利用动态 DAG 表达运行过程中不断产生的新任务和依赖。
开发要求：
支持从传统固定请求 Trace 逐步扩展到具有因果依赖的 Agent 工作负载。
需要明确区分：
外部请求到达。
Agent 内部模型调用。
工具执行。
多轮上下文增长。
KV Cache 与 Recurrent State 保留。
并行子任务。
任务完成条件。
允许 Agent 自主探索新的上下文管理、任务调度和状态保留机制。
如果某项优化改变了模型可见上下文、推理语义或任务质量，则必须单独定义质量约束，不能直接将其与语义完全等价的系统优化混为一谈。
4.5 强制要求：技术来源审计与开发前调研
在正式开发每个核心模块之前，必须完成对应的技术来源审计。
对于每个参考仓库或论文，至少记录：
项目名称。
官方仓库及论文链接。
仓库 Commit Hash 或论文版本。
与当前模块相关的核心贡献。
相关源码文件及目录。
关键类、函数、接口或算法。
真实执行流程。
可以直接复用的能力。
可以借鉴但需要重新实现的思想。
不适合本项目的设计及原因。
开源协议及代码复用限制。
本项目需要新增的能力。
对应的开发计划。
对应的验证方式。
必须通过实际源码或论文确认技术细节。
不得根据项目名称、README 中的宣传描述或未经核实的二手资料，直接认定某个功能已经存在。
如果参考仓库不可访问、论文链接失效或相关功能不存在，应明确记录，并主动寻找可靠的替代来源。
不能编造源码路径、函数名称、算法细节或项目能力。
4.6 建立技术来源与实现追踪文档
在项目中建立并持续维护以下文档：
TECH_SOURCE_MAP.md
记录每个系统模块对应的技术来源。
建议字段：
Module。
Reference Project / Paper。
Repository URL。
Commit / Version。
Relevant Source Files。
Relevant Functions。
Borrowed Concept。
Reuse Type。
Required Adaptation。
Implementation Status。
Validation Evidence。
REFERENCE_ANALYSIS.md
记录各参考项目的深入源码及论文分析，包括：
系统架构。
核心算法。
执行流程。
扩展接口。
优势及局限。
与本项目的差异。
可复用实现。
需要自主研发的部分。
COMPONENT_DESIGN.md
记录本项目每个组件的设计，包括：
组件职责。
技术来源。
设计理由。
输入输出。
接口契约。
依赖关系。
扩展方式。
失败模式。
验证方法。
DECISIONS.md
记录关键技术选择及其理由，包括为什么采用某个参考项目的思想，以及为什么没有采用其他方案。
MECHANISM_CATALOG.md
记录系统已有机制和 Agent 自主生成的新机制，并区分：
已有公开机制。
已有机制的新组合。
对已有机制的改进。
自主提出的新机制。
尚未验证的新架构假设。
不得将已有公开机制重新实现后直接标记为本项目原创创新。
4.7 技术借鉴与自主研发的边界
必须明确区分以下四种技术使用方式：
A. 直接集成
实际调用已有仓库中的代码和能力。
本项目核心直接集成对象：
Simthesizer。
RoofLang。
B. 选择性代码复用
在确认源码质量、接口兼容性及开源协议后，复用其他仓库中确实适合本项目的独立组件。
不要求将完整仓库引入运行时。
C. 设计与算法思想借鉴
阅读参考实现或论文，理解其原理，再根据本项目约束自主设计实现。
主要包括：
ServingStudio 的 Agent 工作流。
BLIS 的模块化接口设计。
KD-DSE 的知识驱动搜索。
AISimulate 的配置搜索。
Frontier 的校准方法。
RSIBench-Infra 的独立验证。
进化搜索、Pareto Archive 和其他成熟算法。
D. 自主研发
针对本项目独有需求开发新能力，主要包括：
Simthesizer 与 RoofLang 的统一执行契约。
跨层 Dense Feedback。
Agent 自主机制生成与规划。
组件级受控演进。
动态扩展设计空间。
无真实系统条件下的独立可信评估。
跨硬件机制迁移。
面向未来推理架构的预测体系。
以上分类是初始建议。
Agent 应在源码审计后更新分类，并记录实际复用和自主开发的边界。
4.8 开发执行原则
对于每个待开发模块，按照以下逻辑自主推进：
明确模块需求
→ 检索技术来源
→ 阅读论文与实际源码
→ 分析可复用能力
→ 比较备选实现
→ 确定技术路线
→ 定义接口与组件边界
→ 实现最小可运行版本
→ 独立验证
→ 集成测试
→ 更新技术来源与设计文档。
不要为了满足形式要求而机械地调研所有参考仓库。
根据当前开发任务选择最相关的技术来源，并在必要时进一步搜索新的研究工作。
每个重要模块必须能够回答：
技术从哪里来、原始实现如何工作、本项目如何借鉴、为什么这样设计、哪些部分属于自主研发，以及如何证明该实现有效。
技术来源审计应贯穿整个研发过程，而不是只在 Phase 0 执行一次。
当 Agent 自主提出新机制时，也应主动检索已有研究，判断其是否已有类似实现，并据此调整机制设计和创新判断。
最终目标是建立一个有明确技术传承、可追溯实现依据、可持续自主创新的仿真研究系统，而不是一个由 Agent 根据抽象需求随意拼接代码的工程项目。
5. 模型与硬件仿真目标
5.1 目标模型
以 Kimi-K3 作为主要研究模型。
首先从官方模型配置、技术报告、模型实现及公开 Serving 框架中获取并核实：
总参数量与激活参数量。
Transformer 层结构。
KDA 与 MLA 的层级分布。
Attention Residuals 等特殊结构。
Latent MoE 结构。
Routed Experts 与 Shared Experts。
专家路由及激活机制。
Tensor Shape。
KV Cache 与 Recurrent State。
权重量化、激活精度及 Scale 数据。
Prefill 和 Decode 的实际计算流程。
不得仅根据模型名称或其他模型的实现进行近似替代。
所有模型参数必须具有可追溯的公开来源或明确的推导依据。
第一阶段优先支持文本推理的完整 Prefill 和 Decode 路径。
视觉编码器、推测解码及其他高级功能可以根据研究需求逐步扩展。
5.2 目标硬件
主要研究两类 GPU 集群：
NVIDIA H200。
NVIDIA B300。
优先采用可比较的 HGX 8-GPU 节点拓扑。
如需研究 GB300 NVL72 等不同 Scale-up 架构，应作为独立硬件场景，不得与 HGX B300 混用。
硬件描述至少包括：
GPU 架构。
显存容量与实际可用容量。
HBM 带宽。
各精度 Tensor Core 算力。
SM 与相关计算资源。
NVLink 与 NVSwitch。
节点内和节点间拓扑。
Scale-up 与 Scale-out 带宽。
通信启动延迟。
Copy Engine 等数据搬运资源。
不同数据精度和算子实现的硬件支持能力。
必须特别处理 H200 与 B300 在 FP4、MXFP4、FP8 和 BF16 执行路径上的差异。
不得在 H200 上直接使用仅由 Blackwell 架构支持的原生低精度计算能力。
5.3 GPU 规模
初步覆盖：
8 GPU。
16 GPU。
32 GPU。
64 GPU。
128 GPU。
256 GPU。
不要求所有 GPU 规模均可部署 Kimi-K3。
系统应自动判断模型权重、缓存、激活、通信缓冲区及运行时资源是否满足显存约束，并排除不可行配置。
5.4 工作负载
至少支持：
固定 Batch 的 Prefill 与 Decode。
不同输入长度与输出长度。
长上下文推理。
混合长短请求。
动态请求到达。
多轮 Coding Agent 工作负载。
多并发 Agent Session。
具有因果依赖的动态任务 DAG。
区分 Open-loop Trace Replay 与 Closed-loop Agent Workload。
对于闭环 Agent 工作负载，后续请求的释放时间应受到前序推理、工具执行和任务依赖的影响。
不得将所有 Agent 工作负载简单转换为固定到达时间的独立请求。
6. 系统架构设计要求
6.1 不预先限制最终系统结构
你需要根据实际仓库分析，自主确定最合理的系统架构。
以下模块是建议的逻辑职责，而非强制要求的目录结构：
Research Agent Controller。
Mechanism Manager。
Component Registry。
Dependency Manager。
Dynamic Serving Simulator。
RoofLang Performance Backend。
Execution IR / Adapter。
Dense Feedback Engine。
Independent Validator。
Experiment Manager。
Knowledge Base。
Pareto Archive。
需要分析哪些模块可以直接复用现有仓库，哪些模块必须新建，哪些模块适合通过 Adapter 实现。
避免不必要的重复开发和过度抽象。
6.2 统一执行契约
设计 Simthesizer 与 RoofLang 之间的统一执行契约。
该契约应能够表达：
模型及算子类型。
Prefill / Decode 阶段。
Batch 中每个请求的状态。
Tensor Shape 和数据精度。
新输入 Token 数量。
已缓存 Token 数量。
KV Cache 与 Recurrent State。
Tensor Layout。
GPU 放置及并行组。
计算与通信依赖。
资源需求。
算子执行时间。
状态读写及更新。
显存占用与生命周期。
需要支持两类执行粒度：
快速 Cost Query：用于已有机制的大规模配置搜索。
细粒度 Execution DAG：用于新机制表达、跨层优化和资源争用分析。
不要默认在每个 Decode Step 中运行完整 RoofLang 搜索。
应设计合理的性能缓存、查询机制、插值或其他加速方法，并保证新机制能够使旧缓存正确失效。
6.3 统一资源与时间语义
必须明确：
哪个组件负责推进仿真时间。
哪个组件负责 GPU 资源状态。
哪个组件负责网络资源。
哪个组件负责显存分配和释放。
哪个组件负责计算通信重叠。
哪个组件负责请求及模型状态更新。
同一计算、通信或等待事件不得重复计费。
如果 RoofLang 已经计算某条通信路径的完整耗时，Simthesizer 不应再次对同一通信过程计时。
对于跨任务资源竞争，应建立一致的资源占用及释放语义。
7. Agent 自主优化与演进机制
7.1 Agent 的自主权
Agent 不应仅作为配置搜索器或代码生成器。
Agent 应具备以下自主决策能力：
自主决定分析哪些性能指标。
自主识别当前最主要的性能瓶颈。
自主判断瓶颈属于硬件、算子、通信、缓存、调度还是系统架构。
自主生成多个具有技术依据的优化假设。
自主决定使用已有机制还是扩展新机制。
自主决定需要修改哪些组件。
自主安排组件扩展顺序。
自主制定实验和验证方案。
自主选择保留、淘汰、组合或继续改进的候选。
自主决定何时扩大设计空间或结束当前方向。
不要人为规定 Agent 必须优化 DCP、EP、PD 或 AF。
这些机制只能作为已有知识和候选能力，而不是强制优化目标。
7.2 开放式机制搜索
Agent 的搜索空间必须允许动态扩展。
可参考以下不同层次：
参数级搜索：调整已有机制的参数。
组合级搜索：重新组合已有组件和机制。
策略级搜索：修改某个组件内部的执行策略。
机制级搜索：增加当前系统不存在的机制。
架构级搜索：改变跨组件、跨层次的资源组织及执行方式。
Agent 可以自主决定当前应该在哪个层次进行探索。
允许提出当前 vLLM、SGLang、TensorRT-LLM 或其他真实框架尚未实现的机制。
不得因为真实框架暂时不支持某个机制，就直接排除该候选。
但新机制必须满足模型数学语义、硬件物理约束和明确的执行可行性条件。
7.3 完整自主优化闭环
每次自主研究运行应包含以下逻辑：
Baseline Simulation
→ Dense Feedback
→ Bottleneck Diagnosis
→ Hypothesis Generation
→ Mechanism Planning
→ Controlled Implementation
→ Independent Validation
→ Design Space Exploration
→ End-to-End Evaluation
→ Candidate Selection
→ Knowledge Update
→ Next Iteration
上述流程应由 Agent 和 Controller 自主组织。
允许根据实际研究情况跳过不必要的步骤、回退失败方案、并行探索多个候选或重新分析瓶颈。
一次运行应能够完成多轮优化，而不是每次修改后都等待人工指定下一步。
8. 组件级受控扩展
这是本项目的核心设计要求之一。
8.1 避免无序修改
禁止 Agent 为实现某项优化而无计划地修改整个仿真系统。
Agent 必须理解：
当前有哪些组件。
每个组件负责什么。
组件提供哪些接口。
组件依赖哪些其他组件。
组件维护哪些状态。
修改后可能影响哪些已有机制。
应优先通过新增组件、注册新策略、扩展接口或组合已有机制完成创新。
尽可能保留原有机制，使其能够作为基线、消融实验对象和回退方案。
8.2 Mechanism Specification
每个新机制在实现前，应具有结构化的机制描述。
至少包含：
机制名称及唯一标识。
父机制或基础版本。
需要解决的瓶颈。
核心技术假设。
预期性能影响。
新增组件。
修改组件。
复用组件。
组件依赖关系。
状态及接口变化。
资源需求。
数学语义约束。
潜在失败模式。
实验与验证计划。
回退方式。
Agent 可以根据实际机制复杂度自主决定描述的详细程度。
8.3 组件依赖管理
建立可查询的组件注册和依赖关系管理机制。
支持 Agent 自动获取：
组件能力。
输入与输出契约。
依赖图。
可扩展位置。
已有测试。
已知限制。
兼容性约束。
当 Agent 发现现有组件接口不足以表达新机制时，允许自主提出接口扩展或引擎级修改。
但必须明确影响范围，并对所有受影响的组件执行相应回归验证。
不要为了限制 Agent 而将系统扩展空间完全固化。
8.4 候选隔离与版本管理
不同候选机制应具有独立的实验环境和版本记录。
建议采用 Git 分支、Worktree、独立插件目录或其他可复现的隔离机制。
必须支持：
并行候选探索。
失败回退。
基线保留。
候选比较。
机制组合。
消融实验。
历史结果复现。
未经验证的候选不得直接覆盖稳定基线。
9. Dense Feedback 与自动瓶颈分析
仿真器不能只输出最终 Tokens/s。
需要提供足够丰富的结构化性能反馈，使 Agent 能够理解性能形成的原因。
9.1 算子级反馈
至少包括：
算子类型。
Tensor Shape。
FLOPs。
权重读取量。
激活读取和写入量。
KV Cache 访问量。
Recurrent State 访问量。
计算耗时。
访存耗时。
Kernel 启动及其他固定开销。
算术强度。
资源利用率。
主要限制因素。
9.2 层级与 GPU 级反馈
至少包括：
每层执行时间。
每张 GPU 的计算、访存和通信开销。
不同 GPU 的负载不均衡。
GPU 空闲与等待时间。
专家路由分布。
专家计算负载。
显存使用及生命周期。
计算通信重叠程度。
关键路径。
9.3 通信反馈
至少包括：
通信原语。
源设备与目标设备。
消息大小。
通信频率。
节点内和节点间流量。
链路利用率。
通信等待。
Collective 同步。
通信缓冲区。
通信与计算的资源竞争。
9.4 Serving 系统级反馈
至少包括：
请求排队时间。
TTFT。
TPOT / ITL。
End-to-End Latency。
请求完成时间。
Batch Size 分布。
GPU 利用率。
KV Cache 命中率。
缓存驱逐与抢占。
Prefill / Decode 资源利用率。
P/D 或 A/F 负载不均衡。
KV 与模型状态迁移开销。
Agent Session 完成时间。
SLO 违约原因。
9.5 失败反馈
对于不可行或性能下降的候选，必须记录具体原因，而不是只返回失败。
例如：
显存不足。
并行组不合法。
硬件不支持相关计算精度。
Tensor Layout 不一致。
KV 状态错误。
资源超额分配。
通信依赖错误。
死锁。
计算量增加。
通信开销超过预期。
资源争用导致重叠收益下降。
调度策略导致尾部延迟恶化。
Agent 应利用这些失败信息调整下一轮研究方向，避免重复探索已经证明无效的候选。
10. 仿真可信度与独立验证
由于本项目没有真实系统，必须将仿真可信度作为最高优先级之一。
10.1 防止 Agent 优化仿真误差
优化 Agent 不得通过以下方式获得虚假性能收益：
删除必要的计算。
忽略必要的通信。
放松显存约束。
隐藏额外的数据搬运。
高估硬件峰值性能。
使用硬件不支持的算子。
忽略状态迁移。
假设不可能实现的完全重叠。
修改吞吐计算方式。
修改评估目标。
删除失败测试。
修改基准负载以降低评估难度。
应将性能优化逻辑与最终评估逻辑进行隔离。
10.2 成本模型扩展
RoofLang 是唯一底层性能预测后端。
但允许 Agent 发现并提出成本模型缺失或不合理的问题。
成本模型修改应与普通机制优化分离，具有独立的变更记录和验证流程。
如果成本模型发生变化，应使用新模型重新评估原始基线及相关候选，确保公平比较。
不能只对新机制使用更有利的性能模型。
10.3 数学语义验证
对于新的计算图、并行机制、通信组织和 Tensor Layout，应建立小规模可执行 Reference。
可以使用 CPU Tensor 计算或其他低成本方式检查：
输出等价性。
Attention 计算语义。
MoE 路由语义。
Tensor 切分与重组。
KV Cache 一致性。
Recurrent State 更新。
跨层依赖。
不得仅凭 Agent 对机制的文字解释判断正确性。
10.4 物理资源验证
所有候选必须满足：
计算量守恒。
必要数据访问完整。
通信数据传输完整。
显存容量约束。
GPU 数量约束。
网络拓扑约束。
硬件精度支持约束。
资源占用与释放一致性。
合法的事件依赖。
无死锁及非法状态转换。
10.5 公开数据校准
虽然没有真实 GPU 集群，但应主动收集可追溯的公开数据：
官方 GPU 参数。
公开 Kernel Benchmark。
Kimi-K3 Serving Benchmark。
H200 与 B300 性能数据。
vLLM、SGLang、Dynamo 的公开实验。
已发表论文中的性能数据。
公开 Profiling 和 Trace。
利用这些数据校准已有机制的 RoofLang 性能模型。
明确区分：
官方规格。
公开实测数据。
解析推导。
校准参数。
未验证假设。
不得编造性能数据或将仿真结果标记为真实实测。
10.6 敏感性与稳健性分析
对于当前真实框架尚未支持的新机制，无法直接获得真实性能验证。
因此应在合理的参数不确定性范围内分析：
算子效率。
有效 HBM 带宽。
有效网络带宽。
通信启动延迟。
Kernel 启动开销。
计算通信重叠效率。
资源争用。
额外缓冲区。
状态迁移成本。
软件调度开销。
评估新机制是否只在过于理想的假设下具有收益。
优先保留在多种合理硬件和成本假设下仍具有优势的候选。
最终结论必须明确区分理论性能上界、仿真预测收益和经过公开数据校准的预测结果。
11. 优化目标与评价指标
不要仅使用单一 Tokens/s/GPU 作为优化目标。
根据具体工作负载和研究任务，建立多目标评价体系。
主要指标包括：
Tokens/s/GPU。
Tokens/s/User。
SLO Goodput。
TTFT P50/P99。
TPOT P50/P99。
End-to-End Latency。
Agent Job Completion Time。
GPU 利用率。
显存利用率。
网络利用率。
能耗及资源成本（在具有可信模型时）。
同时考虑：
机制复杂度。
硬件可实现性。
模型预测不确定性。
跨负载适用性。
跨硬件适用性。
使用 Pareto Frontier 保存不同目标之间的最优候选。
在相同模型语义、硬件预算、工作负载、SLO 和评估模型下比较新旧机制。
Agent 应尝试推动已有机制的 Pareto 前沿，而不是只在预定义设计空间中寻找最优配置。
12. H200 与 B300 跨硬件架构探索
本项目使用 H200 和 B300，不只是为了比较两代 GPU 的性能。
更重要的是研究：
当硬件资源比例发生变化时，最优推理机制及系统架构是否随之改变。
需要支持以下研究流程：
在 H200 上建立初始基线并执行自主优化。
在 B300 上独立执行自主优化。
将 H200 上发现的有效机制迁移到 B300。
将 B300 上发现的有效机制迁移到 H200。
分析机制收益变化及其原因。
判断哪些机制具有跨硬件通用性，哪些机制依赖特定硬件能力。
分析显存容量、HBM 带宽、计算能力、低精度能力和网络拓扑分别如何影响架构选择。
通过参数化未来硬件配置，预测下一阶段推理系统可能的架构演变。
不得提前假设某种机制一定适合 H200 或 B300。
相关判断必须由仿真结果、瓶颈分析和敏感性实验支持。
13. 研究范围与机制创新空间
以下方向可以作为知识库中的参考，但不是预定义的优化任务：
非均匀 TP、DP、EP、CP、PP。
KDA 与 MLA 的独立并行策略。
新型 KV Cache 与 Recurrent State 放置。
动态专家放置与专家负载均衡。
新型 DCP 数据切分和通信组织。
跨层 Tensor Layout 转换。
新型 Attention 与 FFN 解耦。
动态 PD / AF 资源分配。
多阶段或混合式推理架构。
非均匀 GPU 资源映射。
权重共享、预取与动态搬运。
计算通信流水化。
GPU 内部资源分配。
网络拓扑感知调度。
Agent 多轮任务的状态管理。
工作负载感知的架构重配置。
异构硬件及扩展内存组织。
Agent 应根据实际瓶颈决定是否探索这些方向，也允许自主提出列表之外的新机制。
不应将研究空间限制为上述机制的简单组合。
14. 研发流程
Phase 0：现有仓库审计与技术可行性分析
首先完整分析 Simthesizer 和 RoofLang 的实际代码。
输出：
两个仓库的核心架构。
主要执行流程。
组件及接口清单。
现有机制支持情况。
Kimi-K3 支持情况。
H200/B300 支持情况。
算子性能模型覆盖情况。
两个框架的能力重叠。
关键集成难点。
可直接复用的模块。
必须扩展的模块。
现有测试和验证能力。
推荐技术路线及替代方案。
不得在没有完成基本源码分析的情况下直接开始大规模重构。
Phase 1：基础性能模型与硬件基线
完成 Kimi-K3 在 H200/B300 上的 RoofLang 基础性能预测。
验证：
模型结构。
Prefill / Decode。
各类算子。
混合 Attention。
MoE。
精度支持。
显存占用。
计算通信成本。
并行策略。
GPU 拓扑。
Phase 2：Simthesizer 与 RoofLang 集成
建立统一执行契约，完成动态 Serving 仿真。
优先实现最小可运行闭环，并逐步支持更复杂的工作负载和系统机制。
Phase 3：Dense Feedback 与自主配置搜索
实现结构化性能反馈、自动瓶颈分析和 Agent 自主配置探索。
验证 Agent 是否能够根据实际性能瓶颈自主决定优化方向。
Phase 4：受控机制扩展
实现组件注册、依赖分析、机制描述、候选隔离和独立验证。
使 Agent 能够自主增加新的策略和系统机制，而不是只调整已有参数。
Phase 5：开放式跨层架构探索
允许 Agent 同时扩展 Serving DAG、底层计算通信图、Tensor 放置及资源组织。
验证新机制是否能够在相同约束下推动已有系统的 Pareto 前沿。
Phase 6：跨硬件预测与持续自主演进
完成 H200/B300 独立优化、机制迁移、敏感性分析及未来硬件参数探索。
建立可持续积累机制知识和失败经验的长期自主研究能力。
上述阶段是建议的研发路线。
你可以根据实际仓库能力、实现成本和研究收益调整顺序，但必须说明原因。
15. 实验与研究评价要求
至少设计以下对照实验：
固定已有机制与人工配置。
传统参数级设计空间搜索。
Agent 自主参数搜索。
Agent 自主机制组合。
Agent 自主新增机制。
Agent 跨层架构探索。
不同组件控制策略的消融实验。
不同反馈粒度的消融实验。
H200 与 B300 的独立优化及机制迁移。
不同性能模型假设下的稳健性实验。
比较维度包括：
最终预测性能。
Pareto Frontier。
搜索效率。
候选数量。
无效候选比例。
正确性验证通过率。
机制扩展成本。
代码修改范围。
回归测试结果。
仿真运行开销。
Agent Token 与时间成本。
跨硬件迁移能力。
需要证明新增机制搜索能力相对于固定设计空间搜索的实际价值，而不是仅证明 Agent 能够生成代码。
16. 自主执行与关键决策策略
你应当主动推进项目，而不是在每一个普通开发步骤中等待人工确认。
对于以下事项，可以在当前项目授权范围内自主决策并执行：
阅读和分析代码。
调研相关论文及公开仓库。
制定开发计划。
设计模块和接口。
创建实验分支。
编写和修改项目代码。
增加测试。
执行仿真实验。
分析性能瓶颈。
修复普通问题。
扩展候选机制。
回退失败方案。
调整非关键实现细节。
只有遇到以下重大事项时，才需要向用户说明并请求决策：
核心研究目标发生实质性变化。
需要放弃主要技术路线或替换核心框架。
需要不可逆地破坏已有稳定功能或实验数据。
需要修改最终评估标准或关键物理约束。
需要超出当前项目授权范围访问或修改外部资源。
需要显著增加预算、计算资源或其他外部成本。
存在无法通过合理技术分析解决的关键不确定性。
不要因为普通实现细节存在多种选择而频繁中断工作。
在合理授权范围内，优先自主分析、比较方案、做出决策并继续执行。
17. 交付物要求
整个项目应持续形成可复现、可检查的研发成果。
至少包括：
架构文档
系统总体架构。
组件职责。
数据流和控制流。
执行契约。
组件依赖关系。
Agent 自主优化流程。
机制扩展流程。
独立验证流程。
代码与配置
可运行的集成系统。
Kimi-K3 模型描述。
H200/B300 硬件描述。
RoofLang Adapter。
Agent Controller。
Mechanism Registry。
Dense Feedback。
验证与测试模块。
可复现实验配置。
实验结果
原始基线。
每轮优化的候选机制。
瓶颈变化。
性能变化。
资源变化。
失败原因。
Pareto Frontier。
跨硬件比较。
敏感性分析。
自主研究报告
每轮研究至少记录：
为什么选择该优化方向。
当前瓶颈是什么。
提出了什么新机制。
修改了哪些组件。
为什么需要这些修改。
如何验证正确性。
获得了什么预测结果。
结果依赖哪些假设。
是否值得保留。
下一轮应该研究什么。
所有结果必须能够追溯到具体代码版本、模型配置、硬件配置、工作负载及性能模型版本。
18. 最终研究原则
本项目的核心不是简单地集成 Simthesizer 与 RoofLang，也不是构建另一个固定功能的 LLM Serving Simulator。
真正的研究目标是：
构建一个具备开放式机制空间、组件级受控演进、跨层架构探索和独立可信评估能力的自主研究系统。
Agent 应当能够突破当前 Serving 框架的机制限制，但不能突破模型数学语义和硬件物理约束。
系统应当支持不断产生新的研究假设、新的推理机制和新的架构候选，并通过仿真判断这些候选是否具有值得进一步研究的价值。
不要将仿真预测收益直接表述为真实系统性能提升。
不要为了追求表面上的自主性而牺牲系统正确性、可维护性和实验可信度。
19. 立即开始执行
现在开始 Phase 0。
首先完整阅读当前项目中 Simthesizer 和 RoofLang 的源码、README、配置文件、测试、已有 Agent Skills、性能模型及实验结果。
结合本项目研究目标，完成深入的技术可行性分析。
重点回答：
两个框架目前的实际能力分别是什么？
两者应该在哪个抽象层次连接？
RoofLang 能否作为 Simthesizer 唯一的算子性能预测后端？
Kimi-K3、H200 和 B300 的现有支持与能力缺口是什么？
哪些已有模块可以直接复用？
哪些模块需要扩展或重新设计？
如何实现 Agent 自主扩展组件而不破坏原有系统？
如何建立统一的性能反馈和独立验证机制？
如何支持当前真实 Serving 框架尚未实现的新机制？
如何在没有真实系统的情况下保证预测结果具有可信度？
最小可行系统应该是什么？
后续应该如何逐步发展为完整的自主架构探索平台？
请根据实际源码证据提出架构方案，比较可选技术路线，明确推荐方案及其理由，并形成可执行的研发计划。
不要直接照搬本提示词中的建议架构。
在完成基本架构分析后，按照研发计划自主推进低风险、可逆的开发与验证工作，持续记录关键决策和实验结果，直到完成当前阶段目标或遇到确实需要人工决策的重大问题。