训练方案（Training-While-Reasoning 全链路）
0. 总架构
目标系统 = 搜索器（MCTS）+ 验证器（Lean kernel）+ 可训练策略与价值网络 + 在线训练回路。训练与推理不是两回事：MCTS 展开出的每一个节点（state, tactic, verdict）本身就是监督信号——这就是 "training-while-reasoning"：测试时产生的推理轨迹被立刻回灌成训练样本。整套机制：

$$\pi_\theta, V_\phi \ \xrightarrow{\text{rollout}}\ \text{MCTS} \ \xrightarrow{\text{tactic}} \ \text{Lean checkProof} \ \xrightarrow{\text{reward}} \ \text{std loss（GRPO/TTT）}$$

1. 四个组件与职责
组件
Policy $\pi_\theta(a\mid s,\mathrm{ctx})$
Value $V_\phi(s)$
Search $\mathrm{MCTS}(s;\pi,V)$
Verifier $\mathcal{V}(s,a)\in{\mathrm{parse,forbidden,timeout,error,ok,solved}}$
2. Reward 信号（Lean Feedback）
最终奖励只在验证通过时为正（防奖励黑客）：

$$R = \begin{cases} +1 & \exists \text{ kernel 通过的回放脚本} \\ 0 & \text{其他终点} \\ -\lambda_s & \text{每步惩罚} \end{cases}$$

中间辅助 shaping：子目标进度（goal 尺寸对数差），供 $V_\phi$ 与 advantage 计算用，不进最终奖励。
3. 参数更新三步走（每条路径都"边推理边训练"）
第 1 步：SFT（冷启动，不包含在"边推理"内）
以 FrenzyMath/state_tactic_pairs（50k）+ 自有数学库语料，用标准 SFT 损失：

$$\mathcal{L}_{\mathrm{SFT}}=-\mathbb{E}\log\pi_\theta(a\mid s,\mathrm{ctx})$$

第 2 步：RL（GRPO 为主，MCTS 回滚作群体）
对同一 prompt 采样 $G$ 条 MCTS 轨迹，组内相对优势：

$$\hat A=\frac{r_g-\text{mean}(r)}{\text{std}(r)},\quad \mathcal{L}_{\mathrm{GRPO}}=-\frac{1}{|G|}\sum_g\mathrm{clip}\left(\rho_g\hat A\right)+ \beta\mathrm{KL}\big[\pi_\theta\|\pi_{\mathrm{ref}}\big]$$

值网络同轮用蒙特卡洛/λ-advantage 回归：$V_\phi \to \mathbb{E}[G_t]$。
第 3 步：Training-While-Reasoning（TTT，§09）
- 策略：单题搜索内，每次 API 调用的 verdict 即刻构成在线样本，做 1–16 步 LoRA 更新（$\theta \leftarrow \theta-\alpha\nabla[-\hat r \log\pi_\theta(a\mid s)]-\alpha\beta\nabla \mathrm{KL}[\pi\|\pi_{\mathrm{base}}]$）——反复出错 tactic 被下压，验证通过的被拉升；
- 价值：TD 更新 $V_\phi(s)\leftarrow V_\phi(s)+\alpha_V[\hat r+\gamma V_\phi(s')-V_\phi(s)]$，直接使用本轮搜索的折扣备份；
- 纪律：每题的 LoRA 适配器用完即弃/入库，eval 题必须从基准权重（冻结策略）跑——防 bench leak。
4. 迭代环路（一条完整"训练-推理"回合）
启动 788 模板（SSH 直连）→ 环境就绪（torch-RoCM + vLLM/llama.cpp）
 → 数据下拉（ModelScope 镜像）→ SFT 0.8–1.5B（LoRA）
 → 部署 policy server（logprobs 支持）+ 值头服务
 → reap 批量 solver：FATE-M/演进题池 → 每道题 MCTS（SFT 策略）
 → 每校验成功一条 proof：
       (a) 加入候选库（growing mathlib）
       (b)  一条轨道：离线 GRPO 回灌给 π_θ
       (c)  另一条轨道：本题内 TTT 梯度（在线）
 → 每 3-5 题后 heat: 重新标定 Diff → 课程阶梯右移
 → 每轮结束：评估门（固定 30 题 solve@B 曲线）→ 达阈值则进行下一代
5. 本机硬件适配（你刚换到的新话题后端）
- W7900 48GB（gfx1100 / RoCM 6.x）：3B 全参微调 OK，1.5B 有富余（可加 512+ batch、FSDP no-OOM）；
- RoCM 注意：vLLM 官方 ROCm 工位只保证 Instinct 系（MI300 etc.）；消费级 gfx1100 建议用 llama.cpp + 自写 logprobs 服务端，或社区 vllm-hip 构建（vram 48GB 时用 vLLM 尝试也受限；llama.cpp HIP 构建对 gfx1100 成熟）；
- 模型下载：平台在中国华东，HuggingFace 直连通常被墙——一切模型/数据走 ModelScope（Qwen 系在魔搭全有镜像），这个 788 模板本身就是 Qwen 微调教程，正好使用其环境。
6. 下一步（我可以现在就做）
1. 你在平台启动 HuggingFace(788) 模板（ssh:true，资源 16c55g1u）；
2. 实例起来后给我详情页 SSH 地址/端口（或我通过浏览器自动截取）；
3. 我从 WSL ssh root@host -p port（公钥已写入你的 Profile）——后续全部终端级操作，无需再碰浏览器；
4. 第一跑：Qwen3.5-0.8B 魔搭 SFT（作为 π_θ 基线）+ 部署 llama.cpp logprobs 服务 → 第一个 reap 批量 solve@B 基线。
要不要我等你启动完 788，就抓 SSH 地址？
/agents      ▣  Build · DeepSeek V4 Flash Vision Exp · 23.9s
/compact     