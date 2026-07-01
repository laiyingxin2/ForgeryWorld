# 决策：训练时机 + 两处"不如文章"的细节补强

> 写于 2026-06-22，回答两个明确问题：
> **(Q1)** 应该"先不接训练、先攒 skill/自动化/数据，再训练"，还是"一开始就接上训练"？
> **(Q2)** "现在好像有的还不如文章"——具体不如在哪、怎么补。
>
> 本文不重写设计（设计已落在 `DESIGN_V3.md` / `THREE_METHODS.md` / `METHODS_RECORD.md`，且 M1–M5 已实现并跑出结果）。本文只做一件设计文档没做的事：**用你自己 M1–M5 的实测证据，把"何时接训练"这个 deep-thinking 题答死，并定位两处可精确补强的实现细节。**

---

## 0. 一句话结论

**Defer——分三阶段，但从第一天就把数据 instrument 成"可训练"格式。**
不是"先不训练"，也不是"一开始就训练"，而是：

| 阶段 | 谁在动 | 谁冻结 | 用什么学 | 进入条件（gate） |
|---|---|---|---|---|
| **P0 攒 skill/数据** | 攻击者（in-context：skill / MCTS / seed_library / 归因） | 检测器冻结 | 无梯度，纯搜索+技能累积 | 现在就在这里（= M2） |
| **P1 检测器飞轮** | 检测器（监督 SFT/LoRA） | 攻击者仍 in-context | 监督学习（稳、低风险） | P0 攒够 ≥N 张"过 face_valid 的 bypass 图" + judge 可信 |
| **P2 攻击者 RL** | 攻击者策略（GRPO/ADCA） | 检测器按 TTUR 慢动 | 梯度（最高风险，最后上） | judge 可信 + 数据多样 + P2 的数据格式已就绪 |

下面解释**为什么是这个顺序**，而且**你已经用实验证明了它**。

---

## 1. 你自己的实验已经把答案写出来了

这不是纸上推演。仓库里 M2 vs M3/M5 就是"不接训练"vs"一开始接训练"的对照实验，结论非常干净（`METHODS_RECORD.md` 第二部分实测）：

- **一开始就接训练 + 强检测器 = 崩**。
  M3（`outputs/m3v2/`、`m3_coevolution/`）和 M5（`outputs/m5_coevolution/`，强 SCUT 检测器）：每轮 bypass≈0，`det_bypass_vs_D_R` 全 null，coverage 卡在 2 个 **no-op** 家族，M5 三轮 2.4%→0%→0% 直接退化。原因写在你自己的记录里：**capability-gap 地板**——API-only 攻击者打不动一个已经 95% bal-acc 的检测器，一上来就把检测器再训强，攻击者更没立足点。

- **唯一跑出干净军备竞赛曲线的，是 weak-start**（W1/W2，`outputs/weakstart/`）：
  检测器换成 vanilla llava（弱起点），8 轮才有 92→96→48→…→4% 这种漂亮压制。**但你自己也标了硬伤**：这条曲线衡量的是"防御方变强"，不是"攻击者发现新手法"，而且对手不是 M1/M2 的同一个检测器。

- **唯一能解释"攻击者在发现方法"的信号，来自 M2**（冻结检测器、纯 skill 进化）：clean 口径 valid_best 命中真·有效家族（frontal_swap/id_diff）。代价是弱、单 seed、三跑互相矛盾（每轮 attempts 仅 6–10，方差>信号）。

**把三件事并起来读**：

> 检测器一旦先变强，攻击者（不可微调的 API 模型）就被地板压死；
> 攻击者这一侧的"学习"在**冻结检测器 + 纯 in-context 技能进化**下才看得见信号；
> 而这个信号现在**太弱、太不稳**——弱在**数据量/多样性**，不在"少了一个训练 loop"。

所以结论不是"赶紧接训练补强"，恰恰相反：**先把 P0 攒厚（更多 seed、更多 face-valid 样本、降方差），训练才有意义。一上来接训练只会把唯一可解释的信号也淹掉。**

---

## 2. 第一性原理：为什么 defer 在这个任务里几乎是必然

1. **奖励不可信就别优化它。** RL 是"把奖励压榨到极限"的机器。你的奖励 = FakeVLM 判定。而现在：本地 FakeVLM 强 ckpt 走 :8001 代理、弱起点走 vanilla、真权重一度是 0 字节占位（已知坑）。judge 一抖，GRPO 学到的是**钻 judge 漏洞**（reward hacking），不是真伪造。**先把 judge 钉死、把 face_valid 闸门修好，再谈梯度。**

2. **同时训两边 = 非平稳目标。** 攻防都从头一起更新，是移动靶打移动靶，极难调试。标准做法：冻一边、进化另一边，再交替。你的 M2（冻检测器）/ M3（TTUR 节流单训检测器）已经是这个纪律——别破坏它。

3. **训练吃的是 curriculum，而 curriculum 正是 P0 的产物。** P2 的 GRPO 需要一批"已验证的难样本 + 逐步信用标注"。你的 skill 累积、seed_library、self_attributor 归因，**本身就是在生产这批训练数据**。先训 = 在冷启动的低多样性数据上训。

4. **成本与可逆性。** in-context 进化便宜、快、可解释、可回滚（删 `outputs/skills_v2/` 即重来）；LoRA 训一轮要 GPU+1–6h 且引入遗忘风险（你已用 STaSC non-decreasing guard + real-acc floor 0.80 兜底，说明这风险真实存在）。**先用便宜手段把"哪些链/技能真有效"探明，再花贵的梯度去固化。**

5. **当前真瓶颈不是"缺训练"。** `METHODS_RECORD` 自己列的硬伤是：face_valid 通过率仅 ~10%、没对标亚七生产基线、挖掘是闭集 taxonomy。**这四个没一个靠"早点接训练"能解决**，反而都要 P0 把生成质量和数据攒厚才动得了。

---

## 3. 但"defer"≠"忽视训练"——从第一天就 instrument

这是 AgentEvolver 给的最值钱一课：它的数据从产生时就按 ADCA-GRPO 的格式存（trajectory + 逐 step GOOD/BAD flag + advantage），所以训练那天**零改造直接喂**。

你已经有 `self_attributor.py`（逐 step GOOD/BAD + suffix-sum advantage）和 `data_flow.py`（V=1→SFT/V=0→CT）。**P0 阶段唯一要补的纪律**：每条 rollout 落盘时，把 `(family, cluster, op_chain, per-step GOOD/BAD, r_out, advantage, face_valid, det_conf, novelty)` 完整写进 `data_flow_v2.db`。这样 P2 接 GRPO 时，attacker SFT/RL 池是现成的——`lv5_connector.py` 已经在做这件事（`exec_steps[-1].output_path` 作 SFT 图、composite_reward = 0.7·bypass+0.3·skill_GOOD），保持即可。

**P1 检测器飞轮可以早于 P2 启动**：它是监督学习、相对平稳、直接提升 benchmark 难度，风险远低于 attacker RL。你的 M3 `train_defender_round.py`（answer-only loss、real-acc floor、bypass-floor、TTUR）已经是合格的 P1 实现。**顺序就是：P1（训检测器）先于 P2（训攻击者）。**

---

## 4. 两处"不如文章"的细节——已定位 + 给出可直接抄的准确公式

你担心"很多细节方面不如文章"。我把参考代码 git 下来逐行读了（`ref_repos/GPTFuzz`、`external/AgentEvolver`），定位到**两处确实可以更精确**的实现，且都给出 verbatim 公式与改哪一行。

### 4.1 MCTS：现在是裸 UCB1，文章/GPTFuzzer 是 MCTS-Explore（带层惩罚的回传）

- **现状**：`src/simple_baseline.py:175` 的 `SimpleMCTSPlanner` 是 **UCB1 4-step**——选点用标准 UCB1，但**没有 best-leaf 路径下降、没有按节点深度折扣的奖励回传、没有 α 早停**。这正是"不如文章"的地方：GPTFuzzer 的种子选择不是 UCB1，是 MCTS-Explore。
- **GPTFuzzer 真实实现**（`ref_repos/GPTFuzz/gptfuzzer/fuzzer/selection.py:79-134`，`ratio=0.5, alpha=0.1, beta=0.2`）：

  选点（从 initial 节点起，沿子节点贪心下降，α 概率随机早停）：
  ```
  score(pn) = rewards[pn] / (visited(pn) + 1)
            + ratio * sqrt( 2 * ln(step) / (visited(pn) + 0.01) )
  ```
  奖励回传（沿被选路径，按**叶子**层数折扣，floor=β）：
  ```
  reward = succ_num / ( |questions| * |mutants| )
  for node in selected_path:
      rewards[node] += reward * max(beta, 1 - 0.1 * leaf.level)
  ```
  且**变异体只有 num_jailbreak>0 才入种子池**（成功才繁殖，建出 MCTS 树）。

- **怎么改**：把 `SimpleMCTSPlanner` 的选点换成上式（注意指数项分母用 `+0.01`、利用项用 `+1` 的非对称平滑），加一条"成功链才 promote 进 seed_library"的门（你 seed_library 已有 promote，接上即可）。这条改完，M1 的"MCTS"才名副其实，也直接给 M2 的 `pipeline_planner` 一个更强的搜索骨架。

### 4.2 信用分配：现在是单流 `α·prm + r_out`，AgentEvolver 推荐 decouple（组内独立 z-score）

- **现状**：`src/self_attributor.py:169` 的 `composite_reward` 做的是 `composite = α·prm + r_out`（α=0.1），advantage = suffix-sum——这是 AgentEvolver 里**较弱的 allocation 方案**，而且只在单条轨迹内归一。
- **AgentEvolver 推荐的 decouple 方案**（`external/AgentEvolver/.../adv_processor/adca_grpo.py::_build_decouple`，`fix_base=0.2, alpha=0.1, orm_distribution="last_step"`）：
  1. GOOD/BAD → ±`fix_base`（±0.2）；
  2. **在 GRPO group 内**对 PRM 步奖励和 ORM 结果奖励**各自独立 z-score**（`_group_zscore_on_steps`）；
  3. 融合：`combined = alpha * prm_zscored + orm_zscored`（ORM 只加在最后一步）；
  4. `suffix_sum` 得每步 advantage，再 broadcast 到 token。
- **怎么改**：等你进 P2、有了 GRPO group（同一 brief 的 G=8 rollout）时，把 `composite_reward` 从"单轨 α·prm+r_out"升级为"组内独立 z-score 后再融合"。**P0/P1 不用改**（单轨足够），这是 P2 上 RL 时才生效的精度升级——现在先把接口留好。

> 备注：DARWIN（`external/DARWIN`）是 Markov+Q-learning 的**策略池**路线（γ=0.5, α=0.1, 语义去重 cos≥0.95, sandbox ASR≥0.40 准入），和 GPTFuzzer 的 MCTS-over-template-tree 是**两条不同的线**。你的 MAJIC（`markov_family.py`）已对应 DARWIN 这条；MCTS 这条要补的是 §4.1。两条不冲突，分别管"选哪个家族"和"在家族内搜哪条链"。

---

## 5. 不要做什么（避免过度工程）

- **不要再加层。** 已经 10 层 + 5 方法 + 3 改进模块。`METHODS_RECORD` 的硬伤没有一个是"少一层"造成的。
- **不要为了"接训练"而接训练。** P2 的 gate 没满足（judge 可信 + 数据多样 + face_valid 闸门修好）之前上 GRPO，只会 reward-hack。
- **真正该投的三件事**（都属 P0，按性价比排序）：
  1. **把 face_valid 从 ~10% 抬上去**——这是"样本质量需人工保障"痛点的正面战场，也是所有 bypass 数字的分母。
  2. **降 M2 方差**——单 seed 三跑矛盾是因为每轮 attempts 6–10；先把 briefs×rollouts 加厚到信号>方差，再谈任何对比。
  3. **补 §4.1 的 MCTS-Explore**——唯一一处"实现确实弱于文章"的硬伤，且改动小、收益直接落到 M1/M2 的搜索质量。

---

## 6. TL;DR 给你

1. **Defer，分三阶段**（P0 攒数据/技能 → P1 训检测器飞轮 → P2 训攻击者 RL），但 P0 就把数据存成可训练格式。
2. **你的 M2 vs M3/M5 已经证明**：强检测器上一开始就训会被 capability-gap 压死；攻击者侧的学习只有在冻结检测器 + 厚数据下才看得见——所以**先攒后训**。
3. **两处"不如文章"已定位**：MCTS（裸 UCB1 → GPTFuzzer MCTS-Explore，§4.1，可直接抄公式）、信用分配（单流 → AgentEvolver decouple，§4.2，P2 再上）。
4. **现在最该干的不是接训练**，是把 face_valid 通过率和 M2 方差这两个 P0 瓶颈解决——它们才是 benchmark "非常好"的前提。
