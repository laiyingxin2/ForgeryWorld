# 方法 M1–M5 详细记录（机制级，逐步说清"具体怎么做"）

> 项目：自进化人脸伪造红队（AI 驱动攻防自进化 Loop）
> 记录日期：2026-06-22　｜　机制据 `src/` 源码核实，数字据 `outputs/` 实测
> 攻击者：API-only 大模型（Gemini-2.5-flash / gemini-3-pro via viviai 网关，**不可微调**）
> 检测器：FakeVLM(LLaVA-1.5-7B)；强基线=校正 ckpt(:8001, bal-acc 95%)；弱起点=vanilla llava(:8006/:8007)

---

## 第一部分：所有方法共用的攻击者 pipeline（一次 rollout 内部到底发生了什么）

M1/M2/M4 的"出招"都走同一条流水线，差别只在**用哪套记忆/规划器、轮末怎么进化**。先把这条共用链讲透（代码：`src/orchestrator.py:run_one_rollout`、`src/multi_agent_gen.py`、`src/sandbox.py`、`src/operators/`）。

**一次 rollout = 一次"选家族 → 出 brief → 质检 → 规划算子链 → 真生成伪造图 → 检测器判定 → 归因 → 写回记忆"**：

1. **Layer 1 选攻击家族**（`markov_family.py`）
   用 Markov 选择器从 9 个家族里挑下一个：VALID={frontal_swap, profile_swap, id_diff, morph, adv_patch}，no-op={audio_synth, reenact, 3d_mask, replay}。家族被选后，其 bypass/novelty 结果回写更新 Markov 转移概率。

2. **Layer 3 取技能/经验**（v1 与 v2 在此分叉，见第二部分）
   取出该家族的 `skill_doc`（攻击 SOP playbook）+ top-k 历史经验。v2 还会把 reasoning_bank 的 strategy-rule、VideoWeaver 的最佳链形、seed_library 的"已验证成功链"拼进 skill_doc。

3. **seed_library 强制起步（F2 关键）**：若该家族有历史成功链，给 setter 一条硬指令"**必须从 top 链起步、最多变异 2 个算子**，不要凭空发明新链"——这是 exploit 已学到的赢法的机制。

4. **Layer 2 出题组（2 个 setter LLM）**（`multi_agent_gen.generate_brief`）
   setter_a(T=0.7) / setter_b(T=0.9)，gemini-2.5-flash。按家族 system prompt + skill_doc 产出**一条 brief（严格 JSON）**：
   `{attack_class, suggested_chain:[算子名…], params_hints, scene_description, rationale}`。
   即"用哪条算子链、什么参数、针对什么场景、为什么可能骗过检测器"。

5. **Layer 2 质检组（3 个 checker LLM，median 聚合）**（`check_brief`）
   checker_a/b/c(T=0.1/0.3/0.5)，对 brief 打 **4 维分（内部文章 verbatim）**：攻击成功潜力 / 覆盖新攻击面(novelty) / 泛化 / 防御绕过，取中位数 + 列 issues。

6. **Layer 4 规划算子链**（v1=MCTS，v2=pipeline_planner，见第二部分）
   把 brief 的 suggested_chain 落成可执行 `chain`（≤5 步），v2 还产出 lookahead 候选 + proxy 分。

7. **Layer 4 真生成伪造图**（`execute_pipeline` → `src/operators/`）
   **这一步才真正生成图**，调本地/或 API 算子：
   - 换脸：`inswapper_128_local`、`simswap_256_local`（frontal/profile_swap）
   - 形变：`stylegan_morph`（blend morph）、api `nano_banana`（id_diff/morph 高质量）
   - 对抗：`adv_patch_pgd`（对 FAS CNN 做 PGD）
   - 活体/时序：`reenact`（LivePortrait-lite）—— 对静态图检测器属 no-op
   - 后处理：`face_align`、`jpeg_70/85/95`、`gfpgan`（掩盖伪造痕迹）
   - **伪 bypass 防护**：若整条链没真生成新图（算子 503 全失败 → mock 透传 → final=src），标记 `PSEUDO_BYPASS_REJECTED`，判失败。

8. **Layer 5 沙箱判定**（`sandbox.verify(forged, src, family)`）
   - **tier1 = face_valid**：人脸/身份一致性检查（脸还在不在、是不是同一人、姿态合理）。
   - **tier2 = FakeVLM 检测器**：把伪造图喂检测器问 "real or fake?"，得 is_fake/confidence/reasoning。
   - **最终判定**：`sandbox_pass = detector_fooled AND face_valid`（`sandbox.py:688`）。
     → 这就是为什么"检测器级骗过率"≠"沙箱级 bypass"：每轮 ~47 张里只有 2–10 张过 face_valid 闸门。

9. **Layer 6 归因（仅 v2）**（`self_attributor.py`）
   逐算子归因"是哪一步导致 bypass/被抓"，算 composite_reward。

10. **轮内写回记忆（每个 rollout 后）**
    family_agents 经验、VideoWeaver 链形、novelty 记录、reasoning_bank 蒸馏 strategy-rule、data_flow 落盘；Markov 用 `0.7·bypass + 0.3·novelty` 更新。

11. **Layer 7/轮末**（`_end_of_round`）
    diagnose 各家族 bypass 率；v2 尝试把 ≥2 家族共享的模式提升为跨家族 meta-skill；落 JSON 快照。

> **一句话**：setter 出招 → 3 checker 质检 → planner 排算子链 → operators 真生成图 → FakeVLM+face_valid 判定 → 归因写回。**M1–M5 都在这条链上，区别是"记忆怎么进化"和"检测器动不动"。**

---

## 第二部分：逐方法——具体差异点

### M1 — 基线 port（无自学习参照系）

- **调用**：`orchestrator.py --mode v1 --rounds 1 --briefs 8 --rollouts 2 --multi-agent-preset w6_full --fakevlm-endpoint :8001`（外层 shell 跑 R0–R4）。
- **具体怎么做（与共用链的差异）**：
  1. **Layer 3 用 `skill_book_v1`**（简单 playbook），不走 ace_skill_lib 的经验检索。
  2. **Layer 4 用 MCTS planner**（`MUTATION_OPERATORS_V1`）：从 seed 链做蒙特卡洛树搜索 + 变异，**无 lookahead**。
  3. **唯一的"学习"**：每轮把 checker 的 issues 自动写成一条 constraint 追加进 v1 playbook（`append_constraint`），下轮 setter 看到 → 不重犯同一问题。**没有经验库、没有归因、没有 supervisor 战略更新。**
- **实测**（`outputs/p8_faithful/p8_20260621_1409`）：coverage_cum 0,1,2,2,2；best_so_far→20%；attempts=34。
- **问题**：raw 的 20% **全靠 no-op 家族**（clean valid_best=0%，见 `clean_coverage_eval.json`）；作参照足够，不必深究。

### M2 — 主自进化攻击者 vs 冻结检测器（核心方法）

- **调用**：同 M1，仅 `--mode v2`。检测器**冻结**（:8001 不训练）。
- **具体怎么做（v2 的完整自进化栈，这是 M2 的全部价值）**：
  1. **Layer 3 用 `ace_skill_lib.retrieve`**：取 skill_doc + top-k 经验（按 applicability 加权），并注入 reasoning_bank 的 strategy-rules、VideoWeaver 最佳链形、seed_library 已验证链。
  2. **Layer 4 用 `pipeline_planner.plan`**：带 lookahead 候选 + proxy 评分 + seed_library 高分回池（不是盲搜）。
  3. **Layer 6 归因**：self_attributor 给每步信用分 → composite reward。
  4. **轮内进化 `_update_skill_v2`**：
     - (a) **每个 rollout 必做（免费）**：`add_experience` 把本次结果（bypass + 检测器 reasoning）累加进该家族经验库 ℰ_k。
     - (b) **每 N 次（花钱）**：**supervisor（gemini-3-pro-preview）** 读最近 briefs+checker 分+攻击结果，抽一条 **战略 delta Δ𝒮_k 写回 playbook**（"什么 work、什么 fail、下轮建议、新增约束"）。
  5. **跨 run 持久化**：技能库/经验/Markov 跨轮跨 run 累积（`persist_skills_across_runs`）——这才是"自进化"能跨 5 轮积累的根。
  6. **失败纠正**：bypass=False 时 `ui_voyager_correct` 找一条 donor 成功轨迹，把失败链在失败点"嫁接"成功子链，promote 进 seed_library 供下轮复用。
- **实测（单 seed，三跑互相矛盾——这是硬伤）**：

  | 跑 | coverage_cum | best_so_far |
  |---|---|---|
  | p7_definitive | 0,0,1,1,1 | →33% |
  | p8_1409 | 1,2,2,2,2 | →10% |
  | p8_1840 clean | valid 1,1,1,1,2 | **valid_best=100% 全程** |

- **卖点**：clean 口径 valid_best=100% vs M1 20%（M2 命中真·有效家族 frontal_swap/id_diff）。

### M3 — 真共进化（攻防都更新）

- **核心机制（`src/method3_coevolution.py` + 驱动 `scripts/coevo/run_coevolution_v2.py`）**："挂载 + 一起更新"的闭环：
  1. **`start_round`**：attacker + defender 双侧 init，LoRA 始终 mounted in vLLM。
  2. **`run_one_batch(N)`**：跑 N 个 rollout（同共用链）打**当前** detector D_R。
  3. **收集**：把过了 face_valid 的 bypass 伪造图 → 攒成 SFT 训练数据（`build_round_data.py`：本轮 bypass + reals + 历史 replay 采样，swift 格式）。
  4. **`train_defender_round.py`**：在校正 FakeVLM base 上训一轮 LoRA（RAW-completion prompt `<image>Does the image looks real/fake?<answer>`，answer-only loss masking），得 D_{R+1}。
  5. **`reload_lora`**：热重载 vLLM。
  6. **`broadcast_to_attacker`（关键）**：把 detector 新状态（现在看哪些 cue、哪个家族最强/最弱、上轮哪些链被识破）push 进 attacker 下轮的 skill_doc → setter 立刻知道"检测器往哪变强了"。
- **v2 驱动的三个反塌缩杠杆**（防御方太强会瞬间压死 API-only 攻击者）：
  1. **THROTTLE** `--defender-period K`（TTUR）：每 K 轮才训一次，中间打冻结检测器，给慢玩家立足时间。
  2. **弱/两时间尺度更新** `--lora-r 8 --epochs 1 --lr 1e-5`：每次只走一小步、故意欠拟合最新攻击，留可利用余量。
  3. **BYPASS FLOOR** `--bypass-floor`（ACE residual-ASR）：拒绝把 bypass 压到地板以下的候选 D_{R+1}。
  - 叠加 STaSC Non-Decreasing guard + real-acc floor 0.80（防遗忘 + 防"全判 fake"偏执）。
- **两类结果**：
  - **强检测器（退化）**：`outputs/m3v2/`、`outputs/m3_coevolution/`——每轮 bypass≈0，`det_bypass_vs_D_R` 全 null（撞 capability-gap 地板，无 arms-race）。
  - **weak-start（W1/W2，目前唯一能直接用的结果）**：detector=vanilla llava，pool=scut_curated 60，8 轮：
    - **W1(K=1)** `outputs/weakstart/W1_clean_20260622_0607`：检测器级 bypass **92→96→48→27→21→6→4→10%**（8 轮共 383 次判定，real_acc 全程 1.0）→ 干净 MART/CHASE 式压制 ≈−95%。
    - **W2(K=2 节流)** `outputs/weakstart/W2_throttle_20260622_0607`：**88→100→100→91→96→22→23→16%**（共 395 次判定）→ TTUR 节流震荡 + R5 突破。
- **问题**：强检测器退化；漂亮曲线衡量的是**防御方变强**而非攻击者发现新手法；weak-start 与 M1/M2 不是同一个对手。

### M4 — 人脸类型条件化攻击者（Bayesian/Eevee-inspired 消融）

- **核心机制（`src/method4_orchestrator.py`，独立 orchestrator，不走 mode）**：在共用链前面插一个**按人脸类型分桶的技能路由**：
  1. **`face_metadata_router.extract_metadata(src)`**：把源人脸聚成 cluster_id，如 `male_adult_frontal`（性别×年龄×姿态）。
  2. **`pareto_skill_pool.get_top_snippet(family, cluster)`**：从该 (家族, 人脸类型) 桶里取 Pareto 最优技能片段当 skill_doc。
  3. → setter 出 brief → planner 排链 → execute 生成图 → sandbox 判定（同共用链）。
  4. **`promote_on_bypass(family, cluster, …)`**：bypass 则把这条链按 (家族×人脸类型) 升进 Pareto 池；失败则 `record_use(success=False)`。
  5. **协同进化**：每 N 个 rollout 用 LLM mutator 从 top-bypass 链重生成 snippet 候选（Eevee 思路但更省）。
  - 与 M2 的区别：M2 按家族进化一套技能；**M4 按"家族×人脸类型"维度细分进化**，针对不同人脸子群用不同攻法。
- **实测**（`outputs/p8_faithful/p8_20260621_1409`）：coverage_cum 0,1,3,3,3；best_so_far→22%（frozen 轴最高），attempts=38。
- **问题**：22% 同样靠 no-op（audio_synth 40% + reenact 25%）；**clean valid 数据缺失**（没单独跑 eval_clean_coverage）。

### M5 — 种群档案共进化（旗舰，但数据最薄 + 退化）

- **核心机制（`scripts/coevo/run_coevolution_m5.py`）**：M3 是 1 条 lineage；M5 = **K=3 条攻击者 lineage 的种群** 共进化 vs 同一个可训练检测器：
  1. **每轮**：K 条 lineage **并行**（独立子进程，共享 endpoint）各自打当前 D_R。
  2. 读各 lineage 的家族覆盖集（Jaccard 行为距离衡量 lineage 间差异）。
  3. **GEA 新颖度选择**：按"给共享档案新增了多少 distinct 家族"给 lineage 排名晋升；`coverage_cum = |共享档案|`。
  4. **HGM 晋升准则**：按后代覆盖生产力（累计新覆盖）而非瞬时 bypass 晋升 lineage。
  5. **并集全部 lineage 的新伪造** → 训 D_{R+1}（+replay+reals）→ EvoTest guard（recall AND real-acc 都要改善才接受）→ 热重载。
- **文献映射**：DGM(种群档案 2505.22954)、GEA(新颖度选择 2602.04837)、HGM(Clade 生产力 2510.21614)、EvoTest(平衡接受 2510.13220)。
- **实测**（`outputs/m5_coevolution/m5_scut_real_20260621_2026`，detector=强 SCUT）：

  | 轮 | 种群伪造数 | 命中 | population_bypass | coverage_cum | 档案家族 | real_acc |
  |---|---|---|---|---|---|---|
  | r0 | 41 | 1 | 2.4% | 2 | reenact,replay | 0.786→拒绝 |
  | r1 | 39 | 0 | 0% | 2 | reenact,replay | 0.933 |
  | r2 | 35 | 0 | 0% | 2 | reenact,replay | 0.80(临界) |

- **问题（严重）**：只跑 3 轮；退化（bypass 2.4%→0→0，coverage 卡在 2 个 **no-op** 家族）；**K=3 种群多样性卖点完全没被证实**（档案从没出现 distinct valid 家族）。→ 必须换 weak-base 重跑。

---

## 第三部分：横向问题 + 实习目标对标

### 跨方法结构性问题
1. **单 seed 不可复现**：M2 三跑给矛盾数字（每轮 attempts 仅 6–10，方差>信号）。
2. **五方法不在同一协议下测**：detector（强 vs vanilla）、pool（western6/scut60/scut full）、轮数（5/8/3）、briefs/rollouts 都不同 → 不能放一张可比表。
3. **诚实轴(det_bypass) 只有 W1/W2 测了**；其余 null 或 n=2–10 小样本。
4. **no-op 污染贯穿 M1/M4**：高分靠 audio_synth/reenact/replay 撑。
5. **真 KPI(AI 挖掘+生成) 信号弱**：W1 曲线衡量防御方变强，非攻击者发现新手法。

### 对标实习目标（做到了吗）
| 核心能力 / 目标 | 状态 | 证据 / 缺口 |
|---|---|---|
| 攻防自进化 **Loop** | ✅ 跑通 | W1/W2 端到端 8 轮（挖掘→生成→验证→再训练→热重载→广播） |
| AI **挖掘**（新型手法） | ⚠️ 半 | 固定家族 taxonomy 内组合探索，非开放发现；非生产环境 |
| AI **生成**（对抗样本） | ⚠️ 半 | 生成跑通，但 **face_valid 通过率仅 ~10%** → "样本质量需人工保障"痛点未解 |
| 效果**超越亚七基线** | ❌ | 只对标自建 vanilla（92%→4%），从未对标亚七生产基线 |
| 领域认知 + Deepfake 调研 | ✅ | capability-gap 律 + MART/CHASE/ARMs/QDRT/SEAL/STaSC 已梳理 |

### 不发论文/不互相比时可放下的"问题"
- 不必 3 seed 均值±std；不必统一协议塞一张表；不必纠结 M2 被 M4 反超。

### 不管怎样都仍是硬伤（实习目标本身）
1. **生成质量瓶颈**（face_valid ~10%）—— 直接打"样本质量需人工保障"痛点。
2. **没对标亚七生产基线** —— "效果超越"KPI 仍空。
3. **挖掘是闭集 taxonomy** —— "AI 主动发现新型手法"打折。
4. **Loop 只在弱自建检测器上验证** —— 没在贴近生产的检测器上跑。
5. **M5 不完整 + 退化** —— 补 weak-base 8 轮重跑，否则种群线收掉。

---

## 关键文件索引
| 内容 | 路径 |
|---|---|
| 共用攻击者 pipeline | `src/orchestrator.py`（run_one_rollout / run_round）、`src/multi_agent_gen.py`、`src/sandbox.py`、`src/operators/` |
| M1/M2 进化栈 | `src/ace_skill_lib.py`、`reasoning_bank.py`、`self_attributor.py`、`reflexion.py`、`novelty.py`、`pipeline_planner.py` |
| M3 共进化 | `src/method3_coevolution.py`、`scripts/coevo/run_coevolution_v2.py`、`train_defender_round.py`、`build_round_data.py` |
| M4 | `src/method4_orchestrator.py`、`method4_face_metadata_router.py`、`method4_pareto_skill_pool.py` |
| M5 | `scripts/coevo/run_coevolution_m5.py` |
| clean valid 重算 | `scripts/coevo/eval_clean_coverage.py` |
| 结果 | frozen: `outputs/p8_faithful/`；clean: `outputs/clean_coverage_eval.json`；M3: `outputs/m3v2/`、`m3_coevolution/`；weak-start: `outputs/weakstart/W1*/`、`W2*/`；M5: `outputs/m5_coevolution/` |
