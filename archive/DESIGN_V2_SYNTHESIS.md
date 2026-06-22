# Face Forgery 自进化系统设计 v2 — 8 篇前沿论文综合

> 整理: 2026-06-19
> v2 vs v1: v1 借鉴 2+3+1 + DARWIN (中文综述);v2 在精读 8 篇 2025-2026 前沿论文(UI-Voyager / Ace-Skill / VideoWeaver / UI-TARS-2 / WebEvolver / Agent-World / AgentEvolver + Agent0-VL/XSkill/MOBIMEM/MetaForge 合集)基础上重写

---

## 0. 8 篇论文核心洞察(每一个都来自论文 verbatim)

| 论文 | 最值得偷的 1 个机制 | 数字证据 |
|---|---|---|
| **UI-Voyager** (Tencent Hunyuan, 4B) | **Fork-Point Detection**: SSIM/embedding 等价检测 + 单调对齐找到失败/成功轨迹分叉步 → SFT `[失败 prompt \| 成功 response]` | AndroidWorld 4B 81% > human 80% (RFT 73.2 → GRSD 81.0,GRPO 76 plateau) |
| **Ace-Skill** (Amap/Alibaba) | **Beta-variance Prioritized Sampling + 双流 (Experience L=120, Skill Doc W=1000)** | TIR-Bench 23→50 (+35.46% Avg@4),**super-additive**,skill clustering > exp clustering |
| **VideoWeaver** (ZJU+ByteDance) | **三层 skill 架构**: Foundation Skills + Composition Skill (S_k per category) + Creator Skill (cross-category meta) | RankingScore 2.82 → 5.23 (+85%), 285 cases 16 类 |
| **UI-TARS-2** (ByteDance Seed) | **Hybrid Verifier 三种**: function checker / LLM-as-Judge / VLM-as-Verifier;**Rejection Sampling 路由** V=1→SFT,V=0→CT | Online-Mind2Web 88.2, OSWorld 47.5 |
| **WebEvolver** (Tencent AI Lab) | **Co-evolve 节奏**: policy iter-1 + world iter-2 组合最佳;lookahead **depth=2** 严格上限 (d≥3 退化) | WV 38 (baseline) → 51.37 (+WMLA) |
| **Agent-World** (RUC+ByteDance Seed) | **环境分桶 + Tool Graph + Biased Random Walk + Diagnosis-driven** | MCP-Mark 29.5→36.3→38.1 (round 1 主要收益), 2-round 收敛 |
| **AgentEvolver** (Tongyi) | **Self-Attributing**: LLM-as-attributor 反思每步 GOOD/BAD,step→token broadcast | 7B 15.8→45.2 (+29.4), Q单独 +20.3, Q+A 41.3, Q+N+A 45.2 |
| **Agent0-VL** | **Tool-grounded Verifier**: critic 必须 re-run 工具(FaceXray/FFT/ID-cosine)而非凭空说 | +12.5% over base |
| **XSkill** | **双流 skills + experiences,visual-grounded retrieval** | — |
| **MOBIMEM** | **AgentRR Action Memory**: 缓存成功 pipeline keyed by (source, detector),命中直接 replay 跳 LLM | DisGraph 24ms vs GraphRAG 秒级 (280×) |
| **MetaForge** | **Forge-and-Recycle**: 在线写新 Python tool,gate `v=v_exec·v_sem`(必须能执行 AND judge 确认行为) | — |

---

## 1. 系统总架构(v2)

```
┌────────────────────────────────────────────────────────────────────────┐
│ Base Face Pool (CelebA-Spoof zips → 解压 → buffalo_l ArcFace 索引)       │
└──────────────────────────────┬─────────────────────────────────────────┘
                               │
        ┌──────────────────────▼───────────────────────────────────────┐
        │ Layer 0: Environment Buckets (Agent-World) — 4 类工具环境     │
        │ • S_syn (inswapper/SimSwap/GHOST/FaceDancer/9种)              │
        │ • S_id_diff (InstantID/PuLID/PhotoMaker/Arc2Face)             │
        │ • S_restoration (GFPGAN/CodeFormer/RestoreFormer++)           │
        │ • S_degradation (JPEG-q/resize/recompress/blur/noise)         │
        │ Tool Graph: 每个 tool 是 node,边权 = LLM 标注的依赖度          │
        └──────────────────────┬───────────────────────────────────────┘
                               │
        ┌──────────────────────▼───────────────────────────────────────┐
        │ Layer 1: Task Sampler (AgentEvolver + Ace-Skill 联合)         │
        │ ① Self-Questioning: 高温 LLM BFS→DFS 探索 forgery 空间        │
        │    生成 forgery brief (src_id + tgt_id + attack_class + 场景)  │
        │ ② Prioritized w/ Beta-variance:                              │
        │    wₜ(xᵢ) ∝ √(vᵢ(1-vᵢ)) + 0.4·(1-vᵢ) + 0.1                  │
        │    优先采样"刚学会但仍不稳定"的 brief                          │
        └──────────────────────┬───────────────────────────────────────┘
                               │
        ┌──────────────────────▼───────────────────────────────────────┐
        │ Layer 2: Generation = Biased Random Walk on Tool Graph        │
        │ pipeline σ = align→inswap→InstantID→GFPGAN→JPEG→resize        │
        │ 每个 step 可被 Skill Library 约束                              │
        └──────────────────────┬───────────────────────────────────────┘
                               │
        ┌──────────────────────▼───────────────────────────────────────┐
        │ Layer 3: Co-evolving Hybrid Verifier (UI-TARS-2 + WebEvolver) │
        │ ① Cheap Proxy: 小 VLM 蒸馏(SFT 在 FakeVLM 标注上),lookahead    │
        │    depth=2,k=3 candidates → predict FakeVLM 难度分             │
        │ ② Expensive FakeVLM (LLaVA-1.5-7B fakevlm) only on top-1       │
        │ ③ Function check: ArcFace ID-sim, IQA (NIQE/MANIQA), FID       │
        │ Tool-grounded critique (Agent0-VL): 必须引用 FFT/heatmap/cosine │
        └──────────────────────┬───────────────────────────────────────┘
                               │  (pass / fail + step-level critique)
        ┌──────────────────────▼───────────────────────────────────────┐
        │ Layer 4: Self-Attributing (AgentEvolver)                      │
        │ LLM-as-attributor (gemini-3-pro 反思每步 GOOD/BAD):            │
        │   "Step 0 inswap: GOOD (identity gap),                        │
        │    Step 1 GFPGAN: BAD (引入 artifact 让 FakeVLM 警觉),          │
        │    Step 2 JPEG: GOOD (抹掉 artifact)"                          │
        │ → r^attr = [+1, -1, +1], A_t = Σ r_k                          │
        └──────────────────────┬───────────────────────────────────────┘
                               │
        ┌──────────────────────▼───────────────────────────────────────┐
        │ Layer 5: Three-Layer Skill Library (VideoWeaver + Ace-Skill)   │
        │                                                                │
        │ ① Foundation Skills (工具原语): 9 ONNX swap + 4 ID-diff +     │
        │   3 restoration + 5 degradation (固定,by env bucket)          │
        │                                                                │
        │ ② Composition Skill S_k (per attack category, K=6 clusters):  │
        │   - Tactical Experience Pool ℰ_k (L=120 entries, top-k cosine) │
        │   - Strategic Skill Doc 𝒮_k (Markdown SOP, W=1000 words)      │
        │   Routing: text-embedding-3-small + K-means deterministic      │
        │                                                                │
        │ ③ Creator Skill C (single, cross-category meta):              │
        │   生成新的 S_k 当新攻击类别出现                                  │
        │                                                                │
        │ Dedupe: τ=0.70 cosine → LLM Merge;|ℰ_k|>L → 迭代合并最相似对   │
        │ Compress: 𝒮_k ← Compress(Merge(𝒮_k, Δ𝒮_i), W=1000)            │
        └──────────────────────┬───────────────────────────────────────┘
                               │
        ┌──────────────────────▼───────────────────────────────────────┐
        │ Layer 6: Fork-Point SFT (UI-Voyager GRSD-style)               │
        │ 对同一 brief 跑 G=8 rollouts:                                  │
        │  - 取最短 successful τ⁺ 作 teacher                            │
        │  - SSIM+ArcFace+landmark 等价检测 + 单调对齐找分叉步           │
        │  - 构造 (失败 prompt | 成功同步 response) → SFT controller LLM │
        └──────────────────────┬───────────────────────────────────────┘
                               │
        ┌──────────────────────▼───────────────────────────────────────┐
        │ Layer 7: Action Cache (MOBIMEM AgentRR) +                     │
        │          MetaForge Forge-and-Recycle                          │
        │ Cache: 成功 pipeline keyed by (src_id_hash, FakeVLM_ckpt_sig)  │
        │        命中直接 replay 跳 LLM (省 90% LLM call)                │
        │ Forge: 当现有 tool 都打不过 → agent 写新 Python op,            │
        │        gate v=v_exec·v_sem (能跑 + judge 确认行为) 才入库      │
        └───────────────────────────────────────────────────────────────┘
```

---

## 2. 关键参数 cheat sheet(全部 verbatim 来自论文)

```yaml
# Ace-Skill Prioritized
beta_decay_rho: 0.95
difficulty_bias_gamma: 0.4
exploration_floor_epsilon: 0.1

# Ace-Skill Clustering
K_categories: 6  # face-swap-frontal/profile/expression/age/restoration/id-mix
embedder: text-embedding-3-small
similarity_threshold_tau: 0.70
experience_pool_capacity_L: 120
skill_doc_word_budget_W: 1000

# UI-Voyager Group Self-Distillation
group_size_G: 8
SSIM_threshold_theta: 0.85  # 论文未明,我们设
hash_prefilter: 0.80
RFT_rounds: 3
GRSD_rounds: 1

# WebEvolver Lookahead
branching_factor_k: 3
lookahead_depth_d: 2   # 严格 ≤ 2
co_evolve_iters: 3      # policy iter-1 + judge iter-2 组合最佳

# Agent-World Loop
flywheel_rounds: 2      # 论文实测 round 1 主要收益, ≤3 即可
GRPO_tasks_per_step: 32
rollouts_per_task: 8
max_trajectory_tokens: 80_000

# AgentEvolver Self-Attributing
attribution_alpha: 0.1   # composite r̂_t = α·r^attr + 1[t=T]·r^out
selective_boost_eps_high: 0.6  # vs default 0.28

# MOBIMEM AgentRR
cache_key: (src_id_hash, FakeVLM_ckpt_signature, attack_category)
cache_invalidation: UI-hash drift = 任 detector 升级后清空

# MetaForge Forge
forge_validation: v_exec AND v_sem
forge_pool_capacity_K: 32
```

---

## 3. Skill 存储 schema(最终版)

合并 best-skills SKILL.md 规范 + Ace-Skill 双流 + VideoWeaver 三层:

```
outputs/skills/
├── foundation/                          # 工具原语层 (Layer 5-①)
│   ├── inswapper_128/
│   │   └── SKILL.md  # name, desc, input_schema, exec_cmd
│   ├── GFPGAN_v1.4/
│   ├── InstantID/
│   └── ...
│
├── composition/                          # per-category 编排层 (Layer 5-②)
│   ├── face-swap-frontal/
│   │   ├── SKILL.md  # 𝒮_k Strategic Doc (W=1000 字 Markdown)
│   │   ├── experience.jsonl  # ℰ_k Tactical Pool (L=120 entries)
│   │   ├── metadata.json  # source_round, applied_count, success_rate
│   │   ├── reference/
│   │   │   ├── failure_cases.md
│   │   │   └── successful_chains.md
│   │   └── examples/
│   │       └── before_after.png
│   ├── face-swap-profile/
│   ├── expression-transfer/
│   ├── age-edit/
│   ├── restoration-boost/
│   └── identity-mix/
│
├── creator/                              # cross-category meta (Layer 5-③)
│   └── SKILL.md  # 全局 SOP: 如何根据新 brief 生成新 composition skill
│
├── action_cache/                         # MOBIMEM AgentRR
│   └── cache_{src_hash}_{detector_sig}.json  # 直接 replay 跳 LLM
│
└── forged_tools/                         # MetaForge 新生成的 tools
    ├── frequency_band_noiser/
    │   ├── SKILL.md
    │   ├── script.py
    │   ├── validation_test.py
    │   └── v_sem_judge_log.json
    └── ...
```

### Composition Skill SKILL.md 真实样例

```markdown
---
name: composition-face-swap-frontal
description: 正脸 face swap pipeline 的 SOP。当 brief 包含 frontal/正脸/face-swap 关键词,且 src 与 tgt yaw 差 < 30° 时使用。Triggers: frontal, face-swap, swap, identity-replace, ArcFace-id-replacement.
category: face-swap-frontal
cluster_id: 0
source_rounds: [2, 3, 5]
applied_count: 47
applied_success_rate: 0.83
deprecated: false
last_updated_round: 7
---

# Face Swap Frontal SOP

## 适用条件
- src_face yaw, tgt_face yaw 都 ∈ [-30°, +30°]
- 光照差异 < 1.0 EV
- 双方非戴墨镜 / 重度遮挡

## 强制规则 (Must Follow)
1. **必须**先用 InstantID 做 identity-lock(id_strength ∈ [0.75, 0.85])
2. **然后**用 inswapper_128 或 SimSwap-256 做几何替换
3. **如果** detector confidence > 0.7 → 加 GFPGAN-v1.4 (w=0.5)
4. **最后**JPEG QP=85 + bicubic resize 模拟传播

## 推荐 pipeline (Foundation Skills 组合)
```yaml
pipeline:
  - tool: InstantID
    params: {id_strength: 0.80, controlnet_scale: 0.5}
  - tool: inswapper_128
    params: {blend_ratio: 0.6}
  - tool: GFPGAN
    params: {version: v1.4, w: 0.5}
    conditional: detector_conf > 0.7
  - tool: JPEG
    params: {qp: 85}
  - tool: resize
    params: {scale: 0.9, interp: bicubic}
```

## 失败案例(供反思)
查看 `reference/failure_cases.md`,r3_017 / r4_008 失败因为...

## 验证方法
- ArcFace ID-sim with src > 0.65
- FakeVLM evasion_score > 0.6
- NIQE < 8.5
```

### Tactical Experience Pool entry

```json
{
  "id": "exp_0042",
  "trigger_embedding": [...],  // text-embedding-3-small of trigger desc
  "trigger_text": "dark-skin frontal female, low-light scene, occluded jaw, target FakeVLM v2.3",
  "content": "inswapper_128 (blend 0.55) → GFPGAN-v1.4 (w=0.45) → JPEG QP=82 → bicubic resize 0.85 successfully bypassed FakeVLM v2.3 confidence to 0.31 on this case. Note: GFPGAN w<0.5 critical, higher introduces over-smoothing artifact.",
  "source_round": 3,
  "fork_point_step": 2,
  "applicability_score": 0.83,
  "merged_from": ["exp_0017", "exp_0021"]
}
```

---

## 4. 训练 / 推理 loop(伪代码)

```python
# ───────────────── INFERENCE (single brief) ─────────────────
def generate_forgery(brief, skill_lib, judge_proxy, judge_real, generator_tools):
    # 1. Action Cache (MOBIMEM)
    cache_key = (hash(brief.src), judge_real.signature, brief.attack_class)
    if cached := skill_lib.action_cache.get(cache_key):
        return cached.replay(generator_tools)  # 跳 LLM 直接 replay

    # 2. Skill Retrieval (Ace-Skill clustering)
    cluster_id = skill_lib.route(brief)             # K-means on text-emb
    S_k = skill_lib.composition[cluster_id].doc       # 𝒮_k Markdown
    E_k_topk = skill_lib.composition[cluster_id].retrieve(brief, k=5)  # ℰ_k

    # 3. Pipeline Planning (Tool Graph random walk)
    sys_prompt = SYSTEM_PROMPT + S_k + format_experiences(E_k_topk)
    pipeline = LLM_controller(sys_prompt, brief)     # outputs ordered tool list

    # 4. Lookahead (WebEvolver depth=2, k=3)
    candidates = [pipeline.mutate() for _ in range(3)]   # k=3 mutations
    scored = []
    for cand in candidates:
        # depth=2 lookahead with proxy judge
        partial_imgs = [generator_tools.exec(cand[:i+1], brief) for i in range(2)]
        proxy_score = judge_proxy.score_chain(partial_imgs)
        scored.append((cand, proxy_score))
    best_cand = max(scored, key=lambda x: x[1])[0]

    # 5. Execute full pipeline → image
    img = generator_tools.exec(best_cand, brief)

    # 6. Real FakeVLM verification (only top-1)
    judgement = judge_real.score(img)  # {label, confidence, reason, evasion_score}
    return img, best_cand, judgement


# ───────────────── EVOLUTION (per round) ─────────────────
def evolve_round(brief_pool, skill_lib, judge_proxy, judge_real, controller_llm):
    """One round of self-evolution."""
    # === A. Sample (Ace-Skill Prioritized + AgentEvolver Self-Questioning) ===
    weights = [
        sqrt(v*(1-v)) + 0.4*(1-v) + 0.1
        for v in [(1 + s.alpha * 0.95**(round-s.t)) / (2 + s.alpha*0.95**(round-s.t) + s.beta*0.95**(round-s.t))
                  for s in brief_pool]
    ]
    batch = random.choices(brief_pool, weights=weights, k=32)

    # === B. Generate G=8 rollouts per brief (UI-Voyager group) ===
    all_rollouts = {}
    for brief in batch:
        rollouts = []
        for g in range(8):
            img, pipeline, judgement = generate_forgery(brief, skill_lib, judge_proxy, judge_real, ...)
            rollouts.append((pipeline, img, judgement))
        all_rollouts[brief] = rollouts

    # === C. Self-Attributing (AgentEvolver) ===
    attributions = {}
    for brief, rollouts in all_rollouts.items():
        for pipeline, img, judgement in rollouts:
            # LLM reflects on each step GOOD/BAD
            r_attr = llm_attribute(pipeline, judgement)  # [+1, -1, +1, ...]
            # composite: r̂_t = α·r^attr + 1[t=T]·r^out
            attributions[(brief, pipeline)] = compose_reward(r_attr, judgement, alpha=0.1)

    # === D. Tool-grounded Critique (Agent0-VL) ===
    # judge_real must cite FFT, ID-cos, heatmap
    for brief, rollouts in all_rollouts.items():
        for pipeline, img, judgement in rollouts:
            critique = grounded_critique(img, judgement)  # incl. forensic tool outputs

    # === E. Skill Library Update (Ace-Skill + VideoWeaver) ===
    for brief, rollouts in all_rollouts.items():
        cluster_id = skill_lib.route(brief)
        # extract Δℰ_k from successful rollouts
        Delta_E = [extract_lesson(p, img, j) for (p, img, j) in rollouts if j.evasion_score > 0.6]
        # extract Δ𝒮_k strategic insights from attributions
        Delta_S = synthesize_strategic_update(attributions, brief, cluster_id)
        # apply Eq.7 dedupe + Eq.8 compress
        skill_lib.composition[cluster_id].update(Delta_E, Delta_S)

    # === F. Fork-Point SFT (UI-Voyager GRSD) — optional, only if updating controller weights ===
    fork_pairs = []
    for brief, rollouts in all_rollouts.items():
        success = [r for r in rollouts if r[2].evasion_score > threshold]
        fail = [r for r in rollouts if r[2].evasion_score <= threshold]
        if success and fail:
            teacher = min(success, key=lambda r: len(r[0]))  # shortest successful
            for failed_r in fail:
                forks = detect_fork_points(failed_r[0], teacher[0])  # SSIM+ArcFace+landmark
                for fork in forks:
                    fork_pairs.append(build_sft_pair(failed_r, teacher, fork))
    if fork_pairs and config.update_controller:
        controller_llm = sft(controller_llm, fork_pairs, mask_prompt=True)

    # === G. MetaForge: try to forge new tool for stubborn cases ===
    stubborn = [(b, rs) for b, rs in all_rollouts.items()
                if max(r[2].evasion_score for r in rs) < 0.3]
    if len(stubborn) > 5:
        new_tool = controller_llm.forge_tool(stubborn)  # write Python script
        if validate(new_tool):  # v_exec AND v_sem
            skill_lib.forged_tools.register(new_tool)

    # === H. Cache successful pipelines (MOBIMEM) ===
    for brief, rollouts in all_rollouts.items():
        best = max(rollouts, key=lambda r: r[2].evasion_score)
        if best[2].evasion_score > 0.7:
            cache_key = (hash(brief.src), judge_real.signature, brief.attack_class)
            skill_lib.action_cache[cache_key] = best[0]

    # === I. Diagnosis (Agent-World) → next round's brief priority ===
    weak_clusters = diagnose_weak_clusters(all_rollouts)
    for cluster_id in weak_clusters:
        brief_pool.add_synthesized_briefs(cluster_id, count=20)

    return skill_lib, brief_pool, controller_llm

# Main loop (Agent-World 2-round + WebEvolver 3 iter)
for round_id in range(3):
    skill_lib, brief_pool, controller_llm = evolve_round(...)
```

---

## 5. 实施 roadmap(修订后)

### Phase A — 基础设施 (今天/明天)
- [x] best-skills 兼容的 skill_manager.py 已写好测过
- [x] viviai_client.py 已写好测过(gemini-3-pro / gemini-2.5-flash / gpt-5.5 都通)
- [x] FakeVLM judge wrapper 已写好(但 ckpt 是 0 字节 placeholder,需找真权重)
- [ ] **找真 FakeVLM 权重** — 应在 `/cpfs01/oss_dataset/lyx/Forgery/{FakeVLM,FakeVLM_origin,fakevlm_check/checkpoints_5epo/llava-1.5-7b-fakevlm/checkpoint-13040/global_step13040,fakevlm_check/checkpoints4}` 之一
- [ ] **gdown CelebA-Spoof** 续传完成(目前 ~6/52 GB)
- [ ] **fakevlm env CUDA-enabled** onnxruntime-gpu 验证
- [ ] **修 antelopev2 路径**(绕过 insightface bug,直接 ONNX 调用)

### Phase B — 单 Round 跑通 (Day 2-3)
- [ ] Layer 0 — Tool Graph 建模(4 类 env × ~25 tools)
- [ ] Layer 1 — Beta-variance sampler(`brief_pool` + α/β counts + lazy decay)
- [ ] Layer 2 — Random walk pipeline planner
- [ ] Layer 3 — Hybrid verifier (proxy judge ViT 蒸馏 + FakeVLM + ArcFace + IQA)
- [ ] Layer 4 — Self-Attributing(LLM call wrapper)
- [ ] Layer 5 — 三层 Skill Library(K=6 K-means + ℰ_k L=120 + 𝒮_k W=1000)
- [ ] 单轮 inference + skill update end-to-end

### Phase C — 多轮自进化 (Day 4-7)
- [ ] Layer 6 — Fork-Point SFT 训练 controller LLM(可选,看是否更新权重)
- [ ] Layer 7 — Action Cache + Forge-and-Recycle
- [ ] 3 round 跑全,看 detection bypass rate 曲线
- [ ] Diagnosis-driven brief 合成

### Phase D — 评估 + 论文 (Day 8+)
- [ ] 对比 baseline:vanilla forgery vs Ace-Skill-only vs Full-v2
- [ ] Ablation:Prioritized only / Clustered only / Self-Attributing only
- [ ] CelebA-Spoof 测试集 evasion rate
- [ ] 跨 detector 泛化(FakeVLM v1 → v2 → 别人的 detector)

---

## 6. v2 与 v1 的差别(关键修改)

| 维度 | v1 (我自己设计) | v2 (8 论文综合后) |
|---|---|---|
| Skill 数据结构 | 单层 SKILL.md | **三层** Foundation + Composition + Creator |
| 任务采样 | 随机 / supervisor 指定 | **Beta-variance Prioritized** (Ace-Skill Eq.) |
| 评分聚合 | avg(3 judge) | **Hybrid Verifier**: proxy lookahead k=3 d=2 + real FakeVLM top-1 + Function check |
| 失败处理 | 喂给 Supervisor 提炼 skill | **Fork-Point SFT** 构造 [失败 prompt \| 成功 response] pair |
| Reward | trajectory-level pass/fail | **Step-level Self-Attributing** (LLM-as-attributor + composite r̂_t) |
| Skill 更新 | append-only markdown | **τ=0.70 dedupe + LLM Merge + W=1000 Compress** |
| 工具池 | 固定 9 ONNX + 4 ID-diff | **+ MetaForge online tool synthesis** (v_exec·v_sem 验证) |
| 推理加速 | 无 | **+ MOBIMEM AgentRR Action Cache** (命中跳 LLM) |
| 进化轮数 | 5-10 round | **2-3 round 足够**(Agent-World 实测 round 1 主要收益) |
| Co-evolve 节奏 | 同步 | **policy iter-1 + judge iter-2** (WebEvolver 发现) |

---

## 7. 立即可写的代码模块(优先级)

1. **skill_manager.py** ✅ 已完成(best-skills 兼容)
2. **viviai_client.py** ✅ 已完成
3. **fakevlm_judge.py** ✅ 已写,等真权重定位
4. **proxy_judge.py** — ViT-B + LoRA 蒸馏 FakeVLM(WebEvolver 风格)
5. **tool_graph.py** — 4 类 env 的 tool node + edge weight,Biased Random Walk sampler
6. **beta_prioritized.py** — Ace-Skill 公式实现(α/β tracking + lazy decay + sampling)
7. **fork_point_detector.py** — UI-Voyager Algorithm 1 直接实现(SSIM+ArcFace+landmark 等价检测)
8. **self_attributor.py** — AgentEvolver LLM-as-attributor call wrapper
9. **action_cache.py** — MOBIMEM AgentRR with `(src_hash, detector_sig, attack_class)` key
10. **forge_validator.py** — MetaForge `v_exec·v_sem` gate
11. **orchestrator.py** — main loop wiring 全部 layer

---

**整理完成 2026-06-19**

下一步建议: **先做 Phase A 收尾(找真 FakeVLM 权重 + 续 CelebA-Spoof 下载),再开始按模块 4-11 写代码**。可以一边等下载一边写不依赖数据的纯逻辑模块(beta_prioritized.py / tool_graph.py / fork_point_detector.py / self_attributor.py)。
