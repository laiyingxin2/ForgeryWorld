# External repositories

Third-party repos referenced for method design / literature comparison. They are
**not vendored** here (each carries its own license + `.git`). Clone into `external/`
if you want them locally:

```bash
mkdir -p external && cd external

git clone https://github.com/AMAP-ML/Ace-Skill              # ACE skill library (residual-ASR, counter-skill)
git clone https://github.com/modelscope/AgentEvolver        # AgentEvolver (self-evolving agent)
git clone https://github.com/agiresearch/ASB                # Agent Security Bench
git clone https://github.com/inclusionAI/AWorld             # AWorld multi-agent framework
git clone https://github.com/ZJU-LLM-Safety/DARWIN          # DARWIN (LLM-safety self-evolution)
git clone https://github.com/Princeton-AI2-Lab/EEVEE.git    # EEVEE
git clone https://github.com/agentscope-ai/ReMe             # ReMe (reasoning/memory)
git clone https://github.com/Continual-Intelligence/SEAL.git # SEAL (self-adapting LLMs)
git clone https://github.com/VityaVitalich/STaSC.git        # STaSC (Non-Decreasing self-correction)
git clone https://github.com/EricTan7/Veritas              # Veritas
```

These informed the co-evolution design (SEAL self-edit, STaSC Non-Decreasing guard,
AgentEvolver population/novelty, ACE residual-ASR bypass-floor). See the method
docstrings in `scripts/coevo/` for which idea maps where.
