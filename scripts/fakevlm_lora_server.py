"""Co-evolution FakeVLM detector with HOT-RELOADABLE defender LoRA.

Same faithful raw-completion path as fakevlm_raw_server.py (8001), but the
detector can mount/swap a defender LoRA at runtime so Method-3 co-evolution can
harden the detector round-by-round WITHOUT restarting the engine.

  - Round 0: no LoRA mounted -> identical to the frozen 8001 baseline.
  - After each attacker round: train a defender LoRA, POST /v1/load_lora_adapter
    with its path; the next round's attacks face the hardened detector.

OpenAI-compatible so the orchestrator's FakeVLMJudge works unchanged, PLUS the
two vLLM-style LoRA control endpoints CoEvolutionLoop.reload_lora() already posts to:
  POST /v1/load_lora_adapter   {"lora_name":"defender","lora_path":"/abs/dir"}
  POST /v1/unload_lora_adapter {"lora_name":"defender"}

Run (pin to a free GPU, e.g. 6):
  CUDA_VISIBLE_DEVICES=6 <gca-vllm python> fakevlm_lora_server.py --port 8002
"""
import argparse, base64, io, json, threading

from PIL import Image
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

CKPT = "/data/disk4/lyx_ICML/self_evolution_forgery/scripts/fakevlm_correct_ckpt"
PROMPT = "<image>Does the image looks real/fake?"   # verbatim from test.json

_llm = None
_lock = threading.Lock()
_sp = SamplingParams(max_tokens=64, temperature=0.0)

# Hot-swappable defender LoRA. None => base FakeVLM (frozen-equivalent, round 0).
_current_lora = None          # type: LoRARequest | None
_lora_counter = 0             # monotonic int id; bump every reload so vLLM never
                              # serves a stale cached adapter for a reused path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _img_from_messages(body):
    for m in body.get("messages", []):
        content = m.get("content")
        if isinstance(content, list):
            for c in content:
                if c.get("type") == "image_url":
                    url = c["image_url"]["url"]
                    b64 = url.split(",", 1)[1] if "," in url else url
                    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    return None


def _generate(img):
    with _lock:  # LLM.generate is not reentrant; serialize (correctness > throughput)
        kw = {"lora_request": _current_lora} if _current_lora is not None else {}
        outs = _llm.generate([{"prompt": PROMPT, "multi_modal_data": {"image": img}}],
                             _sp, use_tqdm=False, **kw)
    return outs[0].outputs[0].text


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.rstrip("/") in ("/health", "/ping"):
            lp = _current_lora.lora_path if _current_lora else None
            self._send(200, {"status": "ok", "lora": lp})
        elif self.path.endswith("/models"):
            self._send(200, {"object": "list", "data": [
                {"id": "fakevlm_lora", "object": "model"}, {"id": CKPT, "object": "model"}]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        global _current_lora, _lora_counter
        n = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            return self._send(400, {"error": f"bad json: {e}"})

        # ── LoRA control endpoints (match CoEvolutionLoop.reload_lora) ──
        if self.path.endswith("/load_lora_adapter"):
            path = body.get("lora_path")
            name = body.get("lora_name", "defender")
            if not path:
                return self._send(400, {"error": "lora_path required"})
            with _lock:
                _lora_counter += 1
                _current_lora = LoRARequest(name, _lora_counter, path)
            return self._send(200, {"status": "loaded", "lora_name": name,
                                    "lora_path": path, "lora_id": _lora_counter})
        if self.path.endswith("/unload_lora_adapter"):
            with _lock:
                _current_lora = None
            return self._send(200, {"status": "unloaded"})

        # ── inference ──
        if not self.path.endswith("/chat/completions"):
            return self._send(404, {"error": "only /v1/chat/completions"})
        img = _img_from_messages(body)
        if img is None:
            return self._send(400, {"error": "no image_url in messages"})
        try:
            text = _generate(img)
        except Exception as e:
            return self._send(500, {"error": f"generate failed: {e}"})
        lp = _current_lora.lora_path if _current_lora else None
        self._send(200, {"choices": [{"message": {"role": "assistant", "content": text},
                                      "finish_reason": "stop"}],
                         "model": "fakevlm_lora", "lora": lp})


def main():
    global _llm
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--gpu-mem", type=float, default=0.30)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--max-lora-rank", type=int, default=16)
    args = ap.parse_args()
    print(f"[fakevlm_lora_server] loading {CKPT} (enable_lora) ...", flush=True)
    _llm = LLM(model=CKPT, dtype="float16", max_model_len=args.max_model_len,
               gpu_memory_utilization=args.gpu_mem, tensor_parallel_size=1,
               enforce_eager=True, enable_lora=True, max_lora_rank=args.max_lora_rank,
               max_loras=1)
    print(f"[fakevlm_lora_server] ready on :{args.port} (no LoRA = frozen baseline)", flush=True)
    ThreadingHTTPServer(("0.0.0.0", args.port), H).serve_forever()


if __name__ == "__main__":
    main()
