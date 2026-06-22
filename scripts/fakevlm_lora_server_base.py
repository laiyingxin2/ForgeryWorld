"""Base-PARAMETRIZED hot-reloadable detector server (NEW file; the original
fakevlm_lora_server.py hardcodes the strong FakeVLM ckpt and is left untouched).

Purpose: WEAK-START co-evolution. To get a real arms-race curve (attacker ASR high
-> defender hardens -> ASR halved, CHASE/MART style) the detector must START weak.
Serving the vanilla llava-1.5-7b base (no FakeVLM detection training) gives a naive
detector the co-evolution then teaches via per-round defender LoRA. Identical HTTP
surface to fakevlm_lora_server.py (chat/completions + load/unload_lora_adapter), so
run_coevolution_v2.py drives it unchanged — only --base differs.

The per-round defender LoRA MUST be trained on the SAME base (train_defender_round.py
--base <this base>) so the adapter composes correctly on the served model.

Run (pin to a GPU with headroom):
  CUDA_VISIBLE_DEVICES=1 <vllm python> fakevlm_lora_server_base.py \
      --port 8006 --base /cpfs01/.../llava-hf/llava-1.5-7b-hf
"""
import argparse, base64, io, json, threading

from PIL import Image
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

DEFAULT_BASE = "/cpfs01/bob_workspace/students/lyx/Model_download/llava-hf/llava-1.5-7b-hf"
PROMPT = "<image>Does the image looks real/fake?"

_llm = None
_lock = threading.Lock()
_sp = SamplingParams(max_tokens=64, temperature=0.0)
_current_lora = None
_lora_counter = 0
_base = DEFAULT_BASE
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
    with _lock:
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
            self._send(200, {"status": "ok", "lora": lp, "base": _base})
        elif self.path.endswith("/models"):
            self._send(200, {"object": "list", "data": [
                {"id": "fakevlm_lora", "object": "model"}, {"id": _base, "object": "model"}]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        global _current_lora, _lora_counter
        n = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            return self._send(400, {"error": f"bad json: {e}"})

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
    global _llm, _base
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8006)
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--gpu-mem", type=float, default=0.30)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--max-lora-rank", type=int, default=16)
    args = ap.parse_args()
    _base = args.base
    print(f"[base_lora_server] loading {_base} (enable_lora) ...", flush=True)
    _llm = LLM(model=_base, dtype="float16", max_model_len=args.max_model_len,
               gpu_memory_utilization=args.gpu_mem, tensor_parallel_size=1,
               enforce_eager=True, enable_lora=True, max_lora_rank=args.max_lora_rank,
               max_loras=1)
    print(f"[base_lora_server] ready on :{args.port} base={_base} (no LoRA = naive/weak start)", flush=True)
    ThreadingHTTPServer(("0.0.0.0", args.port), H).serve_forever()


if __name__ == "__main__":
    main()
