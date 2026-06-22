"""Guaranteed-faithful FakeVLM detector HTTP service.

WHY: the vLLM OpenAI /chat/completions path does NOT reproduce FakeVLM's validated
eval (eval_vllm.py): the vicuna chat template flips real faces, and stripping the
template breaks image-token injection (garbage output). The validated 98.9% protocol
uses LLM.generate() on the COMPLETION prompt '<image>Does the image looks real/fake?'
with multi_modal_data={'image': PIL}. This service replicates that EXACTLY and wraps
it in an OpenAI-compatible /v1/chat/completions so the orchestrator's FakeVLMJudge
(which posts image_url+text and reads choices[0].message.content) works unchanged.

Run (pin to a free GPU, e.g. 7):
  CUDA_VISIBLE_DEVICES=7 <gca-vllm python> fakevlm_raw_server.py --port 8001
"""
import argparse, base64, io, json, re, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image
from vllm import LLM, SamplingParams

# The PUBLISHED raw-completion FakeVLM (produces the 98.9% gold). NOT the multi_*
# multi-task ckpt, which was vicuna-template-trained and only emits 'Real' on the raw
# completion prompt. This staged dir = llava-1.5-7b-fakevlm weights (no tokenizer of
# its own) + the standard llava-1.5-7b tokenizer/processor copied from the multi_ ckpt.
CKPT = "/data/disk4/lyx_ICML/self_evolution_forgery/scripts/fakevlm_correct_ckpt"
PROMPT = "<image>Does the image looks real/fake?"   # verbatim from test.json conversations[0]['value']

_llm = None
_lock = threading.Lock()
_sp = SamplingParams(max_tokens=64, temperature=0.0)


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
        outs = _llm.generate([{"prompt": PROMPT, "multi_modal_data": {"image": img}}],
                             _sp, use_tqdm=False)
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
            self._send(200, {"status": "ok"})
        elif self.path.endswith("/models"):
            self._send(200, {"object": "list", "data": [
                {"id": "fakevlm_raw", "object": "model"}, {"id": CKPT, "object": "model"}]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            return self._send(400, {"error": f"bad json: {e}"})
        if not self.path.endswith("/chat/completions"):
            return self._send(404, {"error": "only /v1/chat/completions"})
        img = _img_from_messages(body)
        if img is None:
            return self._send(400, {"error": "no image_url in messages"})
        try:
            text = _generate(img)
        except Exception as e:
            return self._send(500, {"error": f"generate failed: {e}"})
        self._send(200, {"choices": [{"message": {"role": "assistant", "content": text},
                                      "finish_reason": "stop"}],
                         "model": "fakevlm_raw"})


def main():
    global _llm
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--gpu-mem", type=float, default=0.30)
    ap.add_argument("--max-model-len", type=int, default=2048)
    args = ap.parse_args()
    print(f"[fakevlm_raw_server] loading {CKPT} ...", flush=True)
    _llm = LLM(model=CKPT, dtype="float16", max_model_len=args.max_model_len,
               gpu_memory_utilization=args.gpu_mem, tensor_parallel_size=1,
               enforce_eager=True)
    print(f"[fakevlm_raw_server] ready on :{args.port}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", args.port), H).serve_forever()


if __name__ == "__main__":
    main()
