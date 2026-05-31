"""Auto-activate the bilt CPU-embedding-offload inside EVERY Python interpreter
that has this dir on PYTHONPATH — including vLLM's EngineCore worker subprocess.

Python imports `sitecustomize` automatically at startup, before any user code,
so our register() patches UnquantizedEmbeddingMethod.create_weights + the
device_loading_context BEFORE the worker builds the model. This is how we make
the offload work under real `vllm serve` (separate worker), where the broken
plugin entry-point never loads. Mirrors manthanquant's activation strategy.

Guarded by BILT_CPU_OFFLOAD=1 so plain `python` invocations are unaffected.
"""
import os, sys

if os.environ.get("BILT_CPU_OFFLOAD") == "1":
    def _activate():
        try:
            import bilt_cpu_offload.embedding as offload
            offload.register()
            from vllm.model_executor.layers.vocab_parallel_embedding import (
                UnquantizedEmbeddingMethod as _U,
            )
            ok = _U.create_weights.__module__ == "bilt_cpu_offload.embedding"
            flag = os.path.expanduser("~/logs/bilt_offload_active.flag")
            os.makedirs(os.path.dirname(flag), exist_ok=True)
            with open(flag, "a") as f:
                f.write(f"pid={os.getpid()} patched={ok}\n")
        except Exception as e:  # never break the interpreter
            sys.stderr.write(f"[bilt_cpu_offload sitecustomize] activate failed: {e}\n")

    def _patch_gemma4_tool_parser():
        """Structure-recovering Gemma4 tool-call parser for NVFP4 W4A4 output.

        E4B-NVFP4 emits the CORRECT native shape
        (<|tool_call>call:NAME{key:<|"|>val<|"|>}<tool_call|>) but W4A4 quantization
        corrupts the special tokens in RANDOM places run-to-run — observed:
          * doubled start tag:  <|tool_call><|tool_call>
          * prefix wobble:      call:::NAME / call:call:NAME / callcall:NAME / call:function:NAME
          * arg wrapper:        {args={city:tokyo}}
          * string-delim drop:  city:<|"|>Tokyo"   (closing <|"|> -> bare ")
          * MISSING end tag:    no <tool_call|> at all
        The stock parser hard-matches '<|tool_call>call:(\\w+){...}<tool_call|>' and
        silently returns tools_called=False on any deviation (~50% loss here).

        We replace extract_tool_calls (non-streaming) with a recovery parser that does
        NOT depend on the exact tokens:
          1. locate each '<|tool_call>' start (dedupe doubles),
          2. brace-MATCH the first {...} after it (so a missing end tag is irrelevant),
          3. name = last identifier immediately before '{' (absorbs any prefix garbage),
          4. parse the body tolerantly: normalize <|"|> -> ", unwrap args={...},
             then extract key:val / key="val" pairs, stripping stray quotes.
        This is a best-effort recovery for a model whose quantization corrupts the
        token stream; it cannot be perfect (a W4A16 model is the sound choice for
        reliable tool calls) but it recovers the large majority of E4B emissions.
        """
        try:
            import re as _re, json as _json
            from vllm.tool_parsers import gemma4_tool_parser as _g
            from vllm.entrypoints.openai.engine.protocol import (
                ExtractedToolCallInformation as _EI,
                FunctionCall as _FC,
                ToolCall as _TC,
            )
            if getattr(_g.Gemma4ToolParser, "_bilt_robust", False):
                return
            START = "<|tool_call>"
            _id_run = _re.compile(r"[\w\-\.]+")
            _kv = _re.compile(r'([\w\-]+)\s*[:=]\s*"?([^",}{]*)"?')
            _wrap = _re.compile(r'^\s*(?:args|arguments)\s*=\s*\{(.*)\}\s*$', _re.DOTALL)

            def _robust_args(body):
                s = body.replace('<|"|>', '"')
                m = _wrap.match(s)
                if m:
                    s = m.group(1)
                out = {}
                for k, v in _kv.findall(s):
                    v = v.strip().strip('"').strip()
                    if k and v != "":
                        out[k] = v
                return out

            def _norm(s):
                return _re.sub(r'[^a-z0-9]', '', s.lower())

            def _resolve_name(prefix, raw_name, request):
                """Snap a possibly-corrupted name to the nearest DECLARED tool.

                W4A4 also corrupts the function name (e.g. a stray space splits
                'get_current_weather' -> '...weather'). The valid names are known from
                request.tools, so we classify into that closed set instead of trusting
                the bytes: pick the declared tool whose normalized name is a substring
                of the normalized prefix (longest wins); else the best char-overlap.
                """
                known = []
                try:
                    for t in (getattr(request, "tools", None) or []):
                        fn = getattr(t, "function", None)
                        n = getattr(fn, "name", None)
                        if n is None and isinstance(t, dict):
                            n = (t.get("function") or {}).get("name")
                        if n:
                            known.append(n)
                except Exception:
                    known = []
                if not known or raw_name in known:
                    return raw_name
                cand = _norm(prefix)
                subs = [k for k in known if _norm(k) and _norm(k) in cand]
                if subs:
                    return max(subs, key=lambda k: len(_norm(k)))
                rn = _norm(raw_name)
                if rn:
                    scored = sorted(((sum(c in _norm(k) for c in set(rn)), k)
                                     for k in known), reverse=True)
                    if scored and scored[0][0] > 0:
                        return scored[0][1]
                return raw_name

            def _brace_match(text, open_idx):
                depth = 0
                for i in range(open_idx, len(text)):
                    c = text[i]
                    if c == "{":
                        depth += 1
                    elif c == "}":
                        depth -= 1
                        if depth == 0:
                            return i
                return -1

            def extract_tool_calls(self, model_output, request):
                if START not in model_output:
                    return _EI(tools_called=False, tool_calls=[], content=model_output)
                try:
                    calls, pos = [], 0
                    while True:
                        s = model_output.find(START, pos)
                        if s == -1:
                            break
                        brace = model_output.find("{", s + len(START))
                        if brace == -1:
                            break
                        end = _brace_match(model_output, brace)
                        if end == -1:
                            # Unclosed brace (W4A4 dropped '}' or output truncated at
                            # max_tokens). Recover the partial body up to the end tag /
                            # next call / EOS so a truncated call still yields its args.
                            tail = model_output[brace + 1:]
                            cut = len(tail)
                            for stop in ("<tool_call|>", "<|tool_call>", "<turn|>"):
                                j = tail.find(stop)
                                if j != -1:
                                    cut = min(cut, j)
                            body = tail[:cut].rstrip().rstrip(",")
                            end = len(model_output)  # stop the loop after this one
                        else:
                            body = model_output[brace + 1:end]
                        prefix = model_output[s + len(START):brace]
                        ids = _id_run.findall(prefix)
                        name = ids[-1] if ids else None
                        if name and name not in ("call", "function", "args", "arguments"):
                            name = _resolve_name(prefix, name, request)
                            args = _robust_args(body)
                            calls.append(_TC(type="function", function=_FC(
                                name=name,
                                arguments=_json.dumps(args, ensure_ascii=False))))
                        pos = end + 1
                    if not calls:
                        return _EI(tools_called=False, tool_calls=[], content=model_output)
                    head = model_output[:model_output.find(START)].strip()
                    return _EI(tools_called=True, tool_calls=calls,
                               content=head or None)
                except Exception as e:
                    sys.stderr.write(f"[bilt gemma4 robust extract] {e}\n")
                    return _EI(tools_called=False, tool_calls=[], content=model_output)

            _g.Gemma4ToolParser.extract_tool_calls = extract_tool_calls
            _g.Gemma4ToolParser._bilt_robust = True
        except Exception as e:
            sys.stderr.write(f"[bilt gemma4 tool-parser patch] {e}\n")

    # vLLM submodules are importable at startup; patch directly. If vllm isn't
    # imported yet that's fine — we import exactly the leaf modules register() needs.
    _activate()
    _patch_gemma4_tool_parser()

# TurboQuant 3-bit KV compression for TritonAttention (E4B). Independent gate so
# it can be toggled without the offload. register() also re-checks BILT_TURBOQUANT.
if os.environ.get("BILT_TURBOQUANT") == "1":
    try:
        import bilt_turboquant_triton as _tq
        _tq.register()
    except Exception as e:
        sys.stderr.write(f"[bilt_turboquant sitecustomize] {e}\n")
