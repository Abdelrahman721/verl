"""FastAPI backend for the medical-QA generation + dual-judge comparison app.

Endpoints
    GET  /                          -> static UI
    GET  /api/health                -> config + vLLM reachability
    GET  /api/datasets/{split}      -> dropdown listing (val | train)
    GET  /api/entry/{split}/{index} -> full entry (question, key_points, gold, ...)
    POST /api/generate              -> policy answer from the vLLM server
    POST /api/judge                 -> ONE shared Deepseek penalty run + BOTH qa
                                       judges (Sonnet, gpt-5.4-mini), each combined
                                       with the shared penalty -> two final scores.

The judge backends differ ONLY in the underlying model: their prompts and scoring
are the same objects from the verbatim qa_bedrock copy (asserted at startup).
"""

from __future__ import annotations

import asyncio
import logging
import os
import secrets

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI
from pydantic import BaseModel

from judging import data
from judging import qa_bedrock
from judging import qa_openrouter
from judging import qa_eval
from judging import combine
from judging import medical_penalty_judge as penalty

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("web_judging.server")

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(HERE, "static")

# Policy (vLLM) endpoint.
POLICY_API_BASE = os.environ.get("POLICY_API_BASE", "http://127.0.0.1:8001/v1")
POLICY_MODEL_NAME = os.environ.get("POLICY_MODEL_NAME", "medical-qa")
GEN_TEMPERATURE = float(os.environ.get("GEN_TEMPERATURE", "1.0"))
GEN_TOP_P = float(os.environ.get("GEN_TOP_P", "0.7"))
GEN_MAX_TOKENS = int(os.environ.get("GEN_MAX_TOKENS", "16384"))

# Blind evaluation: when on (default), the underlying qa-judge model identities
# (Sonnet vs gpt-5.4-mini) are withheld from BOTH the UI and the API responses,
# so a tester can't be biased by knowing which model produced which grade. The
# two systems are shown only as neutral "System A" / "System B".
BLIND = os.environ.get("WEB_JUDGING_BLIND", "1").strip().lower() in ("1", "true", "yes", "on")

# Optional shared access token. When set, every /api/* request must present it
# (header "X-Access-Token" or "?token=" query param). When empty/unset, auth is
# disabled — convenient for purely local use. Treat our own PLACEHOLDER_* default
# as "unset" so a forgotten placeholder doesn't lock everyone out (and isn't a
# real secret anyway).
_raw_token = os.environ.get("WEB_JUDGING_TOKEN", "").strip()
WEB_TOKEN = "" if _raw_token.startswith("PLACEHOLDER") else _raw_token

app = FastAPI(title="web_judging")


@app.middleware("http")
async def _require_token(request: Request, call_next):
    """Gate /api/* behind the shared token (constant-time compared). The page
    and static assets stay open so the browser can load and prompt for it."""
    if WEB_TOKEN and request.url.path.startswith("/api/"):
        supplied = request.headers.get("x-access-token") or request.query_params.get("token") or ""
        if not secrets.compare_digest(supplied, WEB_TOKEN):
            return JSONResponse({"detail": "invalid or missing access token"}, status_code=401)
    return await call_next(request)


@app.on_event("startup")
def _startup() -> None:
    # Hard guarantee: System-B reuses qa_bedrock's prompt objects verbatim.
    qa_openrouter.assert_prompt_parity()
    log.info("prompt parity OK: qa_openrouter reuses qa_bedrock prompts verbatim")
    log.info("policy endpoint: %s (model=%s)", POLICY_API_BASE, POLICY_MODEL_NAME)
    log.info("System A qa model (Sonnet/Bedrock): %s", qa_bedrock.JUDGE_MODEL)
    log.info("System B qa model (OpenRouter):     %s", qa_openrouter.JUDGE_MODEL)
    log.info("penalty judge enabled: %s", penalty.is_enabled())
    log.info("BLIND eval: %s (qa-judge model identities %s in UI/API)",
             BLIND, "HIDDEN" if BLIND else "shown")
    log.info("access token auth: %s", "ON (/api/* requires X-Access-Token)" if WEB_TOKEN else "OFF (no token set)")


def _policy_client() -> AsyncOpenAI:
    # vLLM's OpenAI server ignores the key but the client needs something.
    return AsyncOpenAI(base_url=POLICY_API_BASE,
                       api_key=os.environ.get("POLICY_API_KEY", "EMPTY"))


# --------------------------------------------------------------------------- #
# Request models
# --------------------------------------------------------------------------- #
class ChatMessage(BaseModel):
    role: str
    content: str


class GenerateRequest(BaseModel):
    messages: list[ChatMessage]
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None


class JudgeRequest(BaseModel):
    candidate: str          # full raw generation, including <think>...</think>
    # Dataset mode: reference an entry; gold/key_points loaded server-side.
    split: str | None = None
    index: int | None = None
    # Custom mode: supply the eval inputs directly (no dataset row). Used when
    # the tester asks their own question. All judging fields are optional —
    # without key_points, completeness defaults to full; without a gold
    # reference, the length penalty and penalty judge are skipped.
    question: str | None = None
    eval_mode: str | None = None        # "qa" (default) — custom conversation isn't supported
    key_points: list | None = None      # [{id, point, importance}] or plain strings
    reference: str | None = None        # gold answer (length-penalty + penalty-judge anchor)


# --------------------------------------------------------------------------- #
# Static + health
# --------------------------------------------------------------------------- #
@app.get("/")
def index() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/health")
async def health() -> dict:
    reachable = False
    detail = None
    try:
        client = _policy_client()
        models = await client.models.list()
        reachable = True
        detail = [m.id for m in models.data]
    except Exception as e:  # noqa: BLE001
        detail = f"{type(e).__name__}: {e}"
    out = {
        "policy_api_base": POLICY_API_BASE,
        "policy_model_name": POLICY_MODEL_NAME,
        "policy_reachable": reachable,
        "policy_detail": detail,
        "blind": BLIND,
        "penalty_enabled": penalty.is_enabled(),
    }
    # Only disclose judge/penalty model identities when NOT blinding.
    if not BLIND:
        out["system_a_model"] = qa_bedrock.JUDGE_MODEL
        out["system_b_model"] = qa_openrouter.JUDGE_MODEL
        out["penalty_model"] = os.environ.get("FULL_MIX_JUDGE_MODEL")
    return out


# --------------------------------------------------------------------------- #
# Dataset browsing
# --------------------------------------------------------------------------- #
@app.get("/api/datasets/{split}")
def datasets(split: str) -> dict:
    try:
        return {"split": split, "entries": data.list_entries(split)}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/entry/{split}/{index}")
def entry(split: str, index: int) -> dict:
    try:
        return data.get_entry(split, index)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except IndexError as e:
        raise HTTPException(status_code=404, detail=str(e))


# --------------------------------------------------------------------------- #
# Policy generation
# --------------------------------------------------------------------------- #
@app.post("/api/generate")
async def generate(req: GenerateRequest) -> dict:
    client = _policy_client()
    try:
        resp = await client.chat.completions.create(
            model=POLICY_MODEL_NAME,
            messages=[{"role": m.role, "content": m.content} for m in req.messages],
            temperature=req.temperature if req.temperature is not None else GEN_TEMPERATURE,
            top_p=req.top_p if req.top_p is not None else GEN_TOP_P,
            max_tokens=req.max_tokens if req.max_tokens is not None else GEN_MAX_TOKENS,
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"policy generation failed: {type(e).__name__}: {e}")
    choice = resp.choices[0]
    return {
        "text": choice.message.content or "",
        "finish_reason": getattr(choice, "finish_reason", None),
        "model": resp.model,
    }


# --------------------------------------------------------------------------- #
# Judging: one shared penalty run + both qa judges -> two final scores
# --------------------------------------------------------------------------- #
def _custom_entry(req: JudgeRequest) -> dict:
    """Build a data.get_entry-shaped bundle from a tester-supplied question.

    Only qa mode is supported for custom questions. key_points entries may be
    full dicts ({id, point, importance}) or plain strings (auto-numbered, CORE).
    """
    question = (req.question or "").strip()
    reference = (req.reference or "").strip()
    kps = []
    for i, kp in enumerate(req.key_points or [], start=1):
        if isinstance(kp, dict):
            kp.setdefault("id", i)
            kp.setdefault("importance", "CORE")
            kps.append(kp)
        elif isinstance(kp, str) and kp.strip():
            kps.append({"id": i, "point": kp.strip(), "importance": "CORE"})
    ground_truth = {"key_points": kps, "gold_answer": reference, "gold_response": reference}
    extra_info = {"eval_mode": "qa", "question": question,
                  "prompt": [{"role": "user", "content": question}]}
    return {
        "eval_mode": "qa", "question": question, "user_prompt": question,
        "reference": reference, "key_points": kps,
        "ground_truth": ground_truth, "extra_info": extra_info,
    }


@app.post("/api/judge")
async def judge(req: JudgeRequest) -> dict:
    if req.split is not None and req.index is not None:
        try:
            ent = data.get_entry(req.split, req.index)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"could not load entry: {e}")
    else:
        if not (req.question or "").strip():
            raise HTTPException(status_code=400,
                                detail="custom judging requires a question (or a split+index)")
        ent = _custom_entry(req)

    candidate = req.candidate or ""
    question = ent["question"]
    key_points = ent["key_points"]
    ground_truth = ent["ground_truth"]
    extra_info = ent["extra_info"]
    eval_mode = ent["eval_mode"]
    user_prompt = ent["user_prompt"]
    reference = ent["reference"]

    # Run the three judge calls concurrently:
    #   - penalty judge (sync, OpenRouter Deepseek) -> thread
    #   - System A qa judge (async, Sonnet/Bedrock)
    #   - System B qa judge (async, gpt-5.4-mini/OpenRouter)
    penalty_task = asyncio.to_thread(
        penalty.score_penalties, user_prompt, reference, candidate
    )
    sys_a_task = qa_eval.evaluate_qa(
        qa_bedrock, question=question, key_points=key_points, candidate=candidate,
        ground_truth=ground_truth, extra_info=extra_info, eval_mode=eval_mode,
    )
    sys_b_task = qa_eval.evaluate_qa(
        qa_openrouter, question=question, key_points=key_points, candidate=candidate,
        ground_truth=ground_truth, extra_info=extra_info, eval_mode=eval_mode,
    )

    penalty_res, sys_a, sys_b = await asyncio.gather(
        penalty_task, sys_a_task, sys_b_task, return_exceptions=True,
    )

    # Penalty: if the thread raised, fail open exactly like the source wrapper.
    if isinstance(penalty_res, Exception):
        log.warning("penalty judge raised: %r", penalty_res)
        penalty_res = penalty.empty_result(judge_ok=False)

    def _finalise(sys_res, label, model):
        if isinstance(sys_res, Exception):
            log.error("%s qa judge raised: %r", label, sys_res)
            breakdown = {"score": 0.0, "error": f"{type(sys_res).__name__}: {sys_res}"}
            verdict = {"error": str(sys_res)}
        else:
            breakdown = sys_res["breakdown"]
            verdict = sys_res["verdict"]
        final = combine.normalise_penalty_into(breakdown, penalty_res)
        payload = {"label": label, "verdict": verdict, "breakdown": breakdown, "final": final}
        # Withhold the underlying model id from the response when blinding, so a
        # tester can't uncover it via devtools / the network tab either.
        if not BLIND:
            payload["model"] = model
        return payload

    return {
        "eval_mode": eval_mode,
        "blind": BLIND,
        "key_points": key_points,   # id -> point text, so the UI can label coverage rows
        "length_penalty": {
            "threshold": qa_bedrock.LEN_PENALTY_THRESHOLD,
            "k": qa_bedrock.LEN_PENALTY_K,
            "max": qa_bedrock.LEN_PENALTY_MAX,
        },
        "penalty": penalty_res,
        "system_a": _finalise(sys_a, "System A", qa_bedrock.JUDGE_MODEL),
        "system_b": _finalise(sys_b, "System B", qa_openrouter.JUDGE_MODEL),
    }


# Static assets (if any) under /static.
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
