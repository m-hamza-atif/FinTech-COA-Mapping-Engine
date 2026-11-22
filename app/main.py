import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response

from app.schemas.contracts import AIInput, AIOutput
from app.services.indexer import GEMINI_SHORTLIST_LIMIT, RETRIEVAL_LIMIT, SearchIndex
from app.services.llm_reranker import DEFAULT_MODEL, GeminiReranker
from app.services.normalizer import Normalizer
from app.services.retriever import LocalReranker, MappingEngine

ROOT = Path(__file__).resolve().parents[1]


@asynccontextmanager
async def lifespan(application: FastAPI):
    load_dotenv(ROOT / ".env")
    data_dir = Path(os.getenv("MAPPING_DATA_DIR", str(ROOT / "data" / "processed")))
    cache_dir = Path(os.getenv("MAPPING_CHROMA_DIR", str(ROOT / ".chroma")))
    normalizer = Normalizer(data_dir)
    index = await asyncio.to_thread(SearchIndex, data_dir, cache_dir, normalizer)
    local_model = os.getenv("RERANKER_MODEL", "off").strip()
    local_reranker = None
    index.candidate_limit = GEMINI_SHORTLIST_LIMIT
    if local_model.casefold() not in {"", "off", "none", "disabled"}:
        local_reranker = await asyncio.to_thread(LocalReranker, local_model)
        index.candidate_limit = RETRIEVAL_LIMIT
    reranker = GeminiReranker(
        os.getenv("GEMINI_API_KEY"),
        os.getenv("GEMINI_MODEL", DEFAULT_MODEL),
        pace_seconds=float(os.getenv("GEMINI_PACE_SECONDS", "0")),
    )
    application.state.engine = MappingEngine(
        normalizer, index, reranker, local_reranker
    )
    try:
        yield
    finally:
        await reranker.close()


app = FastAPI(lifespan=lifespan, redirect_slashes=False)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "mapping-engine"}


@app.post("/api/v1/mappings/map-accounts", response_model=AIOutput)
async def map_accounts(payload: AIInput, request: Request, response: Response):
    response.headers["Content-Encoding"] = "identity"
    return AIOutput(
        mappings=await request.app.state.engine.map_accounts(payload.accounts)
    )


def main():
    uvicorn.run("app.main:app", host="127.0.0.1", port=8001)


if __name__ == "__main__":
    main()
