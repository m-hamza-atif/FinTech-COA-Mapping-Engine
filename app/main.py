import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response

from app.schemas.contracts import AIInput, AIOutput
from app.services.indexer import SearchIndex
from app.services.llm_reranker import DEFAULT_MODEL, GeminiReranker
from app.services.normalizer import Normalizer
from app.services.retriever import MappingEngine

ROOT = Path(__file__).resolve().parents[1]


@asynccontextmanager
async def lifespan(application: FastAPI):
    load_dotenv(ROOT / ".env")
    data_dir = Path(os.getenv("MAPPING_DATA_DIR", str(ROOT / "data" / "processed")))
    cache_dir = Path(os.getenv("MAPPING_CHROMA_DIR", str(ROOT / ".chroma")))
    normalizer = Normalizer(data_dir)
    index = await asyncio.to_thread(SearchIndex, data_dir, cache_dir, normalizer)
    reranker = GeminiReranker(
        os.getenv("GEMINI_API_KEY"), os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
    )
    application.state.engine = MappingEngine(normalizer, index, reranker)
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
