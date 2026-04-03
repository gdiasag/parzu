from typing import Any
from fastapi import FastAPI, HTTPException, Response
from contextlib import asynccontextmanager

from parzu.config import Config
from parzu.parzu_class import ParserPool
from .model import ParseRequest


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = Config()

    app.state.parser_pool = ParserPool(config, pool_size=8)
    await app.state.parser_pool.setup()

    yield

    pass


app = FastAPI(lifespan=lifespan)


@app.post("/parse")
async def parse_text(request_data: ParseRequest) -> Response:
    if not request_data.text.strip():
        raise HTTPException(status_code=400, detail="Empty text provided")

    try:
        parses = await app.state.parser_pool.parse(request_data.text)
        result_blob: str = "\n".join(parses)

        return Response(content=result_blob, media_type="text/plain")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
