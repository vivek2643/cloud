import logging
import traceback

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.routers import folders, files, upload, edit, search, logs as logs_router, renders, edl

logger = logging.getLogger(__name__)

app = FastAPI(title="AeroDrive API", version="0.1.0")

# Permissive CORS in dev so the browser can reach us from localhost / 127.0.0.1
# / the LAN IP that Next.js prints on startup. The regex matches:
#   http://localhost:<port>
#   http://127.0.0.1:<port>
#   http://192.168.x.x:<port>  (typical home LAN)
#   http://10.x.x.x:<port>     (typical office LAN)
DEV_ORIGIN_REGEX = (
    r"^http://(localhost|127\.0\.0\.1|192\.168\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3}):\d+$"
)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=DEV_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def cors_aware_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Without this, uncaught exceptions in handlers return a 500 that is missing
    `Access-Control-Allow-Origin`, which the browser surfaces as a generic
    "Failed to fetch" with no body. With this, the response includes CORS
    headers so the real error reaches the devtools network tab.
    """
    logger.exception("Unhandled error in %s %s", request.method, request.url.path)

    origin = request.headers.get("origin", "")
    headers = {}
    if origin:
        headers["Access-Control-Allow-Origin"] = origin
        headers["Access-Control-Allow-Credentials"] = "true"
        headers["Vary"] = "Origin"

    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal Server Error",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc().splitlines()[-6:],
        },
        headers=headers,
    )


app.include_router(folders.router)
app.include_router(files.router)
app.include_router(upload.router)
app.include_router(edit.router)
app.include_router(search.router)
app.include_router(logs_router.router)
app.include_router(renders.router)
app.include_router(edl.router)


@app.get("/health")
def health():
    return {"status": "ok"}
