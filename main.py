# from fastapi import FastAPI
# from app.routers import yield_router
#
# app = FastAPI(title="AI Farming API")
#
# app.include_router(yield_router.router, prefix="/ai")
#
#
# @app.get("/")
# def home():
#     return {"message": "AI Farming API"}
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import yield_router, fertilizer_router, crop_router

app = FastAPI(title="AI Farming API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(yield_router.router, prefix="/ai")
app.include_router(fertilizer_router.router, prefix="/ai")
app.include_router(crop_router.router, prefix="/ai")

@app.get("/")
def home():
    return {"message": "AI Farming API running"}
