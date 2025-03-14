# api.py
# FastAPI backend for triggering Python scripts from Grafana
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Backend API is working!"}
