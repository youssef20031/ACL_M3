"""Railway startup script - reads PORT from environment at runtime."""
import os
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api_main:app", host="0.0.0.0", port=port)
