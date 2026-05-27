import uvicorn
from api.chat_api import app
from config.settings import API_CONFIG

if __name__ == "__main__":
    uvicorn.run(app, host=API_CONFIG.host, port=API_CONFIG.port)