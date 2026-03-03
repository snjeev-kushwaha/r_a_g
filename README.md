# ollama serve
# ollama pull llama3.1
# ollama pull nomic-embed-text
# python main.py

# pip install -r requirements.txt
# Make sure Ollama is running and These are MUCH faster.
ollama pull llama3.2:3b
ollama pull phi3
ollama run llama3.2:3b
ollama ps

ollama serve
# Start FastAPI:
uvicorn main:app --reload

http://127.0.0.1:8000/docs
