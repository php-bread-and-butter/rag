# FastAPI RAG Tutorial Project

A step-by-step FastAPI tutorial project for building a RAG (Retrieval-Augmented Generation) system.

## Features

- ✅ **Versioned API** - `/api/v1` (legacy) and `/api/v2` (modern RAG)
- ✅ **Document ingestion** - TXT, PDF, Word, CSV, Excel, JSON, SQL
- ✅ **Text splitting** - Multiple splitting strategies
- ✅ **Embeddings** - HuggingFace and OpenAI
- ✅ **RAG training** - Unified ingestion + vector storage (ChromaDB)
- ✅ **LLM query** - OpenAI/GROQ + conversational memory
- ✅ **Interactive API documentation** - Auto-generated Swagger UI

## Project Structure

```
fastapi/
├── app/
│   ├── __init__.py
│   ├── main.py                # FastAPI application entry point
│   ├── api/
│   │   ├── v1/
│   │   │   ├── api.py
│   │   │   └── endpoints/     # Legacy endpoints (ingestion, splitting, embeddings)
│   │   └── v2/
│   │       ├── api.py
│   │       └── endpoints/     # Modern RAG v2 endpoints
│   ├── core/
│   │   ├── config.py          # Application settings
│   │   └── logging_config.py  # Logging configuration
│   └── rag/                   # RAG pipeline modules
├── tests/
├── pyproject.toml             # Project configuration (uv)
├── requirements.txt           # Python dependencies (pip compatibility)
├── .python-version            # Python version pin
├── run.py                     # Development server script
└── README.md
```

## Quick Start

### Prerequisites

- Python 3.11 or higher
- [uv](https://github.com/astral-sh/uv) - Fast Python package installer

Install `uv`:
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or with pip
pip install uv
```

### Installation

```bash
# Install dependencies using uv (automatically creates .venv/)
uv sync

# Or install with dev dependencies
uv sync --extra dev

# Activate the virtual environment created by uv (optional)
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# Alternative: Run commands directly with uv without activating
uv run python run.py
```

### Environment Setup

1. **Copy the sample environment file:**
   ```bash
   cp sample.env .env
   ```

2. **Edit `.env` and add your configuration:**
   ```bash
   # Required for OpenAI embeddings (optional - HuggingFace works without it)
   OPENAI_API_KEY=your_openai_api_key_here
   
   # Optional: Logging configuration
   LOG_LEVEL=INFO
   LOG_FILE=
   ```

3. **Get OpenAI API Key (optional):**
   - Visit https://platform.openai.com/api-keys
   - Create a new API key
   - Add it to your `.env` file

**Note:** The `.env` file is already in `.gitignore` and won't be committed to version control. The `sample.env` file contains all available environment variables with descriptions.

### Run the Application

```bash
# Using the run script
python run.py

# Or directly with uvicorn
uvicorn app.main:app --reload

# Or with uv
uv run uvicorn app.main:app --reload
```

The API will be available at:
- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Endpoints

### Root & Health
```http
GET /
GET /health
```

### V1 (Legacy)
```
/api/v1/documents/*    # Document ingestion
/api/v1/splitting/*    # Text splitting
/api/v1/embeddings/*   # Embedding utilities
/api/v1/rag/*          # Legacy RAG training
```

### V2 (Modern RAG)
```
/api/v2/rag/train
/api/v2/rag/train/upload
/api/v2/rag/query
/api/v2/rag/query/rag
/api/v2/rag/collections/{collection_name}
```

## Testing with Postman

1. **Start the server:**
   ```bash
   python run.py
   ```

2. **Test Root endpoint:**
   - Method: `GET`
   - URL: `http://localhost:8000/`
   - Expected response: API metadata with `api_versions`

3. **Test Health Check endpoint:**
   - Method: `GET`
   - URL: `http://localhost:8000/health`
   - Expected response: `{"status": "healthy", ...}`

4. **Test V2 RAG endpoints:**
   - Train: `POST http://localhost:8000/api/v2/rag/train`
   - Query: `POST http://localhost:8000/api/v2/rag/query`

## Running Tests

```bash
# Run tests
pytest

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=app tests/
```

## Development

### Code Formatting

```bash
# Format code with Black
black app/

# Check code style
flake8 app/
```

### Managing Dependencies with uv

```bash
# Add a new dependency
uv add <package-name>

# Add a development dependency
uv add --dev <package-name>

# Add with version constraint
uv add "fastapi>=0.100.0"

# Remove a dependency
uv remove <package-name>

# Sync dependencies (install/update)
uv sync

# Sync with dev dependencies
uv sync --extra dev

# Update all dependencies
uv sync --upgrade
```

## Production Deployment

For production deployment on a Linux server:

### Using Gunicorn with Uvicorn Workers

```bash
# Install gunicorn
pip install gunicorn

# Run with multiple workers
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Using systemd Service

Create `/etc/systemd/system/fastapi-tutorial.service`:

```ini
[Unit]
Description=FastAPI Tutorial Application
After=network.target

[Service]
User=www-data
Group=www-data
WorkingDirectory=/path/to/fastapi
Environment="PATH=/path/to/venv/bin"
ExecStart=/path/to/venv/bin/gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 127.0.0.1:8000

[Install]
WantedBy=multi-user.target
```

Then enable and start:
```bash
sudo systemctl enable fastapi-tutorial
sudo systemctl start fastapi-tutorial
```

### Using Nginx as Reverse Proxy

Example Nginx configuration:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## Next Steps

This is a tutorial project that will be built incrementally. Future additions may include:

- More endpoints (POST, PUT, DELETE)
- Request/response models with Pydantic
- Database integration
- Authentication and authorization
- Error handling
- API versioning
- And more...

## License

MIT License
