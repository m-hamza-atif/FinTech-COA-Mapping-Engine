# Fintech Mapping Engine

Maps source ledger accounts to a real estate chart of accounts. The engine runs
on port 8001. Fintech-Backend connects to it over HTTP.

## Clone, build, and run

Install Git and Docker with Linux containers enabled. Docker supplies Python
and `uv` inside the build. Replace `ENGINE_REPO_URL` with the shared Git URL.
Before building, make sure the prepared datasets are in `data/processed/`.

```sh
git clone "ENGINE_REPO_URL" Fintech-Mapping-Engine
cd Fintech-Mapping-Engine
docker build -t fintech-mapping-engine:local .
docker run --rm --name fintech-mapping-engine -p 127.0.0.1:8001:8001 fintech-mapping-engine:local
```

If you only have the original workbooks, prepare the datasets before building.
Run the following with Python 3.11+ in your Python environment.

```sh
python -m pip install "openpyxl>=3.1.5"
python scripts/prepare_datasets.py --raw-dir "path/to/Given-Resources"
```

Open [engine health](http://127.0.0.1:8001/health) after startup. Expect
`{"status":"ok","service":"mapping-engine"}`. The mapping endpoint is
`POST /api/v1/mappings/map-accounts`.

For optional Gemini reranking, put an unquoted `GEMINI_API_KEY=your-key` in the
ignored `.env` file and add `--env-file .env` before the image name in
`docker run`. With no key, ambiguous matches use local retrieval and go to review.

## Run a shared Docker Hub image

If your team has published an image, use its namespace and tag. The image
includes the model and prepared account data, so use a private repository for
team data and sign in with an account that has access.

```sh
docker login
docker pull YOUR_NAMESPACE/fintech-mapping-engine:v0.1.0
docker run --rm --name fintech-mapping-engine -p 127.0.0.1:8001:8001 YOUR_NAMESPACE/fintech-mapping-engine:v0.1.0
```

## Smoke test with Fintech-Backend

Leave the engine running. In the Fintech-Backend checkout, set up Python as
described in the backend README. Then set these variables and start it in
PowerShell.

```powershell
$env:DATABASE_URL = "sqlite:///./fintech-demo.db"
$env:MAPPING_ENGINE_BASE_URL = "http://127.0.0.1:8001"
$env:MAPPING_ENGINE_PATH = "/api/v1/mappings/map-accounts"
$env:MAPPING_ENGINE_METHOD = "POST"
$env:MAPPING_ENGINE_CONTRACT_PROFILE = "complete-v1"
$env:MAPPING_ENGINE_TIMEOUT_SECONDS = "120"
python -m alembic upgrade head
if ($LASTEXITCODE -ne 0) { throw "Database migration failed." }
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

From the engine checkout, run the smoke test with Python 3.11+.

```sh
python -m pip install "httpx>=0.28.1" "pydantic>=2.10"
python scripts/smoke_e2e.py --backend-url http://127.0.0.1:8000
```

The test saves a demo batch and prints its run ID. In [backend Swagger](http://127.0.0.1:8000/docs),
look up the ID with `GET /api/v1/mappings/runs/{run_id}`. Expect a completed run
with three predictions. Omit `--backend-url` to test the engine alone.

If you use `uv`, skip the helper scripts' pip steps and use `uv run python` for
data preparation and the smoke test.
