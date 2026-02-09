Launch the Streamlit dashboard and verify it's running.

## Steps

1. Activate the virtual environment:
   ```bash
   source .env/bin/activate
   ```

2. Verify dependencies are installed:
   ```bash
   python -c "import streamlit; import plotly; import pandas; print('All imports OK')"
   ```
   If any import fails, run `pip install -r requirements.txt` and retry.

3. Verify the database exists:
   ```bash
   test -f data/loans.db && echo "Database found" || echo "ERROR: data/loans.db not found"
   ```
   If missing, suggest running `python scripts/export_to_sqlite.py` first.

4. Launch Streamlit:
   ```bash
   streamlit run app.py --server.headless true
   ```

5. Report the URL when it's running (typically `http://localhost:8501`)

## Troubleshooting

If Streamlit fails to start:
- **ModuleNotFoundError**: Install the missing package into `.env`
- **FileNotFoundError for loans.db**: Run `python scripts/export_to_sqlite.py`
- **ImportError from src/**: Verify `src/__init__.py` exists
- **Port already in use**: Try `streamlit run app.py --server.port 8502`

## Important

- Always use the `.env` virtual environment
- Do not run with `--server.runOnSave true` in production — it causes unnecessary reloads
- The app should load the full dataset once (cached) and then respond quickly to filter changes
