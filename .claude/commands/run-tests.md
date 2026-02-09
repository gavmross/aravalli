Run the full test suite and report results.

## Steps

1. Activate the virtual environment: `source .env/bin/activate`
2. Run: `pytest tests/ -v --tb=short`
3. Report pass/fail for each test file and each individual test
4. If any tests fail:
   - Show the full traceback for each failure
   - Diagnose the likely root cause (wrong formula, missing column, import error, etc.)
   - Suggest a specific fix but do NOT auto-apply changes — wait for confirmation
5. If all tests pass, report the total count and confirm everything is green

## Important

- Always run from the project root directory
- If `tests/` directory doesn't exist or is empty, say so and suggest creating tests for the existing modules
- If pytest is not installed, run `pip install pytest` first
- Do not modify any test files or source files without explicit approval
