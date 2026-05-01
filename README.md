```markdown
- Install dependencies with [uv](https://docs.astral.sh/uv/): `uv sync`
- Run the app: `uv run python app.py`
- Sample input is in the `dataset` folder; use `uv run python` for CLI steps (e.g. `access_control/input.py`, `access_control/gen.py`, `access_control/plot.py`).
- Run tests: `uv run python -m unittest discover -s tests -p 'test_*.py'`

## reCAPTCHA Configuration

This application uses Google reCAPTCHA v2 to protect against spam and abuse. To enable reCAPTCHA:

1. **Get reCAPTCHA Keys:**
   - Visit [Google reCAPTCHA Admin Console](https://www.google.com/recaptcha/admin)
   - Register your site and get:
     - Site Key (public key)
     - Secret Key (private key)

2. **Set Environment Variables:**
   ```bash
   export RECAPTCHA_SITE_KEY="your-site-key-here"
   export RECAPTCHA_SECRET_KEY="your-secret-key-here"
```

   Or create a `.env` file (if using python-dotenv):

1. **For Development/Testing:**
  - If environment variables are not set, reCAPTCHA verification will be skipped
  - For production, always set both keys
2. **For PythonAnywhere:**
  - Add environment variables directly in your WSGI configuration file:
  - Or create a `.env` file in your project directory (if using python-dotenv)

