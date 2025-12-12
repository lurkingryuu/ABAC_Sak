```markdown
- Run python app.py
- Sample input present in dataset folder

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
   ```
   RECAPTCHA_SITE_KEY=your-site-key-here
   RECAPTCHA_SECRET_KEY=your-secret-key-here
   ```

3. **For Development/Testing:**
   - If environment variables are not set, reCAPTCHA verification will be skipped
   - For production, always set both keys

4. **For PythonAnywhere:**
   - Add environment variables directly in your WSGI configuration file:
   ```python
   import os
   os.environ['RECAPTCHA_SITE_KEY'] = 'your-site-key-here'
   os.environ['RECAPTCHA_SECRET_KEY'] = 'your-secret-key-here'
   ```
   - Or create a `.env` file in your project directory (if using python-dotenv)
