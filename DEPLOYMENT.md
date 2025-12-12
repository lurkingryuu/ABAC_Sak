# PythonAnywhere Deployment Guide

This guide will help you deploy your ABAC (Attribute-Based Access Control) Flask application to PythonAnywhere.

## Prerequisites

1. A PythonAnywhere account (free tier available at [pythonanywhere.com](https://www.pythonanywhere.com))
2. Your application code ready to upload

## Step-by-Step Deployment Instructions

### 1. Prepare Your Application

Ensure your `requirements.txt` includes all dependencies:
- Flask==2.3.2
- numpy>=2.3.5
- scipy>=1.16.3

### 2. Upload Your Code to PythonAnywhere

#### Option A: Using Git (Recommended)
1. Push your code to GitHub/GitLab/Bitbucket
2. In PythonAnywhere Dashboard, go to **Files** tab
3. Open a Bash console
4. Navigate to your home directory: `cd ~`
5. Clone your repository:
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   ```
6. Navigate into the project: `cd your-repo-name`

#### Option B: Using File Upload
1. In PythonAnywhere Dashboard, go to **Files** tab
2. Navigate to your home directory (`/home/yourusername/`)
3. Create a new folder for your project: `mkdir abac_app`
4. Upload all your project files using the file browser
5. Navigate into the project: `cd abac_app`

### 3. Set Up a Virtual Environment

1. In the Bash console, navigate to your project directory
2. Create a virtual environment:
   ```bash
   python3.10 -m venv venv
   ```
   (Use `python3.10` or `python3.11` depending on your PythonAnywhere plan)

3. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```

4. Upgrade pip:
   ```bash
   pip install --upgrade pip
   ```

5. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### 4. Configure the Web App

1. In PythonAnywhere Dashboard, go to **Web** tab
2. Click **Add a new web app**
3. Choose a domain name (e.g., `yourusername.pythonanywhere.com`)
4. Select **Manual configuration** (not WSGI file)
5. Select Python version (3.10 or 3.11)

### 5. Configure WSGI File

1. In the **Web** tab, click on the WSGI configuration file link
2. Replace the entire content with:

```python
import sys
import os

# Add your project directory to the path
path = '/home/yourusername/abac_app'  # Update with your actual path
if path not in sys.path:
    sys.path.insert(0, path)

# Set environment variables (if needed)
# os.environ['RECAPTCHA_SITE_KEY'] = 'your-site-key-here'
# os.environ['RECAPTCHA_SECRET_KEY'] = 'your-secret-key-here'

# Import your Flask app
from app import app as application

# Set working directory
os.chdir(path)
```
/

**Important:** Replace `/home/yourusername/abac_app` with your actual project path.

### 6. Configure Static Files and Directories

1. In the **Web** tab, scroll down to **Static files** section
2. Add static file mappings:
   - **URL:** `/static/`
   - **Directory:** `/home/yourusername/abac_app/static/`

3. Ensure your `uploads` and `outputs` directories exist and are writable:
   ```bash
   mkdir -p uploads outputs
   chmod 755 uploads outputs
   ```

### 7. Set Environment Variables (if needed)

If your app needs environment variables (like reCAPTCHA keys), add them directly in your WSGI configuration file:

```python
import sys
import os

# Add your project directory to the path
path = '/home/yourusername/abac_app'  # Update with your actual path
if path not in sys.path:
    sys.path.insert(0, path)

# Set environment variables
os.environ['RECAPTCHA_SITE_KEY'] = 'your-site-key-here'
os.environ['RECAPTCHA_SECRET_KEY'] = 'your-secret-key-here'

# Import your Flask app
from app import app as application

# Set working directory
os.chdir(path)
```

Alternatively, you can create a `.env` file in your project directory if you're using python-dotenv (which is included in requirements.txt).

### 8. Reload Your Web App

1. In the **Web** tab, click the green **Reload** button
2. Your app should now be live at `yourusername.pythonanywhere.com`

### 9. Test Your Application

1. Visit your website URL
2. Test the file upload functionality
3. Check that outputs are generated correctly

## Troubleshooting

### Common Issues

#### 1. Import Errors
- **Problem:** Module not found errors
- **Solution:** 
  - Ensure virtual environment is activated
  - Verify all dependencies are installed: `pip list`
  - Check that the path in WSGI file is correct

#### 2. Permission Errors
- **Problem:** Cannot write to uploads/outputs directories
- **Solution:**
  ```bash
  chmod 755 uploads outputs
  chmod 644 uploads/* outputs/* 2>/dev/null || true
  ```

#### 3. Subprocess Errors
- **Problem:** Scripts not executing properly
- **Solution:** 
  - The app.py has been updated to use `sys.executable` instead of `python3`
  - Ensure the working directory is set correctly in WSGI file

#### 4. Static Files Not Loading
- **Problem:** CSS/JS files not loading
- **Solution:**
  - Verify static file mapping in Web tab
  - Check file paths in your HTML templates
  - Ensure files exist in the static directory

#### 5. 500 Internal Server Error
- **Problem:** Application crashes
- **Solution:**
  - Check error logs in the **Web** tab → **Error log** section
  - Verify all file paths are absolute or relative to project root
  - Check that all required directories exist

### Viewing Logs

1. **Error Log:** Web tab → Error log section
2. **Server Log:** Web tab → Server log section
3. **Console Output:** Files tab → Bash console

### Updating Your Application

1. If using Git:
   ```bash
   cd ~/your-repo-name
   git pull
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. Reload the web app from the **Web** tab

## File Structure on PythonAnywhere

Your project should have this structure:
```
/home/yourusername/abac_app/
├── app.py
├── requirements.txt
├── access_control/
│   ├── input.py
│   ├── gen.py
│   ├── gen_rules.py
│   └── ...
├── templates/
│   └── index.html
├── static/
│   └── style.css
├── uploads/
├── outputs/
└── venv/
```

## Important Notes

1. **Free Tier Limitations:**
   - Your app will sleep after inactivity
   - Limited CPU time
   - No custom domains on free tier

2. **File Paths:**
   - Always use absolute paths or paths relative to project root
   - PythonAnywhere uses Linux, so use forward slashes `/`

3. **Python Version:**
   - Free tier: Python 3.10
   - Paid tiers: Python 3.10 or 3.11

4. **Database (if needed later):**
   - PythonAnywhere provides MySQL and PostgreSQL
   - Configure in the **Databases** tab

## Additional Resources

- [PythonAnywhere Help](https://help.pythonanywhere.com/)
- [Flask Deployment Guide](https://flask.palletsprojects.com/en/2.3.x/deploying/)
- [PythonAnywhere Community](https://www.pythonanywhere.com/community/)

## Support

If you encounter issues:
1. Check the error logs first
2. Review PythonAnywhere help documentation
3. Check PythonAnywhere community forums

