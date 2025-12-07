from flask import Flask, render_template, request, send_from_directory, jsonify
import os
import subprocess
import sys
import shutil
import signal
import threading
import uuid
import json
import requests
from datetime import datetime
from enum import Enum
from dotenv import load_dotenv
load_dotenv()

# Handle SIGPIPE gracefully (client disconnections)
signal.signal(signal.SIGPIPE, signal.SIG_DFL)


app = Flask(__name__)

# Configure timeouts
PROCESS_TIMEOUT = 3600  # 1 hour timeout for subprocess calls (background jobs)

# reCAPTCHA configuration
RECAPTCHA_SITE_KEY = os.environ.get('RECAPTCHA_SITE_KEY', '').strip()
RECAPTCHA_SECRET_KEY = os.environ.get('RECAPTCHA_SECRET_KEY', '').strip()
RECAPTCHA_VERIFY_URL = 'https://www.google.com/recaptcha/api/siteverify'


class JobStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


class JobManager:
    """Manages background jobs and their status"""

    def __init__(self):
        self.jobs = {}
        self.lock = threading.Lock()

    def create_job(self):
        """Create a new job and return its ID"""
        job_id = str(uuid.uuid4())
        with self.lock:
            self.jobs[job_id] = {
                'status': JobStatus.PENDING.value,
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'error': None,
                'progress': 0
            }
        return job_id

    def update_job(self, job_id, status, error=None, progress=None):
        """Update job status"""
        with self.lock:
            if job_id in self.jobs:
                status_val = (status.value if isinstance(status, JobStatus)
                              else status)
                self.jobs[job_id]['status'] = status_val
                self.jobs[job_id]['updated_at'] = datetime.now().isoformat()
                if error is not None:
                    self.jobs[job_id]['error'] = error
                if progress is not None:
                    self.jobs[job_id]['progress'] = progress

    def get_job(self, job_id):
        """Get job status"""
        with self.lock:
            return self.jobs.get(job_id)

    def cleanup_old_jobs(self, max_age_hours=24):
        """Clean up jobs older than max_age_hours"""
        current_time = datetime.now()
        with self.lock:
            to_remove = []
            for job_id, job_data in self.jobs.items():
                created_at = datetime.fromisoformat(job_data['created_at'])
                age = (current_time - created_at).total_seconds() / 3600
                if age > max_age_hours:
                    to_remove.append(job_id)
            for job_id in to_remove:
                del self.jobs[job_id]


# Global job manager
job_manager = JobManager()


def get_python_executable():
    """
    Get the correct Python executable path.
    On PythonAnywhere/uwsgi, sys.executable points to uwsgi, not Python.
    """
    # If sys.executable is uwsgi, find the actual Python interpreter
    if 'uwsgi' in sys.executable.lower():
        # Try to find python3 in PATH
        python_path = shutil.which('python3')
        if python_path:
            return python_path
        # Fallback to 'python3' command
        return 'python3'
    return sys.executable


UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ACCESS_CONTROL_FOLDER = 'access_control'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def process_file_background(job_id):
    """Process file in background thread"""
    try:
        job_manager.update_job(job_id, JobStatus.PROCESSING, progress=25)

        # Step 1: Convert input.json to config.ini
        script_path = os.path.join(ACCESS_CONTROL_FOLDER, 'input.py')
        cwd = os.path.dirname(os.path.abspath(__file__))
        python_exe = get_python_executable()

        subprocess.run(
            [python_exe, script_path],
            check=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=PROCESS_TIMEOUT
        )

        job_manager.update_job(job_id, JobStatus.GENERATING, progress=50)

        # Step 2: Run gen.py to generate outputs
        script_path = os.path.join(ACCESS_CONTROL_FOLDER, 'gen.py')
        cwd = os.path.dirname(os.path.abspath(__file__))
        python_exe = get_python_executable()

        subprocess.run(
            [python_exe, script_path],
            check=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=PROCESS_TIMEOUT
        )

        job_manager.update_job(job_id, JobStatus.COMPLETED, progress=100)

    except subprocess.TimeoutExpired:
        error_msg = ("Processing timeout: The operation took too long. "
                     "Please try with a smaller file or contact support.")
        job_manager.update_job(job_id, JobStatus.FAILED, error=error_msg)
    except subprocess.CalledProcessError as e:
        error_msg = f"Processing failed: {e.stderr or str(e)}"
        job_manager.update_job(job_id, JobStatus.FAILED, error=error_msg)
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        job_manager.update_job(job_id, JobStatus.FAILED, error=error_msg)


def verify_recaptcha(recaptcha_response):
    """Verify reCAPTCHA response with Google's API"""
    if not RECAPTCHA_SECRET_KEY:
        # If no secret key is configured, skip verification (for development)
        return True

    if not recaptcha_response:
        return False

    try:
        data = {
            'secret': RECAPTCHA_SECRET_KEY,
            'response': recaptcha_response
        }
        response = requests.post(RECAPTCHA_VERIFY_URL, data=data, timeout=5)
        result = response.json()
        return result.get('success', False)
    except Exception:
        return False


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/recaptcha-site-key', methods=['GET'])
def get_recaptcha_site_key():
    """Return reCAPTCHA site key to frontend"""
    return jsonify({'site_key': RECAPTCHA_SITE_KEY}), 200


@app.route('/example', methods=['GET'])
def get_example_json():
    """Return example JSON format"""
    example_json = {
        "subject_size": 10,
        "object_size": 15,
        "environment_size": 5,
        "subject_attributes_count": 3,
        "object_attributes_count": 4,
        "environment_attributes_count": 2,
        "subject_attributes_values": [3, 4, 5],
        "object_attributes_values": [2, 3, 4, 5],
        "environment_attributes_values": [1, 2],
        "subject_mean": 3,
        "subject_variance": 1,
        "object_mean": 4,
        "object_variance": 1.5,
        "environment_mean": 2,
        "environment_variance": 0.5,
        "rules_count": 20
    }
    # Return as formatted JSON string to ensure consistent formatting
    formatted_json = json.dumps(example_json, indent=4, ensure_ascii=False)
    headers = {'Content-Type': 'application/json; charset=utf-8'}
    return formatted_json, 200, headers


@app.route('/upload', methods=['POST'])
def upload_file():
    """Upload file and start background processing"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'}), 400

    if not file.filename.endswith('.json'):
        error_msg = 'Invalid file type. Please upload a JSON file.'
        return jsonify({'success': False, 'error': error_msg}), 400

    # Verify reCAPTCHA
    recaptcha_response = request.form.get('g-recaptcha-response')
    if not verify_recaptcha(recaptcha_response):
        return jsonify({
            'success': False,
            'error': 'reCAPTCHA verification failed. Please try again.'
        }), 400

    try:
        # Save file
        input_path = os.path.join(
            app.config['UPLOAD_FOLDER'], 'input.json'
        )
        file.save(input_path)

        # Create background job
        job_id = job_manager.create_job()

        # Start background processing
        thread = threading.Thread(
            target=process_file_background,
            args=(job_id,),
            daemon=True
        )
        thread.start()

        return jsonify({
            'success': True,
            'job_id': job_id,
            'message': 'File uploaded. Processing started in background.'
        }), 200

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'An unexpected error occurred: {str(e)}'
        }), 500


@app.route('/upload-json', methods=['POST'])
def upload_json():
    """Upload JSON from editor and start background processing"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400

        # Verify reCAPTCHA
        recaptcha_response = data.pop('g-recaptcha-response', None)
        if not verify_recaptcha(recaptcha_response):
            return jsonify({
                'success': False,
                'error': 'reCAPTCHA verification failed. Please try again.'
            }), 400

        # Validate JSON structure (basic validation)
        required_fields = [
            'subject_size', 'object_size', 'environment_size',
            'subject_attributes_count', 'object_attributes_count',
            'environment_attributes_count', 'rules_count'
        ]
        missing_fields = [
            field for field in required_fields if field not in data
        ]
        if missing_fields:
            fields_str = ', '.join(missing_fields)
            return jsonify({
                'success': False,
                'error': f'Missing required fields: {fields_str}'
            }), 400

        # Save JSON to file
        input_path = os.path.join(
            app.config['UPLOAD_FOLDER'], 'input.json'
        )
        with open(input_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)

        # Create background job
        job_id = job_manager.create_job()

        # Start background processing
        thread = threading.Thread(
            target=process_file_background,
            args=(job_id,),
            daemon=True
        )
        thread.start()

        return jsonify({
            'success': True,
            'job_id': job_id,
            'message': 'JSON uploaded. Processing started in background.'
        }), 200

    except json.JSONDecodeError:
        return jsonify({
            'success': False,
            'error': 'Invalid JSON format'
        }), 400
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'An unexpected error occurred: {str(e)}'
        }), 500


@app.route('/status/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """Get status of a background job"""
    job = job_manager.get_job(job_id)

    if not job:
        return jsonify({
            'success': False,
            'error': 'Job not found'
        }), 404

    response = {
        'success': True,
        'job_id': job_id,
        'status': job['status'],
        'progress': job['progress'],
        'created_at': job['created_at'],
        'updated_at': job['updated_at']
    }

    if job['status'] == JobStatus.COMPLETED.value:
        response['message'] = 'Processing completed successfully!'
    elif job['status'] == JobStatus.FAILED.value:
        response['error'] = job.get('error', 'Processing failed')
    elif job['status'] in [JobStatus.PROCESSING.value,
                           JobStatus.GENERATING.value]:
        response['message'] = 'Processing in progress...'

    return jsonify(response), 200


@app.route('/download')
def download_outputs():
    """Download the output file"""
    file_path = os.path.join(
        app.config['OUTPUT_FOLDER'], 'output.json'
    )
    if not os.path.exists(file_path):
        error_msg = ("Error: output.json not found. "
                     "Ensure the process completed successfully.")
        return error_msg, 404
    return send_from_directory(
        app.config['OUTPUT_FOLDER'], 'output.json'
    )


if __name__ == '__main__':
    app.run(debug=True)
