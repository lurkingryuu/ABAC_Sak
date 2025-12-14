from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    send_file,
    send_from_directory,
)
import os
import subprocess
import sys
import shutil
import signal
import threading
import uuid
import json
import requests  # type: ignore[import-untyped]
from datetime import datetime
from enum import Enum
import io
import zipfile
import time
from dotenv import load_dotenv
try:
    load_dotenv()
except PermissionError:
    # Some sandboxes / deployments may disallow reading `.env` even if present.
    # Environment variables can still be provided through the process manager.
    pass

# Handle SIGPIPE gracefully (client disconnections)
signal.signal(signal.SIGPIPE, signal.SIG_DFL)


app = Flask(__name__)

# Configure timeouts (1 hour timeout for subprocess calls / background jobs)
PROCESS_TIMEOUT = 3600

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
                'progress': 0,
                # Per-job paths (filled in by the upload handler)
                'work_dir': None,
                'input_json_path': None,
                'config_ini_path': None,
                'output_dir': None,
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

    def set_job_paths(self, job_id, work_dir, input_json_path, config_ini_path,
                      output_dir):
        """Attach per-job filesystem paths used by the pipeline."""
        with self.lock:
            if job_id in self.jobs:
                self.jobs[job_id]['work_dir'] = work_dir
                self.jobs[job_id]['input_json_path'] = input_json_path
                self.jobs[job_id]['config_ini_path'] = config_ini_path
                self.jobs[job_id]['output_dir'] = output_dir
                self.jobs[job_id]['updated_at'] = datetime.now().isoformat()

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

# Per-job workspace folder (prevents concurrent users clobbering each other)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JOBS_FOLDER = os.path.join(BASE_DIR, 'jobs')
os.makedirs(JOBS_FOLDER, exist_ok=True)

# Pipeline env vars (consumed by access_control/input.py and
# access_control/gen.py).
ENV_INPUT_JSON = "ABAC_INPUT_JSON"
ENV_CONFIG_INI = "ABAC_CONFIG_INI"
ENV_OUTPUT_DIR = "ABAC_OUTPUT_DIR"

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Storage/retention controls (important for small quotas, e.g. 480MB)
JOBS_MAX_AGE_HOURS = float(os.environ.get("ABAC_JOBS_MAX_AGE_HOURS", "12"))
JOBS_MAX_TOTAL_MB = float(os.environ.get("ABAC_JOBS_MAX_TOTAL_MB", "400"))
JOBS_CLEANUP_INTERVAL_SECONDS = float(
    os.environ.get("ABAC_JOBS_CLEANUP_INTERVAL_SECONDS", "900")
)


def _dir_size_bytes(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            fp = os.path.join(root, name)
            try:
                total += os.path.getsize(fp)
            except OSError:
                continue
    return total


def cleanup_job_folders(max_age_hours: float, max_total_mb: float) -> None:
    """
    Delete old job folders and enforce a total size cap for JOBS_FOLDER.

    This keeps the app within small storage quotas (e.g. 480MB).
    """
    now = datetime.now().timestamp()
    max_age_seconds = max_age_hours * 3600.0

    job_dirs = []
    try:
        for name in os.listdir(JOBS_FOLDER):
            p = os.path.join(JOBS_FOLDER, name)
            if not os.path.isdir(p):
                continue
            try:
                mtime = os.path.getmtime(p)
            except OSError:
                mtime = now
            job_dirs.append((name, p, mtime))
    except OSError:
        return

    # 1) Age-based deletion
    for _job_id, p, mtime in sorted(job_dirs, key=lambda x: x[2]):
        if now - mtime <= max_age_seconds:
            continue
        try:
            shutil.rmtree(p)
        except OSError:
            continue

    # Refresh list after deletions
    remaining = []
    for name, p, _mtime in job_dirs:
        if os.path.isdir(p):
            try:
                mtime = os.path.getmtime(p)
            except OSError:
                mtime = now
            remaining.append((name, p, mtime))

    # 2) Total size cap eviction (delete oldest first)
    max_total_bytes = int(max_total_mb * 1024 * 1024)
    total_bytes = sum(_dir_size_bytes(p) for _name, p, _mtime in remaining)
    for _job_id, p, _mtime in sorted(remaining, key=lambda x: x[2]):
        if total_bytes <= max_total_bytes:
            break
        try:
            size = _dir_size_bytes(p)
            shutil.rmtree(p)
            total_bytes -= size
        except OSError:
            continue

    # Keep the in-memory registry from growing forever too
    job_manager.cleanup_old_jobs(max_age_hours=max_age_hours)


def _start_cleanup_thread() -> None:
    def loop():
        while True:
            try:
                cleanup_job_folders(
                    max_age_hours=JOBS_MAX_AGE_HOURS,
                    max_total_mb=JOBS_MAX_TOTAL_MB,
                )
            except Exception:
                # Never let cleanup kill the process.
                pass
            time.sleep(JOBS_CLEANUP_INTERVAL_SECONDS)

    t = threading.Thread(target=loop, daemon=True)
    t.start()


# Start periodic cleanup to stay within small storage quotas.
# (Safe in dev; in multi-worker deployments each worker may run it.)
_start_cleanup_thread()


def job_paths(job_id: str):
    """
    Return per-job paths used by the pipeline.

    Layout:
      jobs/<job_id>/uploads/input.json
      jobs/<job_id>/config.ini
      jobs/<job_id>/outputs/*
    """
    work_dir = os.path.join(JOBS_FOLDER, job_id)
    uploads_dir = os.path.join(work_dir, 'uploads')
    outputs_dir = os.path.join(work_dir, 'outputs')
    input_json_path = os.path.join(uploads_dir, 'input.json')
    config_ini_path = os.path.join(work_dir, 'config.ini')
    return {
        'work_dir': work_dir,
        'uploads_dir': uploads_dir,
        'output_dir': outputs_dir,
        'input_json_path': input_json_path,
        'config_ini_path': config_ini_path,
    }


def process_file_background(job_id):
    """Process file in background thread"""
    try:
        job = job_manager.get_job(job_id)
        if not job:
            return

        input_json_path = job.get('input_json_path')
        config_ini_path = job.get('config_ini_path')
        output_dir = job.get('output_dir')

        if not input_json_path or not config_ini_path or not output_dir:
            raise RuntimeError("Job paths not initialized")

        job_manager.update_job(job_id, JobStatus.PROCESSING, progress=25)

        # Step 1: Convert input.json to config.ini
        script_path = os.path.join(ACCESS_CONTROL_FOLDER, 'input.py')
        cwd = os.path.dirname(os.path.abspath(__file__))
        python_exe = get_python_executable()

        env = os.environ.copy()
        env[ENV_INPUT_JSON] = input_json_path
        env[ENV_CONFIG_INI] = config_ini_path
        env[ENV_OUTPUT_DIR] = output_dir
        # Avoid issues with native libs (OpenBLAS/MKL) in constrained runtimes.
        for k in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            env.setdefault(k, "1")

        subprocess.run(
            [python_exe, script_path],
            check=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=PROCESS_TIMEOUT,
            env=env,
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
            timeout=PROCESS_TIMEOUT,
            env=env,
        )

        # NOTE: plot/error-summary generation is intentionally NOT run here.
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
    """Return example JSON (dataset/input5.json)"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    example_path = os.path.join(base_dir, 'dataset', 'input5.json')

    try:
        with open(example_path, 'r', encoding='utf-8') as f:
            example_json = json.load(f)
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to load example JSON: {str(e)}'
        }), 500

    # Return as formatted JSON string to preserve consistent formatting
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
        # Best-effort cleanup before allocating new storage.
        cleanup_job_folders(
            max_age_hours=JOBS_MAX_AGE_HOURS,
            max_total_mb=JOBS_MAX_TOTAL_MB,
        )

        # Create background job
        job_id = job_manager.create_job()
        paths = job_paths(job_id)
        os.makedirs(paths['uploads_dir'], exist_ok=True)
        os.makedirs(paths['output_dir'], exist_ok=True)

        # Save upload into per-job sandbox
        file.save(paths['input_json_path'])

        job_manager.set_job_paths(
            job_id=job_id,
            work_dir=paths['work_dir'],
            input_json_path=paths['input_json_path'],
            config_ini_path=paths['config_ini_path'],
            output_dir=paths['output_dir'],
        )

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
        # Best-effort cleanup before allocating new storage.
        cleanup_job_folders(
            max_age_hours=JOBS_MAX_AGE_HOURS,
            max_total_mb=JOBS_MAX_TOTAL_MB,
        )

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

        # Create background job
        job_id = job_manager.create_job()
        paths = job_paths(job_id)
        os.makedirs(paths['uploads_dir'], exist_ok=True)
        os.makedirs(paths['output_dir'], exist_ok=True)

        # Save JSON into per-job sandbox
        with open(paths['input_json_path'], 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)

        job_manager.set_job_paths(
            job_id=job_id,
            work_dir=paths['work_dir'],
            input_json_path=paths['input_json_path'],
            config_ini_path=paths['config_ini_path'],
            output_dir=paths['output_dir'],
        )

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
        response['downloads'] = {
            'bundle_zip': f'/download/{job_id}/bundle.zip',
            'files': {
                'output.json': f'/download/{job_id}?file=output.json',
                'rules_temp.txt': f'/download/{job_id}?file=rules_temp.txt',
                'ACM.txt': f'/download/{job_id}?file=ACM.txt',
                'access_data.txt': f'/download/{job_id}?file=access_data.txt',
            }
        }
    elif job['status'] == JobStatus.FAILED.value:
        response['error'] = job.get('error', 'Processing failed')
    elif job['status'] in [JobStatus.PROCESSING.value,
                           JobStatus.GENERATING.value]:
        response['message'] = 'Processing in progress...'

    return jsonify(response), 200


@app.route('/download/<job_id>')
def download_outputs(job_id):
    """
    Download a single output file from a job's outputs directory.

    - Default: output.json
    - Also supports other known output artifacts via `?file=...`
    """
    requested = (request.args.get('file') or 'output.json').strip()

    # Prevent path traversal and keep this endpoint scoped to outputs/.
    if not requested or os.path.basename(requested) != requested:
        return "Invalid file name.", 400

    # Allow only known artifacts to avoid exposing arbitrary files.
    allowed_files = {
        'output.json',
        'rules_temp.txt',
        'ACM.txt',
        'access_data.txt',
    }
    if requested not in allowed_files:
        return "Requested file is not available.", 404

    job = job_manager.get_job(job_id)
    if not job:
        return "Job not found.", 404
    if job.get('status') != JobStatus.COMPLETED.value:
        return "Job is not completed yet.", 409

    outputs_dir = job.get('output_dir') or os.path.join(
        JOBS_FOLDER, job_id, 'outputs'
    )
    file_path = os.path.join(outputs_dir, requested)
    if not os.path.exists(file_path):
        return (
            f"Error: {requested} not found. "
            "Ensure the process completed successfully.",
            404,
        )

    return send_from_directory(outputs_dir, requested)


@app.route('/download/<job_id>/bundle.zip')
def download_bundle(job_id):
    """
    Download a zip containing:
      - job outputs/* (output.json, rules_temp.txt, ACM.txt, access_data.txt)
    """
    job = job_manager.get_job(job_id)
    if not job:
        return "Job not found.", 404
    if job.get('status') != JobStatus.COMPLETED.value:
        return "Job is not completed yet.", 409

    outputs_dir = job.get('output_dir') or os.path.join(
        JOBS_FOLDER, job_id, 'outputs'
    )

    mem = io.BytesIO()
    with zipfile.ZipFile(
        mem, mode='w', compression=zipfile.ZIP_DEFLATED
    ) as zf:
        if os.path.isdir(outputs_dir):
            allowed_files = {
                'output.json',
                'rules_temp.txt',
                'ACM.txt',
                'access_data.txt',
            }
            for name in sorted(allowed_files):
                abs_path = os.path.join(outputs_dir, name)
                if not os.path.exists(abs_path):
                    continue
                # Make the zip contain files under outputs/
                zf.write(abs_path, os.path.join('outputs', name))

    mem.seek(0)
    return send_file(
        mem,
        mimetype='application/zip',
        as_attachment=True,
        download_name='abac_outputs.zip'
    )


if __name__ == '__main__':
    app.run(debug=True)
