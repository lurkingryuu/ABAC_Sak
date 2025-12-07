from flask import Flask, render_template, request, send_from_directory, jsonify
import os
import subprocess
import sys
import shutil

app = Flask(__name__)


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


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'}), 400

    if not file.filename.endswith('.json'):
        error_msg = 'Invalid file type. Please upload a JSON file.'
        return jsonify({'success': False, 'error': error_msg}), 400

    try:
        input_path = os.path.join(
            app.config['UPLOAD_FOLDER'], 'input.json'
        )
        file.save(input_path)

        # Step 1: Convert input.json to config.ini
        try:
            script_path = os.path.join(
                ACCESS_CONTROL_FOLDER, 'input.py'
            )
            cwd = os.path.dirname(os.path.abspath(__file__))
            python_exe = get_python_executable()
            subprocess.run(
                [python_exe, script_path],
                check=True,
                cwd=cwd,
                capture_output=True,
                text=True
            )
        except subprocess.CalledProcessError as e:
            error_msg = f"Failed to process input.json: {e.stderr or str(e)}"
            return jsonify({'success': False, 'error': error_msg}), 500
        except Exception as e:
            error_msg = f"Unexpected error during input processing: {str(e)}"
            return jsonify({'success': False, 'error': error_msg}), 500

        # Step 2: Run gen.py to generate outputs
        try:
            script_path = os.path.join(
                ACCESS_CONTROL_FOLDER, 'gen.py'
            )
            cwd = os.path.dirname(os.path.abspath(__file__))
            python_exe = get_python_executable()
            subprocess.run(
                [python_exe, script_path],
                check=True,
                cwd=cwd,
                capture_output=True,
                text=True
            )
        except subprocess.CalledProcessError as e:
            error_msg = f"Failed to generate outputs: {e.stderr or str(e)}"
            return jsonify({'success': False, 'error': error_msg}), 500
        except Exception as e:
            error_msg = f"Unexpected error during output generation: {str(e)}"
            return jsonify({'success': False, 'error': error_msg}), 500

        return jsonify({
            'success': True,
            'message': 'Files generated successfully!'
        }), 200
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'An unexpected error occurred: {str(e)}'
        }), 500


@app.route('/download')
def download_outputs():
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
