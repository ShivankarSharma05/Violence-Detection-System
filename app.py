from flask import Flask, jsonify
import subprocess
import os

app = Flask(__name__)

@app.route('/run-ai', methods=['GET'])
def run_ai():
    try:
        # Absolute path (IMPORTANT for reliability)
        script_path = os.path.join(os.getcwd(), "detect.py")

        # Run AI script
        subprocess.Popen(["python", script_path])

        return jsonify({"status": "AI started successfully"})
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(port=5000, debug=True)