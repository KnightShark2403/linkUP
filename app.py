from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
from linkedin_profile_analyzer import LinkedInProfileAnalyzer

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Log the file details
        print(f"File received: {filename}, size: {os.path.getsize(file_path)} bytes")
        
        job_role = request.form.get('job_role', 'Software Engineer')
        
        analyzer = LinkedInProfileAnalyzer()
        try:
            results = analyzer.analyze_profile(file_path, job_role)
        except Exception as e:
            print(f"Error analyzing PDF: {e}")
            return jsonify({"error": "Error analyzing PDF. Please try again."}), 500
        
        # Clean up the uploaded file after analysis
        os.remove(file_path)
        
        return jsonify(results)
    else:
        return jsonify({"error": "Invalid file type"}), 400