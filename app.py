from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import subprocess

app = Flask(__name__,static_folder='static')

@app.route('/')
def index():
    return render_template('index.html')

# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'fileInput' not in request.files:
#         return jsonify({'error': 'No file provided'}), 400

#     file = request.files['fileInput']

#     if file.filename == '':
#         return jsonify({'error': 'No selected file'}), 400

#     if file:
#         # Save the file in a secure way
#         filename = secure_filename(file.filename)
#         file_path = os.path.join('uploads', filename)
#         file.save(file_path)

#         # Run your Python script with the uploaded file
#         subprocess.run(['python', 'pdf_parser.py', file_path])

#         # Read the output file and send it to the client
#         with open('output.txt', 'r') as output_file:
#             output_content = output_file.read()

#         # return output_content, 1000
#         return jsonify({'output': output_content}), 200

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import send_file

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'fileInput' not in request.files:
        return jsonify({'error': 'No file provided', 'files':request.files}), 400

    file = request.files['fileInput']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        # Save the file in a secure way
        filename = secure_filename(file.filename)
        file_path = os.path.join('uploads', filename)
        file.save(file_path)

        # Run your Python script with the uploaded file
        subprocess.run(['python', 'pdf_parser.py', file_path])

        # Send the output file as a download
        return send_file('output.txt', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)

