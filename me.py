import pickle
import cv2
import face_recognition
import firebase_admin
from firebase_admin import credentials, storage, db
import os
from flask import Flask, request, jsonify
import base64
import numpy as np
import uuid 
import asyncio
import websockets
import spacy
import pandas as pd

import requests
from bs4 import BeautifulSoup
import wikipediaapi
from datetime import datetime # Import the uuid module

ALLOWED_EXTENSIONS = {'jpg', 'png', 'jpeg'}
app = Flask(__name__)

# Initialize Firebase
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://face-recognition-system-ebb56-default-rtdb.firebaseio.com/",
    'storageBucket': "face-recognition-system-ebb56.appspot.com"
})
bucket = storage.bucket(app=firebase_admin.get_app())
ref = db.reference('/registered_faces')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_registered(face_encoding, tolerance=0.3):
    registered_faces = ref.get() or {}

    for outer_key, outer_value in registered_faces.items():
        for inner_key, inner_value in outer_value.items():
            registered_face_encoding = inner_value.get('encoding')

            if registered_face_encoding is None or face_encoding is None:
                continue

            results = face_recognition.compare_faces([registered_face_encoding], face_encoding, tolerance=tolerance)

        # print(f"Results for {inner_key}: {results}")

            if results and results[0]:
                return inner_value  # Return the entire inner dictionary when a match is found

    return None  # Return None if no match is found
  # Return None if no match is found


def decode_base64_image(base64_string):
    encoded_data = base64_string.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def save_base64_image(base64_string, local_image_path):
    encoded_data = base64_string.split(',')[1]
    with open(local_image_path, 'wb') as f:
        f.write(base64.b64decode(encoded_data))

@app.route('/home')
def home():
    return "home"

MAX_BASE64_LENGTH= float('inf')
@app.route('/api/check-face', methods=['POST'])
def check_or_register_face():
    try:


        request_secret_key = request.headers.get("token")
        if not request_secret_key:
            return jsonify({'status': False, 'message': "Secret key required!!"}), 400
        # Retrieve data from the form
        base64_image = request.form.get('base64_image')
        name = request.form.get('name')
        age = request.form.get('age')
        if len(base64_image) > MAX_BASE64_LENGTH:
            return jsonify({'status': False, 'message': f"Base64 image length exceeds the allowed limit of {MAX_BASE64_LENGTH} bytes"}), 400
        if not base64_image:
            return jsonify({'status': False, 'message': "Please provide a base64-encoded image in the 'base64_image' field"}), 400

        raw_picture = decode_base64_image(base64_image)
        raw_picture_rgb = cv2.cvtColor(raw_picture, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(raw_picture_rgb)

        if face_encodings and len(face_encodings) > 0:
            raw_face_encoding = face_encodings[0]
            unique_id = str(uuid.uuid4()) 
            # print(raw_face_encoding) # Generate a unique ID

            registered_id = is_registered(raw_face_encoding)
            # print(registered_id.items('name'))
            if registered_id:
                
                user_name = registered_id.get('name')
                user_age = registered_id.get('age')
                image_path=registered_id.get('image_path')
                data = [
                    {'name': user_name, 'age': user_age, 'image_path': image_path}
                ]

                return jsonify({
                    "message": f"Welcome!! {user_name}",
                    'status': True,
                    "details": data
                }), 200
            else:
                if not name or not age:
                    return jsonify({'status': False, 'message': "You are not Registered!!! Please Provide Name and Age"}), 400

                local_image_path = os.path.join('images', f"{unique_id}.jpg")
                save_base64_image(base64_image, local_image_path)

                image = face_recognition.load_image_file(local_image_path)
                face_encodings = face_recognition.face_encodings(image)

                if not face_encodings:
                    return jsonify({'status': False, 'message': "No face found in the provided image"}), 400

                encodings_directory = os.path.join('encodings')
                os.makedirs(encodings_directory, exist_ok=True)

                firebase_safe_filename = os.path.basename(local_image_path).replace('.', '_')
                encodings_file_path = os.path.join(encodings_directory, f'{firebase_safe_filename}_encodings.p')
                with open(encodings_file_path, 'wb') as encodings_file:
                    pickle.dump(face_encodings, encodings_file)

                storage_image_path = f"images/{firebase_safe_filename}"
                storage_blob = bucket.blob(storage_image_path)
                storage_blob.upload_from_filename(local_image_path)

                new_registration = {
                    'name': name,
                    'age': age,
                    'encoding': face_encodings[0].tolist(),
                    'image_path': storage_image_path,
                    'encodings_file_path': encodings_file_path
                }

                ref.child(request_secret_key).child(unique_id).set(new_registration)

                # print("Location of the stored base64 image:", local_image_path)

                return jsonify({'is_registred':True,'status': True,
                                'message': f"Registration successful! Your unique ID is: {unique_id}"}), 201

        else:
            return jsonify({
                'status': False,
                'message': "No face found in the provided image"
            }), 404

    except Exception as e:
        return jsonify({
            'status': False,
            'message': f"Error processing image: {str(e)}"
        }), 500
    

questions_file_path = 'questions.xlsx'

nlp = spacy.load("en_core_web_sm")

def read_qa_pairs(filename):
    qa_df = pd.read_excel(filename)
    qa_df['Keywords'] = qa_df['Questions'].apply(generate_keywords)
    qa_pairs = {row['Questions']: {'Answer': row['Answers'], 'Keywords': row['Keywords']} for _, row in qa_df.iterrows()}
    return qa_pairs

def generate_keywords(question):
    doc = nlp(question)

    while len(doc) == 1:
        return [token.text.lower() for token in doc if token.is_alpha and len(token.text) > 1]

    return [token.text.lower() for token in doc if not token.is_stop and token.is_alpha and len(token.text) > 1]

qa_pairs = read_qa_pairs(questions_file_path)

from openpyxl import load_workbook

def add_qa_pair(question, answer, filename='questions.xlsx'):
    try:
        if question.lower() in [existing_question.lower() for existing_question in qa_pairs.keys()]:
            confirmation = request.form.get('confirmation', '')
            if confirmation.lower() != 'yes':
                return jsonify({'message': 'Question already exists. Aborted the process.', 'status': False, 'status_code': 400})

        try:
            qa_df = pd.read_excel(filename)
        except FileNotFoundError:
            qa_df = pd.DataFrame(columns=['Questions', 'Answers', 'Keywords'])

        keywords = generate_keywords(question)
        new_qa_df = pd.DataFrame({'Questions': [question], 'Answers': [answer], 'Keywords': [keywords]})
        qa_df = pd.concat([qa_df, new_qa_df], ignore_index=True)

        with pd.ExcelWriter(filename, engine='openpyxl', mode='w') as writer:
            qa_df.to_excel(writer, index=False, header=True, sheet_name='Sheet1')

        return jsonify({'message': 'Question-Answer pair added successfully!', 'status': True, 'status_code': 200})
    except Exception as e:
        return jsonify({'message': f'Error: {e}', 'status': False, 'status_code': 500})

@app.route('/api/add', methods=['POST'])
def add_qa_pair_route():
    question = request.form.get('question', '')
    answer = request.form.get('answer', '')
    response = add_qa_pair(question, answer)
    return response

wiki_wiki = wikipediaapi.Wikipedia(language='en', user_agent='sona-ai')

async def handle_message(websocket, path):
    async for message in websocket:
        response = process_message(message)
        await websocket.send(response)

def process_message(message):
    doc = nlp(message)

    if any(token.text.lower() in ['date', 'time'] for token in doc):
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return jsonify({'data': f'Today\'s date and time: {current_datetime}', 'status': True, 'status_code': 200})

    matching_questions = [question for question, data in qa_pairs.items() if set(data['Keywords']) == set(token.text.lower() for token in doc if not token.is_stop and token.is_alpha)]

    if not matching_questions:
        matching_questions = [question for question, data in qa_pairs.items() if any(keyword in message.lower() for keyword in data['Keywords'])]

    if matching_questions:
        response = {'data': qa_pairs[matching_questions[0]]['Answer'], 'status': True, 'status_code': 200}
    else:
        response = wikipedia_answer(message)

    return jsonify(response)

def wikipedia_answer(query):
    try:
        page_py = wiki_wiki.page(query)
        result_text = page_py.summary[:500]
        print(result_text)
        
        return {'data': result_text, 'status': True, 'status_code': 200}
    except wikipediaapi.exceptions.DisambiguationError as e:
        options = e.options[:3]
        return {'data': f"Multiple matching pages found: {', '.join(options)}", 'status': False, 'status_code': 404}
    except wikipediaapi.exceptions.HTTPTimeoutError as e:
        return {'data': "Wikipedia request timed out. Please try again later.", 'status': False, 'status_code': 500}
    except Exception as e:
        print(f"Error during Wikipedia search: {e}")
        return {'data': "I encountered an error while searching Wikipedia.", 'status': False, 'status_code': 500}

@app.route('/api/chat', methods=['POST'])
def api_chat():
    user_message = request.form.get('message')
    response = process_message(user_message)
    bot_response_data = response.get_json() if response.is_json else {}
    bot_response = bot_response_data.get('data', 'Error: No data in the response')
    return jsonify({'bot_response': bot_response, 'status': True, 'status_code': 200})

@app.route('/api/get', methods=['GET'])
def get_all_questions():
    return jsonify({'data': qa_pairs, 'status': True, 'status_code': 200})







if __name__ == '__main__':
    ws_server = websockets.serve(handle_message, "localhost", 5000)
    flask_server = asyncio.gather(
        ws_server,
        app.run(host='localhost', port=5000, debug=True)
    )

    print("Combined server is running. Waiting for connections...")

    try:
        asyncio.get_event_loop().run_until_complete(flask_server)
        asyncio.get_event_loop().run_forever()
    except KeyboardInterrupt:
        print("Server stopped.")