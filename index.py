from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)

app.config['CORS_HEADERS'] = 'Content-Type'
CORS(app, resources={r"/*": {"origins": "*"}}) 

@app.route('/receive_post', methods=['POST'])
def receive_post():
    data = request.get_json()
    received_data = data['data']

    with open('received_data.txt', 'w') as file:
        file.write(received_data)

    print(received_data)
    return jsonify({'message': received_data}), 200

@app.route('/send_text', methods=['GET'])
def send_text():
    with open('received_data.txt', 'r') as file:
      file_contents = file.read()
    return jsonify({'message': file_contents}), 200

if __name__ == '__main__':
    app.run(port=5000, debug=True)