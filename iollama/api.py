from flask import Flask
from flask import jsonify
from flask import request
from flask_cors import CORS
import logging
import sys
from model import *
from config import *

app = Flask(__name__)
CORS(app)

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@app.route('/api/question', methods=['POST'])
def post_question():
    json = request.get_json(silent=True)
    question = json['question']
    user_id = json['user_id']
    logging.info("post question `%s` for user `%s`", question, user_id)

    resp = chat(question, user_id)
    data = {'answer':resp}

    return jsonify(data), 200

if __name__ == '__main__':
    init_llm()
    index = init_index(Settings.embed_model)
    init_query_engine(index)

    app.run(host='0.0.0.0', port=HTTP_PORT, debug=True)
