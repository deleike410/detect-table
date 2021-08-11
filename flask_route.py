from flask import Flask, render_template
from flask import request
import json
import base64
import os

from table_line_create import gene_json

app = Flask(__name__)


@app.route('/tableline/index')
def hello_world():
    return 'Hello, World!'


@app.route("/tableline/line", methods=['POST'])
def get_frame():
    # 接收图片
    data = request.get_data()
    data = json.loads(data)
    file_name = data.get('file_name')
    res = gene_json(file_name)
    return res


if __name__ == '__main__':
    app.run(debug=True, port=8081)
