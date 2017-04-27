from flask import Flask, request, json
import tensorflow as tf, sys
from os import listdir
import os
from shutil import copyfile
from werkzeug.utils import secure_filename

app = Flask(__name__)

label_lines = [line.rstrip() for line in tf.gfile.GFile("/catapp/retrained_labels.txt")]

with tf.gfile.FastGFile("/catapp/retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

def classify_image(input_image_path):
    results = {}
    with tf.Session() as sess:
        image_data = tf.gfile.FastGFile(input_image_path, 'rb').read()
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})

        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]

            results[human_string] = str(score * 100)
    return results

@app.route('/api/classify', methods=['POST'])
def api_classify():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file:
        filename = secure_filename(file.filename)
        full_file_name = os.path.join('/tmp', filename)
        file.save(full_file_name)
        return json.dumps(classify_image(full_file_name))
    return '''500'''


@app.route('/')
def index():
        return app.send_static_file('index.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
