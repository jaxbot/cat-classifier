import tensorflow as tf, sys
from os import listdir
from shutil import copyfile
import json

input_path = "input_batch"
output_path = "batch_output"

label_lines = [line.rstrip() for line in tf.gfile.GFile("/tf_files/retrained_labels.txt")]

with tf.gfile.FastGFile("/tf_files/retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    last_results = {}
    last_sequence_id = 0
    images = []

    feeding_sessions = []
    image_results = {}

    for f in listdir(input_path):
        print('%s processing: ' % f)
        seq = f.split("-")[0]

        if seq != last_sequence_id:
            averages = {}
            highest = 0
            highest_key = ''

            if len(images) > 0:
                for key in last_results:
                    averages[key] = sum(last_results[key]) / len(last_results[key])
                    if averages[key] > highest:
                        highest = averages[key]
                        highest_key = key

                print('Winner: ' + highest_key)
                print(averages)

                thumbnail = images[len(images) / 2]

                for image in images:
                    copyfile(input_path + '/' + image, output_path + '/' + highest_key + '/' + image)

                feeding_sessions.append({'winner': highest_key, 'thumbnail': thumbnail, 'averages': averages})

            last_sequence_id = seq
            last_results = {}
            images = []

        with open(output_path + '/classification.json', 'w') as outfile:
            json.dump(image_results, outfile)
        with open(output_path + '/feedings.json', 'w') as outfile:
            json.dump(feeding_sessions, outfile)

        try:
            image_data = tf.gfile.FastGFile(input_path + '/' + f, 'rb').read()
            softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
            predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
            
            top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
            images.append(f)

            image_results[f] = {}

            for node_id in top_k:
                human_string = label_lines[node_id]
                score = predictions[0][node_id]

                if human_string not in last_results:
                    last_results[human_string] = []
                last_results[human_string].append(score)
                image_results[f][human_string] = str(score)
                print('%s (score = %.5f' % (human_string, score))

        except:
            print('%s failed!')
