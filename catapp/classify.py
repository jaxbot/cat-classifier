import tensorflow as tf, sys
from os import listdir
from shutil import copyfile

image_path = sys.argv[1]



label_lines = [line.rstrip() for line in tf.gfile.GFile("/tf_files/retrained_labels.txt")]

with tf.gfile.FastGFile("/tf_files/retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    last_results = {}
    last_sequence_id = 0
    images = []
    for f in listdir(image_path):
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

                for image in images:
                    copyfile(image_path + '/' + image, '/tf_files/test_output/' + highest_key + '/' + image)

            last_sequence_id = seq
            last_results = {}
            images = []

        try:
            image_data = tf.gfile.FastGFile(image_path + '/' + f, 'rb').read()
            softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
            predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
            
            top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
            images.append(f)

            for node_id in top_k:
                human_string = label_lines[node_id]
                score = predictions[0][node_id]

                if human_string not in last_results:
                    last_results[human_string] = []
                last_results[human_string].append(score)
                print('%s (score = %.5f' % (human_string, score))

        except:
            print('%s failed!')
