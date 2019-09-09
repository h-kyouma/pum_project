from model.data.model import ATTRIBS, DataFrameSelector
import joblib
import pandas as pd
import socket


def input_message_parser(message):
    words = message.split(',')
    moments = words[:-14]

    features = [moments[0][15:]]
    for moment in moments[1:]:
        if '\n' not in moment:
            if 'u' in moment:
                features.append(moment[6:])
            else:
                features.append(moment[5:])
        else:
            if 'u' in moment:
                features.append(moment[7:])
            else:
                features.append(moment[6:])

    features = [float(feature) for feature in features]
    [features.append(float(word)) for word in words[-13:-8]]

    cnt_area_cnt_len_ratio = features[-5]/features[-4]
    hull_area_hull_len_ratio = features[-3]/features[-2]
    hull_area_cnt_area_ratio = features[-3]/features[-5]
    hull_len_cnt_len_ratio = features[-2]/features[-4]
    features.append(cnt_area_cnt_len_ratio)
    features.append(hull_area_hull_len_ratio)
    features.append(hull_area_cnt_area_ratio)
    features.append(hull_len_cnt_len_ratio)
    [features.append(float(word)) for word in words[-8:]]

    vertical_height = features[-6] - features[-8]
    horizontal_height = features[-2] - features[-4]
    vertical_to_horizontal_ratio = vertical_height/horizontal_height
    features.append(vertical_height)
    features.append(horizontal_height)
    features.append(vertical_to_horizontal_ratio)

    return features


receiver = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
receiver.bind((socket.gethostname(), 6666))
receiver.listen(5)

data_preprocessor = joblib.load('model/data/data_preprocessor.pkl')
model = joblib.load('model/data/best_model.pkl')

while True:
    clientSocket, address = receiver.accept()
    msg = clientSocket.recv(4096)
    decoded = msg.decode('utf-8')
    if len(decoded) > 0:
        features_df = pd.DataFrame(data=[input_message_parser(decoded)], columns=ATTRIBS)
        X = data_preprocessor.transform(features_df)
        label = model.predict(X)
        sender = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sender.connect((address[0], 7777))
        sender.send(str(label).encode('utf-8'))
