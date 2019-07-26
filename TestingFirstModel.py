from keras.models import model_from_json, Model
import numpy as np
import first_model

    # load json and create model
json_file = open('model_00.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model_00.h5")


def testing():
    result = first_model.decimalDecoding(model.predict(first_model.testX))
    print(result)
    ref_result = first_model.decimalDecoding(first_model.testY)
    print(ref_result)
    acc = 100 * (1 - float(np.count_nonzero(result - ref_result))/float(len(result)))
    print('Acc = ' + str(acc))

testing()