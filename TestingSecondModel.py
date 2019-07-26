from keras.models import model_from_json, Model
import numpy as np
import Second_Model


 # load json and create model
json_file = open('model_50.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# load weights into new model
model.load_weights('model_50.h5')

def testing():
    result = Second_Model.decimalDecoding(model.predict(Second_Model.testX))
    print(result)
    ref_result = Second_Model.decimalDecoding(Second_Model.testY)
    print(ref_result)
    testAccuarcy = 100 * (1 - float(np.count_nonzero(result - ref_result))/float(len(result)))
    print('The Accuarcy of the testing is = ' + str(testAccuarcy))

testing()