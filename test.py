from loaddata import load_data
from cnn_model import cnn_model_structure
import glob
import numpy as np

def test_model():
    # some declared variables
    inputImageShape = (224, 224, 3)
    num_of_output_classes = 2
    _, testX, _, testY = load_data()
    model = cnn_model_structure(input_shape=inputImageShape, num_classes=num_of_output_classes)
    weights = ''
    for w in glob.glob('models\\*.h5'):
        weights = w
    model.load_weights(weights)
    eval = model.predict(testX)
    out_class=np.array([np.argmax(out) for out in eval])
    ref_class = np.array([np.argmax(out) for out in testY])
    print(out_class)
    print(ref_class)
    print('Acc = '+str((1-float(np.count_nonzero(ref_class-out_class))/float(len(ref_class)))*100))