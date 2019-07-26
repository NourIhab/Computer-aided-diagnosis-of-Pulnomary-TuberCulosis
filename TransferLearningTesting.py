from keras.models import model_from_json, Model
import numpy as np
import TransferLearning

# evaluate the network
json_file = open('Models\\' + TransferLearning.Model_name + '\\model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

model = model_from_json(loaded_model_json)
# load weights into new model

def testing():
    print("[INFO] evaluating network...")
    model.load_weights(('Models\\' + TransferLearning.Model_name + '\\cp-0200.' + str(TransferLearning.Learning_rate) + '.ckpt'))
    predictions = model.predict(TransferLearning.testX)
    report = TransferLearning.classification_report(TransferLearning.testY.argmax(axis=1), predictions.argmax(axis=1))
    print(report)
    TransferLearning.classification_report_csv(report, 'test\\' + TransferLearning.Model_name + '\\' + str(TransferLearning.Learning_rate))

    result = TransferLearning.decimalDecoding(predictions)
    print(result)
    ref_result = TransferLearning.decimalDecoding(TransferLearning.testY)
    print(ref_result)
    accuracy = 100 * (1 - float(np.count_nonzero(result - ref_result)) / float(len(result)))
    print(' Transfer learning Accuracy = ' + str(accuracy))

testing()