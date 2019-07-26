
import matplotlib.pyplot as plt

def plot(Model_name):
    # plot the training loss and accuracy
    N = np.arange(0, 200)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, H.history["loss"], label="train_loss")
    plt.plot(N, H.history["val_loss"], label="val_loss")
    plt.plot(N, H.history["acc"], label="train_acc")
    plt.plot(N, H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy "+ Model_name +"_"+str(Learning_rate))
    plt.xlabel("Epoch #")
    plt.ylabel("Loss\\Accuracy")
    plt.legend()

    plt.savefig('Plots\\'+Model_name)

