from net import Net
import data
from Hazelnut.utils import dataloader

def train(epochs=100, batch_size=256):
    images, labels = data.get_training_files()
    images = images / 255 #Normalize
    one_hot_labels = data.convert_to_one_hot(labels)
    net = Net()

    images = net.np.array(images[:, None, :, :])
    labels = net.np.array(labels)
    one_hot_labels = net.np.array(one_hot_labels).T
    last_epoch = 1

    try:
        for batch_images, batch_labels, epoch in dataloader(batch_size, epochs, images, one_hot_labels): 
            loss, out = net.train(batch_images, batch_labels.T)
            accuracy = float(net.np.sum((net.np.argmax(out, axis=0) == net.np.argmax(batch_labels, axis=1)) * 1) / batch_size * 100)
            print(f"Epoch: {epoch}, Loss:{loss}, Accuracy:{accuracy}%")

    except KeyboardInterrupt:
        raise ValueError

if __name__ == '__main__':
    train(epochs=50, batch_size=256)