from net import Net
import data
from Hazelnut.utils import dataloader
from alive_progress import alive_bar

def train(epochs=100, batch_size=256):
    images, labels = data.get_training_files()
    images = images / 255 #Normalize
    one_hot_labels = data.convert_to_one_hot(labels)
    net = Net()

    images = net.np.array(images[:, None, :, :])
    labels = net.np.array(labels)
    one_hot_labels = net.np.array(one_hot_labels).T

    batch = 1
    last_epoch = 0

    with alive_bar(epochs, dual_line=True) as bar:
        try:
            for batch_images, batch_labels, epoch in dataloader(batch_size, epochs, images, one_hot_labels):
                loss, out = net.train(batch_images, batch_labels.T)
                accuracy = float(net.np.sum((net.np.argmax(out, axis=0) == net.np.argmax(batch_labels, axis=1)) * 1) / batch_size * 100)
                batch += 1
                if last_epoch != epoch:
                    print(f"Epoch: {epoch}, Loss:{loss}, Accuracy:{accuracy}%")
                    print(net.np.argmax(out, axis=0))
                    last_epoch = epoch
                    batch = 1
                    bar()

                bar.text = f"Loss:{loss}, Accuracy:{accuracy}%, {batch} out of {len(images)//batch_size}"

        except KeyboardInterrupt:
            raise ValueError

if __name__ == '__main__':
    train(epochs=50, batch_size=512)