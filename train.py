import numpy as np
import wandb
from net import Net
import data
from alive_progress import alive_it as bar

def train(epochs=10000, batch_size=256):
    images, labels = data.get_training_files()
    images = images / 255 #Normalize
    one_hot_labels = data.convert_to_one_hot(labels)
    net = Net()

    try:
        for _ in bar(range(epochs)):
            random_vector = np.random.randint(0, len(labels)-1, size=batch_size)

            loss, out = net.train(images[random_vector, None, :, :], one_hot_labels[:, random_vector])

            accuracy = np.sum((np.argmax(out, axis=0) == labels[random_vector]) * 1) / batch_size * 100

            print(f"Loss:{loss}, Accuracy:{accuracy}%")

    except KeyboardInterrupt:
        random_vector = np.random.randint(0, len(labels)-1, size=batch_size)

        loss, out = net.train(images[random_vector, None, :, :], one_hot_labels[:, random_vector])

        accuracy = np.sum((np.argmax(out, axis=0) == labels[random_vector]) * 1) / batch_size * 100

        print(f"Loss:{loss}, Accuracy:{accuracy}%")

        raise ValueError

if __name__ == '__main__':
    train(epochs=5000, batch_size=512)