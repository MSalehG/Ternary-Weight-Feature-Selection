import scipy.io as sio
import numpy as np
import tensorflow as tf
from keras.datasets import fashion_mnist
from keras.datasets import mnist
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from Ternary_Layer import *

#To determine if we want to further train a saved model or start training a new model
Training = True

#To determine if we want to load an already trained model
Loading = False

#To determine if we want to save the model after working with it
Saving = True

#The quantizer used to quantize the weights as numpy arrays
def Clipper (w):
    threshold = 0.7 * np.sum(np.abs(w)) / float(w.size)
    return np.sign(np.sign(w + threshold) + np.sign(w - threshold))

#The dictionary to select which dataset we want to test
Dataset_dict = {1:'mnist', 2:'fashion', 3:'utk_face', 4:'channel', 5:'coil', 6:'isolet'}
Dataset_num = 2
Dataset = Dataset_dict[Dataset_num]

#The dictionary to select which saved model we want to load (if we wanted to load a model)
checkpoint_dict = {1:'paper/mnist_weights.h5', 2:'paper/fashion_weights_3.h5',
                   3:'paper/utk_face_weights.h5', 4:'paper/channel_weights.h5',
                   5: 'paper/coil_weights_3.h5', 6: 'paper/isolet_weights_1.h5'}

if Dataset == 'mnist':

    save_dir = os.path.join(os.getcwd(), 'paper')
    weight_name = 'mnist_weights.h5'
    log_dir = os.path.join(os.getcwd(), 'log')

    (x_trainval, _), (x_test, _) = mnist.load_data()
    x_trainval = x_trainval.astype('float32') / 255.
    x_trainval = x_trainval.reshape((len(x_trainval), 28 * 28 * 1))

    index = np.random.permutation(60000)
    pics = index[1: 5401]
    x_train = np.array([p for p in x_trainval[pics]])
    pics = index[5401: 6001]
    x_validate = np.array([p for p in x_trainval[pics]])

    x_test = x_test.astype('float32') / 255.
    x_test = x_test.reshape((len(x_test), 28 * 28 * 1))

    print(np.shape(x_train))
    print(np.shape(x_validate))
    print(np.shape(x_test))

    num_input = 784

elif Dataset == 'fashion':

    save_dir = os.path.join(os.getcwd(), 'paper')
    weight_name = 'fashion_weights_2.h5'
    log_dir = os.path.join(os.getcwd(), 'log')

    (x_trainval, _), (x_test, _) = fashion_mnist.load_data()
    x_trainval = x_trainval.astype('float32') / 255.
    x_trainval = x_trainval.reshape((len(x_trainval), 28 * 28 * 1))

    index = np.random.permutation(60000)
    pics = index[1: 5401]
    x_train = np.array([p for p in x_trainval[pics]])
    pics = index[5401: 6001]
    x_validate = np.array([p for p in x_trainval[pics]])

    x_test = x_test.astype('float32') / 255.
    x_test = x_test.reshape((len(x_test), 28 * 28 * 1))

    print(np.shape(x_train))
    print(np.shape(x_validate))
    print(np.shape(x_test))

    num_input = 784

elif Dataset == 'utk_face':

    save_dir = os.path.join(os.getcwd(), '')
    weight_name = 'utk_face_weightsG.h5'
    log_dir = os.path.join(os.getcwd(), 'log')

    MAT = sio.loadmat('..//UTKFace//UTKFace.mat')
    Channel = MAT['pix']
    Channel = np.reshape(Channel,[Channel.shape[0],Channel.shape[1]* Channel.shape[2]])
    np.random.shuffle(Channel)

    x_train = Channel[1:13001]
    x_validate = Channel[13001:16001]
    x_test = Channel[16001:23001]

    num_input = 2500

elif Dataset == 'channel':

    save_dir = os.path.join(os.getcwd(), 'paper')
    weight_name = 'channel_weightsG.h5'
    log_dir = os.path.join(os.getcwd(), 'log')

    #Noiseless data of the channel
    MAT = sio.loadmat('..//Channel//My_perfect_H_22.mat')
    Channel = MAT['My_perfect_H']
    Channel = np.reshape(Channel,[Channel.shape[0],Channel.shape[1]* Channel.shape[2]])

    #Noisy data of the channel
    MAT2 = sio.loadmat('..//Channel//My_noisy_H_22.mat')
    Channel2 = MAT2['My_noisy_H']
    Channel2 = np.reshape(Channel2,[Channel2.shape[0],Channel2.shape[1]* Channel2.shape[2]])

    #Noiseless data of the channel as the output
    y_train = np.real(Channel[1:28801])
    y_validate = np.real(Channel[28801:32001])
    y_test = np.real(Channel[32001:-1])

    #Noisy data of the channel as the input
    x_train = np.real(Channel2[1:28801])
    x_validate = np.real(Channel2[28801:32001])
    x_test = np.real(Channel2[32001:-1])

    y_train = y_train.astype('float32')/y_train.max()
    y_validate = y_validate.astype('float32')/y_validate.max()
    y_test = y_test.astype('float32')/y_test.max()

    x_train = x_train.astype('float32')/x_train.max()
    x_validate = x_validate.astype('float32')/x_validate.max()
    x_test = x_test.astype('float32')/x_test.max()

    num_input = 1008

elif Dataset == 'coil':

    save_dir = os.path.join(os.getcwd(), 'paper')
    weight_name = 'coil_weights_3.h5'
    log_dir = os.path.join(os.getcwd(), 'log')

    MAT = sio.loadmat('..//COIL-20//coil.mat')
    Channel = MAT['pix']
    Channel = np.reshape(Channel,[Channel.shape[0],Channel.shape[1]* Channel.shape[2]])
    np.random.shuffle(Channel)

    x_train = Channel[1:1037]
    x_validate = Channel[1037:1153]
    x_test = Channel[1153:-1]

    num_input = 400

elif Dataset == 'isolet':

    save_dir = os.path.join(os.getcwd(), 'paper')
    weight_name = 'isolet_weights_5.h5'
    log_dir = os.path.join(os.getcwd(), 'log')

    MAT = sio.loadmat('..//ISOLET//isolet.mat')
    Channel = MAT['pix']
    np.random.shuffle(Channel)

    x_train = Channel[1:5614]
    x_validate = Channel[5614:6238]
    x_test = Channel[6238:-1]

    num_input = 617


#The number of neurons in the layers of the autoencoder
F1 = 256


Latent = 64

#Since the channel data is ranged between -1 to 1 it needs leaky_relu and tanh
#activation in the architecture
if Dataset == 'channel':

    input_img = tf.keras.Input(shape=(num_input,), name='Input')
    encoded = MyLayer(num_input, kernel_constraint=Weights(), name='FS_Layer')(input_img)
    encoded = tf.keras.layers.Dense(F1, name='Encoder_Dense1')(encoded)
    encoded = tf.keras.layers.LeakyReLU(alpha=0.1, name='Encdoer_Activation1')(encoded)

    latent = tf.keras.layers.Dense(Latent, name='Latent_Space')(encoded)
    latent = tf.keras.layers.LeakyReLU(alpha=0.5, name='Latent_Space_Activation')(latent)

    decoded = tf.keras.layers.Dense(F1, name='Decoder_Dense2')(latent)
    decoded = tf.keras.layers.LeakyReLU(alpha=0.1, name='Decdoer_Activation2')(decoded)

    out = tf.keras.layers.Dense(num_input, activation='tanh', name='Output')(decoded)

elif Dataset == 'isolet':

    input_img = tf.keras.Input(shape=(num_input,), name='Input')
    encoded = MyLayer(num_input, kernel_constraint=Weights(), name='FS_Layer')(input_img)
    encoded = tf.keras.layers.Dense(F1, name='Encoder_Dense1')(encoded)
    encoded = tf.keras.layers.LeakyReLU(alpha=0.1, name='Encdoer_Activation1')(encoded)

    latent = tf.keras.layers.Dense(Latent, name='Latent_Space')(encoded)
    latent = tf.keras.layers.LeakyReLU(alpha=0.5, name='Latent_Space_Activation')(latent)

    decoded = tf.keras.layers.Dense(F1, name='Decoder_Dense2')(latent)
    decoded = tf.keras.layers.LeakyReLU(alpha=0.1, name='Decdoer_Activation2')(decoded)

    out = tf.keras.layers.Dense(num_input, activation='tanh', name='Output')(decoded)

else:
    input_img = tf.keras.Input(shape=(num_input,), name='Input')
    encoded = MyLayer(num_input, kernel_constraint=Weights(), name='FS_Layer')(input_img)
    encoded = tf.keras.layers.Dense(F1, activation='relu', name='Encoder_Dense1')(encoded)

    latent = tf.keras.layers.Dense(Latent, name='Latent_Space', activation='relu')(encoded)

    decoded = tf.keras.layers.Dense(F1, activation='relu', name='Decoder_Dense2')(latent)

    out = tf.keras.layers.Dense(num_input, activation='sigmoid', name='Output')(decoded)

autoencoder = tf.keras.Model(input_img, out)
autoencoder.summary()

#The factor of the L1 regularizer
landa = 1e-10
#The learning rate of the optimizer
LR = 1e-3

#The custom loss of the network
def FSLoss(w):

    def MSELOSS(y_true, y_pred):
        mse_loss = tf.keras.losses.mse(y_true, y_pred)
        l1 = landa * tf.keras.backend.sum(tf.keras.backend.abs(w))
        loss = mse_loss + l1
        return loss
    return MSELOSS

#The custom metrics of the network to be measured each epoch
def WMetric(w):

    def NonZeroW(y_true, y_pred):
        weight = tfquantizer(w)
        return tf.math.count_nonzero(weight)
    return NonZeroW

#Another custom metric to be measured each epoch
def L1Loss(w):
    def L1(y_true, y_pred):
        return landa * tf.keras.backend.sum(tf.keras.backend.abs(w))
    return L1


autoencoder.compile(loss=FSLoss(tf.global_variables()[0]),
                    optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
                    metrics=['mse', L1Loss(tf.global_variables()[0]), WMetric(tf.global_variables()[0])])

if Loading == True:
    autoencoder.load_weights(checkpoint_dict[Dataset_num])



# tensorboard = tf.keras.callbacks.TensorBoard(log_dir=(log_dir + str(time())))
if (Dataset == 'mnist' or Dataset == 'fashion') and Training == True:

    iter = 0
    training_losses = list()
    training_mses = list()
    validation_losses = list()
    validation_mses = list()
    l1s = list()
    wns = list()
    while (iter < 5000):
        iter = iter + 1
        print(iter)
        hist = autoencoder.fit(x_train, x_train,
                        epochs=1, verbose=1,
                        batch_size=256,
                        shuffle=True,
                        validation_data=(x_test, x_test))

        history = hist.history
        tloss = history['loss'][0]
        tmse = history['mean_squared_error'][0]
        vloss = history['val_loss'][0]
        vmse = history['val_mean_squared_error'][0]
        l1 = history['L1'][0]
        wn = history['NonZeroW'][0]

        training_mses.append(tmse)
        validation_mses.append(vmse)
        l1s.append(l1)
        wns.append(wn)

        # Stop the training if we've already selected the 50 pixels
        if wn < 51 or wn == 50:
            break

elif Dataset == 'utk_face' and Training == True:

    iter = 0
    training_losses = list()
    training_mses = list()
    validation_losses = list()
    validation_mses = list()
    l1s = list()
    wns = list()
    while (iter < 3000):
        iter = iter + 1
        print(iter)
        hist = autoencoder.fit(x_train, x_train,
                               epochs=1, verbose=1,
                               batch_size=256,
                               shuffle=True,
                               validation_data=(x_validate, x_validate))

        history = hist.history
        tmse = history['mean_squared_error'][0]
        vmse = history['val_mean_squared_error'][0]
        l1 = history['L1'][0]
        wn = history['NonZeroW'][0]

        training_mses.append(tmse)
        validation_mses.append(vmse)
        l1s.append(l1)
        wns.append(wn)

        #Stop the training if we've already selected the 50 pixels
        if wn < 51 or wn == 50:
            break

elif Dataset == 'channel' and Training == True:

    iter = 0
    training_losses = list()
    training_mses = list()
    validation_losses = list()
    validation_mses = list()
    l1s = list()
    wns = list()
    while (iter < 3000):
        iter = iter + 1
        print(iter)
        hist = autoencoder.fit(x_train, y_train,
                               epochs=1, verbose=1,
                               batch_size=256,
                               shuffle=True,
                               validation_data=(x_validate, y_validate))

        history = hist.history
        tmse = history['mean_squared_error'][0]
        vmse = history['val_mean_squared_error'][0]
        l1 = history['L1'][0]
        wn = history['NonZeroW'][0]

        training_mses.append(tmse)
        validation_mses.append(vmse)
        l1s.append(l1)
        wns.append(wn)

        # Stop the training if we've already selected the 48 pilots
        if wn < 49 or wn == 48:
            break

elif Dataset == 'coil' and Training == True:

    iter = 0
    training_losses = list()
    training_mses = list()
    validation_losses = list()
    validation_mses = list()
    l1s = list()
    wns = list()
    while (iter < 6000):
        iter = iter + 1
        print(iter)
        hist = autoencoder.fit(x_train, x_train,
                               epochs=1, verbose=1,
                               batch_size=64,
                               shuffle=True,
                               validation_data=(x_validate, x_validate))

        history = hist.history
        tmse = history['mean_squared_error'][0]
        vmse = history['val_mean_squared_error'][0]
        l1 = history['L1'][0]
        wn = history['NonZeroW'][0]

        training_mses.append(tmse)
        validation_mses.append(vmse)
        l1s.append(l1)
        wns.append(wn)

        #Stop the training if we've already selected the 50 pixels
        if wn < 51 or wn == 50:
            break

elif Dataset == 'isolet' and Training == True:

    iter = 0
    training_losses = list()
    training_mses = list()
    validation_losses = list()
    validation_mses = list()
    l1s = list()
    wns = list()
    while (iter < 6000):
        iter = iter + 1
        print(iter)
        hist = autoencoder.fit(x_train, x_train,
                               epochs=1, verbose=1,
                               batch_size=128,
                               shuffle=True,
                               validation_data=(x_validate, x_validate))

        history = hist.history
        tmse = history['mean_squared_error'][0]
        vmse = history['val_mean_squared_error'][0]
        l1 = history['L1'][0]
        wn = history['NonZeroW'][0]

        training_mses.append(tmse)
        validation_mses.append(vmse)
        l1s.append(l1)
        wns.append(wn)

        #Stop the training if we've already selected the 50 pixels
        if wn < 51 or wn == 50:
            break

# Score trained model.
if Dataset == 'channel':
    scores = autoencoder.evaluate(x_test, y_test, verbose=1)
else:
    scores = autoencoder.evaluate(x_test, x_test, verbose=1)
print('Test loss: ', scores[0])

if Training == True:
    Training_Mses = np.asarray(training_mses)
    Validation_Mses = np.asarray(validation_mses)
    L1s = np.asarray(l1s)
    Wns = np.asarray(wns)
    epoch = np.linspace(1, iter, iter)

    plt.figure()
    plt.plot(epoch, Training_Mses, 'b')
    plt.plot(epoch, Validation_Mses, 'r')
    plt.xlabel('Epoch')
    plt.ylabel('Reconstruction MSE')
    plt.title('Training vs Validation Reconstruction Mean Squared Error')
    plt.legend(('Training MSE', 'Validation MSE'))
    plt.show()


    plt.figure()
    plt.plot(epoch, L1s)
    plt.xlabel('Epoch')
    plt.ylabel('L1')
    plt.title('L1 Loss At Each Epoch')
    plt.show()

    plt.figure()
    plt.plot(epoch, Wns)
    plt.xlabel('Epoch')
    plt.ylabel('Number of Non Zero Weights')
    plt.title('Number of Non Zero Weights At the End of Each Epoch')
    plt.show()

    plt.figure()
    plt.plot(Wns, Training_Mses)
    plt.xlabel('# Selected Features')
    plt.ylabel('Reconstruction MSE')
    plt.title('Reconstruction MSE with respect to the number of selected features')
    plt.show()

if Saving == True:
    #Saving the model and the weights
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    weight_path = os.path.join(save_dir, weight_name)
    autoencoder.save_weights(weight_path)
    print('Saved trained weights at %s ' % weight_path)

test = autoencoder.predict(x_test)

if Dataset == 'mnist':

    w = autoencoder.layers[1].get_weights()[0]
    w = Clipper(w)
    labels = {0: 'Weight = -1', 1: 'Weight = 0', 2: 'Weight = 1'}
    values = np.unique(w.ravel())
    pix = (w).reshape(28, 28)
    plt.figure()
    plt.imshow(pix)
    im = plt.imshow(pix)
    colors = [im.cmap(im.norm(value)) for value in values]
    patches = [mpatches.Patch(color=colors[i],
                              label=labels[i]) for i in range(len(values))]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
    plt.title('The Pixels of the MNIST Images with k = 50')
    plt.show()

    #How many results we will display
    n = 40
    row = 5
    column = 8

    plt.figure(facecolor='black')
    num = 1
    while num < n or num == n:
        # display original
        ax = plt.subplot(row, column, num)
        plt.imshow(x_test[num + 100].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        num = num + 1
    plt.show()

    plt.figure(facecolor='black')
    num = 1
    while num < n or num == n:
        w = autoencoder.layers[1].get_weights()[0]
        w = Clipper(w)
        pixel = w * x_test[num + 100]
        ax = plt.subplot(row, column, num)
        plt.imshow(pixel.reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        num = num + 1
    plt.show()

    plt.figure(facecolor='black')
    num = 1
    while num < n or num == n:
        # display reconstruction
        ax = plt.subplot(row, column, num)
        plt.imshow(test[num + 100].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        num = num + 1
    plt.show()

elif Dataset == 'fashion':

    w = autoencoder.layers[1].get_weights()[0]
    w = Clipper(w)
    labels = {0: 'Weight = -1', 1: 'Weight = 0', 2: 'Weight = 1'}
    values = np.unique(w.ravel())
    pix = (w).reshape(28, 28)
    plt.figure()
    plt.imshow(pix)
    im = plt.imshow(pix, )
    colors = [im.cmap(im.norm(value)) for value in values]
    patches = [mpatches.Patch(color=colors[i],
                              label=labels[i]) for i in range(len(values))]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
    plt.title('The Pixels of the FMNIST Images with k = 50')
    plt.show()

    n = 40
    row = 5
    column = 8

    plt.figure(facecolor='black')
    num = 1
    while num < n or num == n:
        ax = plt.subplot(row, column, num)
        plt.imshow(x_test[num + 100].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        num = num + 1
    plt.show()

    plt.figure(facecolor='black')
    num = 1
    while num < n or num == n:
        w = autoencoder.layers[1].get_weights()[0]
        w = Clipper(w)
        pixel = w * x_test[num + 100]
        ax = plt.subplot(row, column, num)
        plt.imshow(pixel.reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        num = num + 1
    plt.show()

    plt.figure(facecolor='black')
    num = 1
    while num < n or num == n:
        ax = plt.subplot(row, column, num)
        plt.imshow(test[num + 100].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        num = num + 1
    plt.show()

elif Dataset == 'utk_face':

    w = autoencoder.layers[1].get_weights()[0]
    w = Clipper(w)
    labels = {0: 'Weight = -1', 1: 'Weight = 0', 2: 'Weight = 1'}
    values = np.unique(w.ravel())
    pix = (w).reshape(50, 50)
    plt.figure()
    plt.imshow(pix)
    im = plt.imshow(pix)
    colors = [im.cmap(im.norm(value)) for value in values]
    patches = [mpatches.Patch(color=colors[i],
                              label=labels[i]) for i in range(len(values))]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title('The Pixels of the UTK Face Images with k = 50')
    plt.show()

    n = 40
    row = 5
    column = 8

    plt.figure(facecolor='black')
    num = 1
    while num < n or num == n:
        ax = plt.subplot(row, column, num)
        plt.imshow(x_test[num + 100].reshape(50, 50))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        num = num + 1
    plt.show()

    plt.figure(facecolor='black')
    num = 1
    while num < n or num == n:
        w = autoencoder.layers[1].get_weights()[0]
        w = Clipper(w)
        pixel = w * x_test[num + 100]
        ax = plt.subplot(row, column, num)
        plt.imshow(pixel.reshape(50, 50))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        num = num + 1
    plt.show()

    plt.figure(facecolor='black')
    num = 1
    while num < n or num == n:
        ax = plt.subplot(row, column, num)
        plt.imshow(test[num + 100].reshape(50, 50))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        num = num + 1

    plt.show()


elif Dataset == 'channel':

    w = autoencoder.layers[1].get_weights()[0]
    w = Clipper(w)
    labels = {0: 'Weight = -1', 1: 'Weight = 0', 2: 'Weight = 1'}
    values = np.unique(w.ravel())
    pix = (w).reshape(72, 14)
    plt.figure()
    plt.imshow(pix)
    im = plt.imshow(pix)
    colors = [im.cmap(im.norm(value)) for value in values]
    patches = [mpatches.Patch(color=colors[i],
                              label=labels[i]) for i in range(len(values))]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title('The Pilots of the Channel with k = 48')
    plt.show()

elif Dataset == 'coil':

    w = autoencoder.layers[1].get_weights()[0]
    w = Clipper(w)
    labels = {0: 'Weight = -1', 1: 'Weight = 0', 2: 'Weight = 1'}
    values = np.unique(w.ravel())
    pix = (w).reshape(20, 20)
    plt.figure()
    plt.imshow(pix)
    im = plt.imshow(pix)
    colors = [im.cmap(im.norm(value)) for value in values]
    patches = [mpatches.Patch(color=colors[i],
                              label=labels[i]) for i in range(len(values))]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title('The Pixels of the COIL-20 Images with k = 50')
    plt.show()

    n = 40
    row = 5
    column = 8

    plt.figure(facecolor='black')
    num = 1
    while num < n or num == n:
        ax = plt.subplot(row, column, num)
        plt.imshow(x_test[num + 10].reshape(20, 20))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        num = num + 1
    plt.show()

    plt.figure(facecolor='black')
    num = 1
    while num < n or num == n:
        w = autoencoder.layers[1].get_weights()[0]
        w = Clipper(w)
        pixel = w * x_test[num + 10]
        ax = plt.subplot(row, column, num)
        plt.imshow(pixel.reshape(20, 20))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        num = num + 1
    plt.show()

    plt.figure(facecolor='black')
    num = 1
    while num < n or num == n:
        ax = plt.subplot(row, column, num)
        plt.imshow(test[num + 10].reshape(20, 20))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        num = num + 1

    plt.show()



