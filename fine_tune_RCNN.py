import os.path
from RCNN import preprocessing_RCNN as prep
from RCNN import config
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

# Use a already trained alexnet with the last layer redesigned
def create_alexnet(num_classes, restore=False):
    # Building 'AlexNet'
    network = input_data(shape=[None, config.IMAGE_SIZE, config.IMAGE_SIZE, 3])
    network = conv_2d(network, 96, 11, strides=4, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 256, 5, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 256, 3, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    # do not restore this layer(参数相比train_alexnet中的create Alexnet多了restore=False)
    network = fully_connected(network, num_classes, activation='softmax', restore=restore)
    network = regression(network, optimizer='momentum',
                         loss='categorical_crossentropy',
                         learning_rate=0.001)
    return network

def fine_tune_Alexnet(network, X, Y, save_model_path, fine_tune_model_path):
    # Training
    #tflearn.DNN:将网络结构封装在model中
    model = tflearn.DNN(network, checkpoint_path='rcnn_model_alexnet',
                        max_checkpoints=1, tensorboard_verbose=2, tensorboard_dir='output_RCNN')
    #加载fine_tune模型
    if os.path.isfile(fine_tune_model_path + '.index'):
        print("Loading the fine tuned model")
        model.load(fine_tune_model_path)
    #加载预训练好的模型
    elif os.path.isfile(save_model_path + '.index'):
        print("Loading the alexnet")
        model.load(save_model_path)
    else:
        print("No file to load, error")
        return False

    model.fit(X, Y, n_epoch=1, validation_set=0.1, shuffle=True,
              show_metric=True, batch_size=64, snapshot_step=200,
              snapshot_epoch=False, run_id='alexnet_rcnnflowers2')
    # Save the model
    model.save(fine_tune_model_path)


if __name__ == '__main__':
    data_set = config.FINE_TUNE_DATA
    #第一次进行fine_tune
    if len(os.listdir(config.FINE_TUNE_DATA)) == 0:
        print("Reading Data")
        prep.load_train_proposals(config.FINE_TUNE_LIST, 2, save=True, save_path=data_set)
    print("Loading Data")
    X, Y = prep.load_from_npy(data_set)
    restore = False
    if os.path.isfile(config.FINE_TUNE_MODEL_PATH + '.index'):
        restore = True
        print("Continue fine-tune")
    # three classes include background
    net = create_alexnet(config.FINE_TUNE_CLASS, restore=restore)
    fine_tune_Alexnet(net, X, Y, config.SAVE_MODEL_PATH, config.FINE_TUNE_MODEL_PATH)