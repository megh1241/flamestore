import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import random
from tensorflow.keras import backend as K
from flamestore.client import Client
import resnet50
import spdlog


logger = spdlog.ConsoleLogger("Benchmark")
logger.set_pattern("[%Y-%m-%d %H:%M:%S.%F] [%n] [%^%l%$] %v")


def __load_dataset():
    return resnet50.load_dataset(
        train_file='train_signs.h5',
        test_file='test_signs.h5',
        train_set_x='train_set_x',
        train_set_y='train_set_y',
        test_set_x='test_set_x',
        test_set_y='test_set_y',
        list_classes='list_classes')


def create_and_train_new_model(workspace, dataset):
    logger.info('===> Creating FlameStore client')
    client = Client(workspace=workspace)
    logger.info('===> Creating Keras model')
    model = resnet50.create_model()
    logger.info('===> Building model')
    resnet50.build_model(model)
    logger.info('===> Registering model')
    client.register_model('my_model', model, include_optimizer=False)
    logger.info('===> Training model')
    resnet50.train_model(model, dataset, batch_size=32, epochs=1)
    logger.info('===> Saving model data')
    client.save_weights('my_model', model, include_optimizer=False)
    logger.info('===> Evaluating the model')
    score = resnet50.evaluate_model(model, dataset, verbose=0)
    logger.info('===> Scores: '+str(score))
    del model
    K.clear_session()


def reload_and_eval_existing_model(workspace, dataset):
    logger.info('===> Creating FlameStore client')
    client = Client(workspace=workspace)
    logger.info('===> Reloading model config')
    model = client.reload_model('my_model', include_optimizer=False)
    logger.info('===> Rebuilding model')
    resnet50.build_model(model)
    logger.info('===> Reloading model data')
    client.load_weights('my_model', model, include_optimizer=False)
    logger.info('===> Evaluating the stored model')
    score = resnet50.evaluate_model(model, dataset, verbose=0)
    logger.info('===> Scores: '+str(score))
    del model
    K.clear_session()


def duplicate_and_eval_existing_model(workspace, dataset):
    logger.info('===> Creating FlameStore client')
    client = Client(workspace=workspace)
    logger.info('===> Duplicating model')
    client.duplicate_model('my_model', 'my_duplicated_model')
    logger.info('===> Reloading duplicate')
    model = client.reload_model('my_duplicated_model', include_optimizer=False)
    logger.info('===> Rebuilding model')
    resnet50.build_model(model)
    logger.info('===> Reloading model data')
    client.load_weights('my_duplicated_model', model, include_optimizer=False)
    logger.info('===> Evaluating the stored model')
    score = resnet50.evaluate_model(model, dataset, verbose=0)
    logger.info('===> Scores: '+str(score))
    del model
    K.clear_session()


if __name__ == '__main__':
    random.seed(1234)
    if(len(sys.argv) < 2):
        logger.info("Usage: python client-lenet5.py <workspace>")
        sys.exit(-1)
    logger.info('=> Loading MNIST dataset')
    dataset = __load_dataset()
    workspace = sys.argv[1]
    logger.info('=> Workspace is '+workspace)
    logger.info('=> Creating and training a new model')
    create_and_train_new_model(workspace, dataset)
    logger.info('=> Reloading and evaluating existing model')
    reload_and_eval_existing_model(workspace, dataset)
    logger.info('=> Duplicating and evaluating a model')
    duplicate_and_eval_existing_model(workspace, dataset)
