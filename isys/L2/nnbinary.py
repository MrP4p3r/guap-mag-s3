import numpy as np


def to_categorical(labels, labels_uniq_sorted=None):
    """Транслирует список меток входных образов в массив выходных векторов"""
    if not labels_uniq_sorted:
        labels_uniq_sorted = sorted(set(labels))
    output_vectors = np.identity(len(labels_uniq_sorted))
    output_of = {
        label: output_vectors[i]
        for i, label in enumerate(labels_uniq_sorted)
    }
    result = np.array([output_of[label] for label in labels])
    return result


def calculate_accuracy(predictions, labels):
    """Сравнивает два переданных массива и возвращает долю совпадений"""
    matches = (predictions==labels).all(axis=-1)
    return np.sum(matches)/len(predictions)


class OneLayerBinaryNN:
    
    def __init__(self, input_shape, outputs_number, activation=None):
        assert len(input_shape) == 2 and outputs_number > 0
        self.input_shape = input_shape
        self.outputs_number = outputs_number
        if callable(activation):
            self.activate_f = activation
        self.weights = self._generate_weights()

    def fit(self, X_train, Y_train, epochs, X_test=None, Y_test=None, verbose=1):
        self.verbose = verbose
        perform_test = (np.any(X_test) and X_test.size and np.any(Y_test) and Y_test.size)
        input_images_num = X_train.shape[0]
        for epoch_num in range(epochs):
            self.log('Epoch %i.' % epoch_num, end='')
            # Тренировка
            for idx in range(input_images_num):
                image = X_train[idx]
                expectation = Y_train[idx]
                reality = self.predict_one(image)
                self._correct_weights(image, expectation, reality)
            # Измерение точности на `тренировочной' выборке
            predictions = self.predict(X_train)
            train_acc = calculate_accuracy(predictions, Y_train)
            self.log('  train acc: %.4f' % train_acc, end='')
            # Измерение точности на `тестовой' выборке
            if perform_test:
                predictions = self.predict(X_test)
                test_acc = calculate_accuracy(predictions, Y_test)
                self.log('  test acc: %.4f' % test_acc, end='')
            self.log()
            if train_acc == 1:
                self.log('Early stopping. Train accuracy is 100%')
                break
        self.log('Done')
            
    def predict(self, images):
        result = []
        images_num = images.shape[0]
        for idx in range(images_num):
            prediction = self.predict_one(images[idx])
            result.append(prediction)
        return np.array(result)
    
    def predict_one(self, image):
        product = self.weights * image
        raw_outputs = np.sum(product, axis=(1,2))
        outputs = self.activate_f(raw_outputs)
        return outputs
    
    @staticmethod
    def activate_f(raw_outputs):
        outputs = (raw_outputs > 0).astype('float32')
        return outputs
    
    def log(self, *msg, end='\n'):
        if self.verbose:
            print(*msg, end=end)
    
    def _correct_weights(self, image, expectation, reality):
        sh = self.weights.shape
        image_tiled = np.tile(image, (sh[0], 1)).reshape(sh)
        deltas = expectation - reality
        deltas = deltas.reshape(deltas.shape + (1, 1))
        self.weights += deltas * image_tiled
        
    def _generate_weights(self):
        return np.zeros((self.outputs_number,) + self.input_shape)
    