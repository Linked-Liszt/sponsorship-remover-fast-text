import fastText
import numpy as np

TRAIN_FILE = "data/train.txt"
TEST_FILE = "data/test.txt"

MODEL_OUTPUT = "model.bin"

#BASE_PARAMS
EPOCH = 1000

#Simple Train Params
LEARNING_RATE = 0.3
NGRAMS = 3

#Hyperparam Search Parameters
HYPERPARAM_SEARCH = True
LEARNING_RATE_START = 0.1
LEARNING_RATE_END = 0.3
LEARNING_RATE_STEPS = 5

NGRAM_BEGIN = 1
NGRAM_END = 3
#Will increment by 1

def print_results(N, p, r):
    print("N\t" + str(N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))


if HYPERPARAM_SEARCH:
    best_accuracy = 0
    best_lr = 0
    best_ngram = 0
    print("Begin Hyperparamter Search with {0} epochs".format(EPOCH))
    for lr in np.linspace(LEARNING_RATE_START, LEARNING_RATE_END, LEARNING_RATE_STEPS):
        for ngram in range(NGRAM_BEGIN, NGRAM_END):
            print("Training Model with {0} learning rate and {1} ngrams".format(lr, ngram))
            model = fastText.train_supervised(TRAIN_FILE, lr=lr, epoch=EPOCH, wordNgrams=ngram)
            _, accuracy, _ = model.test(TEST_FILE)
            print("Model Accuracy: {0}".format(accuracy))
            print("Previous Best Model Accuracy: {0}".format(best_accuracy))
            print("\n--------------------------------\n")
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_lr = lr
                best_ngram = ngram
                model.save_model(MODEL_OUTPUT)
                print("Creatd Model {0}".format(MODEL_OUTPUT))
    print("Search Complete")
    print("Best Model Accuracy: {0}".format(best_accuracy))
    print("Best Model Params: {0} learning rate; {1} ngrams".format(best_lr, best_ngram))


else:
    model = fastText.train_supervised(TRAIN_FILE, lr=LEARNING_RATE, epoch=EPOCH, wordNgrams=NGRAMS)
    model.save_model(MODEL_OUTPUT)
    print_results(*model.test(TEST_FILE))