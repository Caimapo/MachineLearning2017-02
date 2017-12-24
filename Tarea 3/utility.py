import itertools
import json
import numpy             as np
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plot_img_dataset(x_data, y_data, classes, classes_names=None, n_len=10, _figsize=(10, 10)):
    count_classes = len(classes)
    img_list = {i:[] for i in classes}

    list_full = False
    i = 0
    while not list_full and len(y_data) > i:
        element = y_data[i]
        if len(img_list[element]) < n_len:
            img_list[element].append(i)
        i += 1

        list_full = True
        for j in classes:
            if len(img_list[j]) < n_len:
                list_full = False
                break
        
    f, axarr = plt.subplots(count_classes, n_len, figsize=_figsize)
    for i in range(count_classes):
        curr_class = classes[i]
        img_class = img_list[curr_class]
        for j in range(n_len):
            index = img_class[j]
            axarr[i, j].imshow(x_data[index].reshape(28, 28), cmap="gray")
            axarr[i, j].get_xaxis().set_visible(False)
            axarr[i, j].get_yaxis().set_visible(False)
        if classes_names:
            axarr[i, 0].set_ylabel(classes_names[i])
        else:
            axarr[i, 0].set_ylabel(curr_class)
        axarr[i, 0].get_yaxis().set_visible(True)
        axarr[i, 0].get_yaxis().set_ticks([])
    plt.show()

def get_loss(elem):
    return elem['loss']

def load_hyperopt_out(file_name):
    data_json = json.load(open(file_name))
    maper_json = {
        'Dense': [30, 45, 60],
        'activation': ['relu', 'elu'],
        'optimizer': ['rmsprop', 'adam', 'sgd'],
        'batch_size': [64, 128],
        'conditional': [False, True],
    }
    list_json = []
    for data in data_json:
        reg = {
            'acc': data['result']['loss'],
            'conf': { 
                'Dense': maper_json['Dense'][data['misc']['vals']['Dense'][0]],
                'Dense_1': maper_json['Dense'][data['misc']['vals']['Dense_1'][0]],
                'Dense_2': maper_json['Dense'][data['misc']['vals']['Dense_2'][0]],
                'Dropout': data['misc']['vals']['Dropout'][0],
                'Dropout_1': data['misc']['vals']['Dropout_1'][0],
                'Dropout_2': data['misc']['vals']['Dropout_2'][0],
                'activation': maper_json['activation'][data['misc']['vals']['activation'][0]],
                'activation_1': maper_json['activation'][data['misc']['vals']['activation_1'][0]],
                'activation_2': maper_json['activation'][data['misc']['vals']['activation_2'][0]],
                'batch_size': maper_json['batch_size'][data['misc']['vals']['batch_size'][0]],
                'conditional': maper_json['conditional'][data['misc']['vals']['conditional'][0]],
                'optimizer': maper_json['optimizer'][data['misc']['vals']['optimizer'][0]],
            }
        }
        list_json.append(reg)
    list_json.sort(key=get_loss)
    return list_json

def plot_learning_curve(history_fold):
    n_plots = len(history_fold)
    x = len(hist[0][0].history['acc'])
    x_space = np.linspace(1, x, x)

    f, axarr = plt.subplots(n_plots, 2,figsize=(16,5*n_plots))
    for i in range(n_plots):
        train_acc = [hist[i][j].history['acc'] for j in range(5)]
        train_loss = [hist[i][j].history['loss'] for j in range(5)]
        val_acc = [hist[i][j].history['val_acc'] for j in range(5)]
        val_loss = [hist[i][j].history['val_loss'] for j in range(5)]

        train_acc_mean = np.mean(train_acc, axis=0)
        train_acc_std = np.std(train_acc, axis=0)
        train_loss_mean = np.mean(train_loss, axis=0)
        train_loss_std = np.std(train_loss, axis=0)
        val_acc_mean = np.mean(val_acc, axis=0)
        val_acc_std = np.std(val_acc, axis=0)
        val_loss_mean = np.mean(val_loss, axis=0)
        val_loss_std = np.std(val_loss, axis=0)

        axarr[i, 0].grid(True, linestyle='dashed')
        axarr[i, 0].fill_between(x_space, train_acc_mean - train_acc_std, train_acc_mean + train_acc_std, alpha=0.25, color="r")
        axarr[i, 0].fill_between(x_space, val_acc_mean - val_acc_std, val_acc_mean + train_acc_std, alpha=0.25, color="g")
        axarr[i, 0].plot(x_space, train_acc_mean, marker='o', linestyle='--', color="r", label="Training score")
        axarr[i, 0].plot(x_space, val_acc_mean, marker='o', linestyle='--', color="g", label="Cross-validation score")
        axarr[i, 0].set_xlabel('epoch')
        axarr[i, 0].set_ylabel('accuracy')
        axarr[i, 0].legend(loc='lower right')
        
        axarr[i, 1].grid(True, linestyle='dashed')
        axarr[i, 1].fill_between(x_space, train_loss_mean - train_loss_std, train_loss_mean + train_loss_std, alpha=0.25, color="r")
        axarr[i, 1].fill_between(x_space, val_loss_mean - val_loss_std, val_loss_mean + train_loss_std, alpha=0.25, color="g")
        axarr[i, 1].plot(x_space, train_loss_mean, marker='o', linestyle='--', color="r", label="Training score")
        axarr[i, 1].plot(x_space, val_loss_mean, marker='o', linestyle='--', color="g", label="Cross-validation score")
        axarr[i, 1].set_xlabel('epoch')
        axarr[i, 1].set_ylabel('loss')
        axarr[i, 1].legend(loc='upper right')
    plt.show()
    
