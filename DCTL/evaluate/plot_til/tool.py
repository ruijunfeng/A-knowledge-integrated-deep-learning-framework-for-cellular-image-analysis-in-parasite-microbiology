import os
import numpy as np
from itertools import cycle
import matplotlib.pyplot as plt
from scipy import interp
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, classification_report

def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset+labels_dense.ravel()] = 1
    return labels_one_hot

# calucate F1-score precision accuracy recall
def calucate(ground_truth_labels, prediction_argmax_results):
    target_names = ['class 0', 'class 1', 'class 2', 'class 3']

    accuracy = accuracy_score(ground_truth_labels, prediction_argmax_results)
    
    f1_s1 = f1_score(ground_truth_labels, prediction_argmax_results, average='micro')
    f1_s2 = f1_score(ground_truth_labels, prediction_argmax_results, average='macro')
    f1_s3 = f1_score(ground_truth_labels, prediction_argmax_results, average='weighted')
    f1_s4 = f1_score(ground_truth_labels, prediction_argmax_results, average=None)

    re1 = recall_score(ground_truth_labels, prediction_argmax_results, average='micro')
    re2 = recall_score(ground_truth_labels, prediction_argmax_results, average='macro')
    re3 = recall_score(ground_truth_labels, prediction_argmax_results, average='weighted')
    re4 = recall_score(ground_truth_labels, prediction_argmax_results, average=None)

    pre1 = precision_score(ground_truth_labels, prediction_argmax_results, average='micro')
    pre2 = precision_score(ground_truth_labels, prediction_argmax_results, average='macro')
    pre3 = precision_score(ground_truth_labels, prediction_argmax_results, average='weighted')
    pre4 = precision_score(ground_truth_labels, prediction_argmax_results, average=None)
    
    matrix = confusion_matrix(ground_truth_labels, prediction_argmax_results)
    
    print('accuracy', accuracy)
    print('-'*60)
    
    print('Calculated using the average method of micro')
    print('f1_sore_micro', f1_s1)
    print('recall_micro', re1)
    print('precision_micro', pre1)
    print('-'*60)
    
    print('Calculated using the average method of macro')
    print('f1_sore macro', f1_s2)
    print('recall macro', re2)
    print('precision macro', pre2)
    print('-'*60)
    
    print('Calculated using the average method of weighted')
    print('f1_sore_weighted', f1_s3)
    print('recall_weighted', re3)
    print('precision_weighted', pre3)
    print('-'*60)

    print('Calculated using the average method of None (metric value of each class)')
    print('f1_sore_None', f1_s4)
    print('recall_None', re4)
    print('precision_None', pre4)
    print('-'*60)
    
    print('confusion_matrix \n', matrix)
    print('-'*60)
    
    print('classification', classification_report(ground_truth_labels, prediction_argmax_results, target_names=target_names))
    print('-'*60)

def roc(sum_label, f_yroc, num_classes, roc_dir):
    # init
    sum_label = np.array(sum_label)
    f_yroc = np.array(f_yroc)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    # roc curve for each class
    for i in range(num_classes): 
        fpr[i], tpr[i], _ = roc_curve(sum_label[:, i], f_yroc[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # micro-average roc curve and roc area
    fpr["micro"], tpr["micro"], _ = roc_curve(sum_label.ravel(), f_yroc.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # macro-average roc curve and roc area
    # aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(4)]))
    # interpolate all roc curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # auc area compute
    mean_tpr /= num_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # save curve data
    np.save(roc_dir + '/vgg_fpr.npy', fpr)
    np.save(roc_dir + '/vgg_tpr.npy', tpr)
    np.save(roc_dir + '/vgg_roc_auc.npy', roc_auc)
    
    # show the results
    print("auc values summary")
    for i in range(num_classes):
        print('auc for class %d is %.6f' %(i, roc_auc[i]))
    print('auc calculated by micro-average %.6f' %(roc_auc[i]))
    print('auc calculated by macro-average %.6f' %(roc_auc[i]))
    print('-'*60)
    
    # plot
    print('Plot the ROC curve for each class')
    lw = 2
    fig = plt.figure(figsize=(12, 6))
    ax = plt.subplot(111)
    # plot micro-average and macro-average
    plt.plot(fpr["micro"], tpr["micro"],
             label='ROC curve micro-average (area = {0:0.2f})'.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)
    plt.plot(fpr["macro"], tpr["macro"],
             label='ROC curve macro-average (area = {0:0.2f})'.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)
    # plot for each class
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue','red','yellow','block'])
    for i, color in zip(range(num_classes), colors):
        ax.plot(fpr[i], tpr[i], color=color, lw=lw, 
                label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))
    # format
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC_Curve')
    # shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(os.path.join(roc_dir, 'roc_curve.png'))
    plt.show()