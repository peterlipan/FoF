import numpy as np
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, \
    roc_auc_score, precision_score, matthews_corrcoef, cohen_kappa_score
from imblearn.metrics import sensitivity_score, specificity_score


def compute_avg_metrics(ground_truth, activations, avg='micro'):
    ground_truth = ground_truth.cpu().detach().numpy()
    activations = activations.cpu().detach().numpy()
    predictions = np.argmax(activations, -1)
    mean_acc = accuracy_score(y_true=ground_truth, y_pred=predictions)
    f1_macro = f1_score(y_true=ground_truth, y_pred=predictions, average=avg)
    try:
        auc = roc_auc_score(y_true=ground_truth, y_score=activations, multi_class='ovr', average=avg)
    except ValueError as error:
        print('Error in computing AUC. Error msg:{}'.format(error))
        auc = 0
    bac = balanced_accuracy_score(y_true=ground_truth, y_pred=predictions)
    sens_macro = sensitivity_score(y_true=ground_truth, y_pred=predictions, average=avg)
    spec_macro = specificity_score(y_true=ground_truth, y_pred=predictions, average=avg)
    prec_macro = precision_score(y_true=ground_truth, y_pred=predictions, average=avg)
    mcc = matthews_corrcoef(y_true=ground_truth, y_pred=predictions)
    kappa = cohen_kappa_score(y1=ground_truth, y2=predictions)

    return mean_acc, f1_macro, auc, bac, sens_macro, spec_macro, prec_macro, mcc, kappa
