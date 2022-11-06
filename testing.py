from sklearn.metrics import classification_report, f1_score, recall_score, precision_score

class Testing:

    @classmethod
    def test(cls, pred_classes, labels, target_names=None):

        report = classification_report(y_true=labels,y_pred=pred_classes, target_names=target_names, output_dict=True)
        report["f1_score"] = f1_score(y_true=labels,y_pred=pred_classes)
        report["recall_score"] = recall_score(y_true=labels,y_pred=pred_classes)
        report["precision_score"] = precision_score(y_true=labels,y_pred=pred_classes)
        return report

