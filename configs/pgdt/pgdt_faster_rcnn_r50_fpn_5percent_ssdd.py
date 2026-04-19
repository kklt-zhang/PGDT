_base_ = ['pgdt_faster_rcnn_r50_fpn_10percent_ssdd.py']
# 5% coco train2017 is set as labeled dataset
labeled_dataset = _base_.labeled_dataset
unlabeled_dataset = _base_.unlabeled_dataset
labeled_dataset.ann_file = 'annotations/instances_train_5percent.json'
unlabeled_dataset.ann_file = 'annotations/instances_unlabeled_95percent.json'
train_dataloader = dict(
    dataset=dict(datasets=[labeled_dataset, unlabeled_dataset]))
