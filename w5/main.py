from AICity import AICity

if __name__== "__main__":
    aic = AICity(data_path="../../data/AICity_data/train/",
                 model_yaml="COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml",
                 epochs=5000,
                 batch_size=2,
                 train_val_split=0.2,
                 train_seq=["S01", "S04"],
                 test_seq=["S03"])

    # aic.train_reid(backbone='resnet50', backbone_epochs=5, triplet_epochs=25, batch_size=16, lr=0.001, finetune=True)

    aic.multi_camera_reid(model_name='resnet50_finetune')