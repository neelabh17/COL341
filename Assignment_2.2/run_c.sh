cd /mnt/disk1/jatin/ml-ass-neel/COL341/Assignment_2.2/part_c
# RUN_NAME=Basicnet_step_lr_gamma=0.1_at_40_100_epochs_lr=1e-3
# RUN_NAME=SmallResnet_step_lr_gamma=0.1_at_40_100_epochs_lr=1e-3
# python train.py /mnt/disk1/jatin/ml-ass-neel/COL341/Assignment_2.2/data/cifar10/train_data.csv /mnt/disk1/jatin/ml-ass-neel/COL341/Assignment_2.2/data/cifar10/public_test.csv model_$RUN_NAME$.pth loss_$RUN_NAME.txt accuracy_$RUN_NAME.txt $RUN_NAME
# python train.py /mnt/disk1/jatin/ml-ass-neel/COL341/Assignment_2.2/data/cifar10/train_data.csv /mnt/disk1/jatin/ml-ass-neel/COL341/Assignment_2.2/data/cifar10/public_test.csv model.pth loss.txt accuracy.txt
python test.py /mnt/disk1/jatin/ml-ass-neel/COL341/Assignment_2.2/data/cifar10/public_test.csv model.pth public_test_pred.txt