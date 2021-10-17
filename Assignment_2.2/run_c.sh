cd /mnt/disk1/jatin/ml-ass-neel/COL341/Assignment_2.2/part_c
RUN_NAME=BasicNN_step_lr_gamma=0.1_at_30
python train.py /mnt/disk1/jatin/ml-ass-neel/COL341/Assignment_2.2/data/cifar10/train_data.csv /mnt/disk1/jatin/ml-ass-neel/COL341/Assignment_2.2/data/cifar10/public_test.csv model_$RUN_NAME$.pth loss_$RUN_NAME.txt accuracy_$RUN_NAME.txt $RUN_NAME