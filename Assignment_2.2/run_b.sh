cd /mnt/disk1/jatin/ml-ass-neel/COL341/Assignment_2.2/part_b
python train.py /mnt/disk1/jatin/ml-ass-neel/COL341/Assignment_2.2/data/cifar10/train_data.csv /mnt/disk1/jatin/ml-ass-neel/COL341/Assignment_2.2/data/cifar10/public_test.csv model.pth loss.txt accuracy.txt
python test.py /mnt/disk1/jatin/ml-ass-neel/COL341/Assignment_2.2/data/cifar10/public_test.csv model.pth pred.txt