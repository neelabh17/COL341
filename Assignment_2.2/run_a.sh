cd /mnt/disk1/jatin/ml-ass-neel/COL341/Assignment_2.2/part_a
python train.py /mnt/disk1/jatin/ml-ass-neel/COL341/Assignment_2.2/data/devnagri/train_data_shuffled.csv /mnt/disk1/jatin/ml-ass-neel/COL341/Assignment_2.2/data/devnagri/public_test.csv model.pth loss.txt accuracy.txt
python test.py /mnt/disk1/jatin/ml-ass-neel/COL341/Assignment_2.2/data/devnagri/public_test.csv model.pth pred.txt