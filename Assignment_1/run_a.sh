python3 linear.py a data/train.csv data/test.csv outputfile_a.txt weightfile_a.txt
python3 grade_a.py outputfile_a.txt weightfile_a.txt model_outputfile_a.txt model_weightfile_a.txt
python3 linear.py b data/train.csv data/test.csv regularization.txt model_outputfile_b.txt model_weightfile_b.txt bestparameter.txt