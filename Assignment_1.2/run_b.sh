TEST=6
python3 logistic.py b data/train.csv data/test.csv testcases/$TEST/param$TEST.txt testcases/$TEST/model_outputfile$TEST.txt testcases/$TEST/model_weightfile$TEST.txt
python3 grade_b.py testcases/$TEST/model_outputfile$TEST.txt testcases/$TEST/model_weightfile$TEST.txt testcases/$TEST/outputfile$TEST.txt testcases/$TEST/weightfile$TEST.txt