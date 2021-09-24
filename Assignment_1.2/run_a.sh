TEST=3
python3 logistic.py a data/train.csv data/test.csv testcases/$TEST/param$TEST.txt testcases/$TEST/model_outputfile$TEST.txt testcases/$TEST/model_weightfile$TEST.txt
python3 grade_a.py testcases/$TEST/model_outputfile$TEST.txt testcases/$TEST/model_weightfile$TEST.txt testcases/$TEST/outputfile$TEST.txt testcases/$TEST/weightfile$TEST.txt