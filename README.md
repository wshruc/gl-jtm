## GL-JTM: A Joint Training Framework for Knowledge Base Question Answerings

* Data
  > 1. Glove [glove.6B.300d.txt] —— Download pre-trained word vectors from <https://nlp.stanford.edu/projects/glove/>
  > 2. All the pre-processed dataset and the trained model can download from https://pan.baidu.com/s/1G77q15u6AumCFvobP19jxw code: s3rk

* Relation Detection Train/Test
  > 1. Run `python main.py -train --data_type sq or python main.py -train --data_type wq` to train our GL-JTM model.
  > 2. Run `python main.py -test --data_type sq or python main.py -test --data_type wq` to test our GL-JTM model, and the results saved in ../result/

* Test our KBQA system
  > 1. Run `python sq_kbqa_system.py` to test our KBQA system on SimpleQuestions.
  > 2. Run `python wq_kbqa_system.py` to test our KBQA system on WebQSP.
