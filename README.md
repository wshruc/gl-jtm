## GL-JTM: A Joint Training Framework for Knowledge Base Question Answerings
* Relation detection and entity recognition are two critical subtasks in semantic parsing-based methods for KBQA and are also important research topics in information retrieval. Most previous studies have considered them as two separate subtasks and have therefore trained their models separately. As a result, these studies have often ignored the potential correlation between them, which might restrict further performance improvements. This paper proposes a global-local joint training model (GL-JTM) to address this problem by combining entity recognition and relation detection under a unified framework.

* Data
  > 1. The directory structure of the GL-JTM code is as follows:
    - gl-jtm
       - data
         - sq_relations
         - webqsp_relations
       - embedding
       - src
         - save_models
    - results
    
    First of all, we need to create a folder according to the directory structure and put the above code into the **./gl-jtm/src** directory.
  > 2. Glove [glove.6B.300d.txt] —— Download pre-trained word vectors from <https://nlp.stanford.edu/projects/glove/>
  > 3. All the pre-processed dataset and the trained model can download from https://pan.baidu.com/s/1G77q15u6AumCFvobP19jxw code: s3rk   
       Download all the data and put it in the corresponding folder.
  > 4. For SimpleQuestions, the data format is *question pattern || question || label || topic entity || answer || relation || entity mention || golden relation || candidate relations ||* 
       For WebQSP, the data format is *question pattern || question || label || entity mention || entity mention || topic entity || relation || answer || golden relation || candidate 
       relations*
       Among them, we only use the four columns of question, label, golden relation, and candidate relations when training and testing GL-JTM.
  > 5. We provide a trained model that can be placed in the **./gl-jtm/src/save_models** folder for direct testing.
  
* GL-JTM Train/Test
  > 1. Run `python main.py -train --data_type sq or python main.py -train --data_type wq` to train our GL-JTM model.
  > 2. Run `python main.py -test --data_type sq or python main.py -test --data_type wq` to test our GL-JTM model, and the test results saved in ../result/

* Before testing our KBQA system, we need to link the entity mentions obtained from the model to the knowledge base using the entity linking method.
  > 1. The *entity_linking.py* file provides an example of entity linking using the Knowledge Graph Search API.
* Test our KBQA system
  > 1. Run `python sq_kbqa_system.py` to test our KBQA system on SimpleQuestions.
  > 2. Run `python wq_kbqa_system.py` to test our KBQA system on WebQSP.
