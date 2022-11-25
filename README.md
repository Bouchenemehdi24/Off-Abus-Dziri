# Offensive-and-Abusive-language-Classification-on-Dziri-Language

The emergence of abusive and offensive content in social media is disturbing, and most governments are fighting this phenomenon. In the literature, several research works have been proposed to detect such content automatically. However, most of the works are addressing formal and well-known languages (e.g. English, Arabic, French, etc.), while there is a scarcity of works dealing with under-resourced languages such as Dziri (i.e. Algerian dialectal Arabic) language. To tackle this problem, in this article, we deal with offensive/abusive Dziri language detection, where we use two approaches to improve the identification performance. In the first approach, we propose a set of preprocessing tasks dedicated to this language. In addition, we use Bayesian optimization to optimize the Chi-2 feature selection and the RBF-SVM classifier hyper-parameters. In the second approach, we fine-tuned pre-trained transformer language models (i.e. DziriBERT, BERT base model (uncased) and XLM-Roberta). The comparative study with baseline systems under the same conditions figured out that our proposal is promising, and highly improved the accuracy with 12.07\% and 5.92\% for three-class and two-class respectively using the first approach in comparison with baseline. While in the second approach, the DziriBERT model reported the highest performances in these experiments (i.e. 14.35\% and 10.72\% of $\Delta$ accuracy for three-class and two-class respectively) in comparison with baseline.

# Content

## Dataset

### Two-classes {0 : 'Normal', 2 : 'Abusive/Offensive'}
 
* keras_train_pre_2c_trans_.txt Training data for two-classes.
* keras_test_pre_2c_trans_.txt test data for two-clasess.

### Three-classes {0 : 'Normal', 1 : 'Abusive', 2 : 'Offensive'}

* keras_test_pre_3c_trans_.txt test data for three-classes.
* keras_train_pre_3c_trans_.txt Training data for for three-classes.

 ## Code
 
### Classic ML approach (Read the paper carefully)

* 3C_prechi2svm.ipynb Python code in jupyter notebook interface for three-classes
* 2C_prechi2svm.ipynb Python code in jupyter notebook interface for for two-clasess.

### Deep learning approach( Transfert learning)
* 2C_dziri_bert.py Fine-Tuning DziriBERT  for two-clasess.
* 3C_dziri_bert.py Fine-Tuning DziriBERT  for three-classes.

# Cite Us
