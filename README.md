# Offensive-and-Abusive-language-Classification-on-Dziri-Language
The emergence of abusive and offensive content in social media is disturbing, and the most of governments are fighting this phenomenon. In the literature, several research works have been proposed to detect such content automatically. However, the most of the works are addressing formal and well-known languages (e.g. English, Arabic, French, etc.), while there is a scarcity of works dealing with under-resourced languages. In this investigation, we deal with the offensive Dziri (i.e. Algerian dialectal Arabic) language detection, where we use two approaches to improve the identification score. In the first approach,
  we propose a set of preprocessing tasks dedicated to this language. In addition, we use the Bayesian optimization to optimize the Chi-2 feature selection and the RBF-SVM classifier hyper-parameters. In the second approach, we fine-tuned pre-trained transformer language models (i.e. DziriBERT, BERT base model (uncased) and XLM-Roberta). The comparative study with baseline systems under the same conditions figured out that our proposal is promising, and highly improved the accuracy with 12.07\% and 4.8\% for three-class and two-class classification using the first approach, while in the second approach, DziriBERT model reported the highest performances in these experiments
  (i.e. 13.94\% and 8.7\% of $\Delta$ accuracy for three-class and two-class classification) in comparison with baseline.
# Content
 ## Dataset
 
keras_test_pre_2c_trans_.txt Training data for two-class.
keras_test_pre_2c_trans_.txt test data for two-class.

keras_test_pre_3c_trans_.txt test data for three-classes.
keras_test_pre_3c_trans_.txt Training data for for three-classes.





