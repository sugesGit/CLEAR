# CLEAR

**CLEAR** mainly consists of three components that are documented in model/CLEAR.py, including 

1. the **C**ounterfactual **P**rompt **L**earning (**CPL**) Module,
2. the **A**daptive **D**ynamic **I**mputation (**ADI**) Module, 
3. the **M**ultimodal **R**epresentation **F**usion (**MRF**) Module.

### **CPL Module** 
(The corresponding codes can be found in model/CLEAR.py for the prompt-based disparity learning.)

1. The *querylinear* and *mquerylinear* function projects both factual and counterfactual prompts into a proper embedding space.
2. The *prompt_attention_** function captures the associations between the prompt query and the extracted representations.
3. The *enhanced_\*_attributes0* variables or *enhanced_\*_representations0* variables calculates the  counterfactual representation discrepancy, $CRD_(r^m_i)$.
4. The *dispairty* variable summarizes all representation discrepancies and is combined with *classifier_disparity* function to minimize the representation discrepancies and improve their discriminability as introduced in Eq12, in which these middle results are further fed to the counterfacutal prompt learning loss *'loss_d + loss_cf'* in train.py for the model training.

### **ADI Module**   
(The corresponding codes can be found in model/CLEAR.py for representation calibration.) 

1. The *sigmoid* function is used to quantify the representation discrepancies to a discriminability assessment metric as in Eq13.
2. The *tao* variable removes the contaminated representations as in Eq14.
3. The *gate_\** function calculates the representation weights $alpha\_*$ that reweights the calibrated representations and the representation discrepancies to get the enhanced representations *enhanced_\*_representations*.

### **MRF Module**   
(The corresponding codes can be found in model/CLEAR.py for multimodal representation fusion.) 

1. The *intra_modality_self_attention* function is used to capture the intra-modality representation associations.
2. The *inter_modality_representation_interaction* function is used to calculates inter-modality representation correlations.



















