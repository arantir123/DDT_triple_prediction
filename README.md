# FLONE
### FLONE: Fully Lorentz network embedding for inferring novel drug targets

__The original bioRxiv version link: [https://www.biorxiv.org/content/10.1101/2023.03.20.533432v1.abstract]__

__The accepted version link: [https://academic.oup.com/bioinformaticsadvances/article/3/1/vbad066/7178008]__

__How to use it:__

__Our Test Environment (Windows or Linux, the higher environment version beyond this might be also feasible):__
* Python 3.6.13
* Pytorch 1.10.2
* CUDA tool kit 11.3.1
* geoopt 0.4.1

Step1. __Unfold the file ./DDT_triple_prediction/DDT/data/Cross_validation_split.rar, to obtain the required model input for each independent repeat (based on DTINet).__

Step2. __Follow the instruction in each following .py file to execute different variants described in the manuscript on each independent repeat:__
  * __Eucli_ECFP6_seqsimilarity.py:__ for Euclidean based variants using domain knowledge.
  * __Eucli_noECFP6_noseqsimilarity.py:__ for Euclidean based variants without using domain knowledge.
  * __Eucli_visualization.py:__ visualize the spatial layout of target embeddings under given drug and disease based on Euclidean KGC methods.
  * __Hypo_ECFP6_noontology_noppi.py:__ for hyperbolic based variants using ECFP6 + self-contained target embedding look-up table.
  * __Hypo_ECFP6_ontology_ppi.py:__ for hyperbolic based variants using ECFP6 + GO/PPI graph.
  * __Hypo_ECFP6_ontology_ppi_detached1.py:__ for pre-training hyperbolic target embeddings using GO/PPI graph.
  * __Hypo_ECFP6_ontology_ppi_detached2.py:__ mainly for hyperbolic based variants using ECFP6 + pre-trained target embeddings from detached1.py or using ECFP6 + target sequence similarity.
  * __Hypo_noECFP6_noontology_noppi.py:__ for hyperbolic based variants using self-contained drug and target embedding look-up tables.
  * __Hypo_noECFP6_ontology_ppi.py:__ for hyperbolic based variants using self-contained drug embedding look-up table + GO/PPI graph.
  * __Hypo_visualization.py:__ visualize the spatial layout of target embeddings under given drug and disease based on hyperbolic KGC methods.
  
__Check visualization results in the manuscript: files in ./DDT_triple_prediction/DDT/outputs:__
  * __hypo_coordinate_set_original_fold4_515_449.pickle and hypo_coordinate_set_transformed_fold4_515_449.pickle:__ 2-D coordinates of the hyperbolic spatial layout in the illustration (A) and (B) of the manuscript (based on Hypo_visualization.py).
  * __eucli_coordinate_set_original_fold4_515_449.pickle and eucli_coordinate_set_transformed_fold4_515_449.pickle:__ 2-D coordinates of the Euclidean spatial layout in the illustration (C) and (D) of the manuscript (based on Eucli_visualization.py).
  
  
  

