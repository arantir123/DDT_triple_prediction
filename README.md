# FLONE: fully Lorentz network embedding for inferring novel drug targets

How to use it:

1. __Unfold the file ./DDT_triple_prediction/DDT/data/Cross_validation_split.rar, to obtain the required model input for each fold.
2. __Use the instruction of each .py file to execute different variants described in the manuscript on each fold:
  * Eucli_ECFP6_seqsimilarity.py: for Euclidean based variants using domain knowledge.
  * Eucli_noECFP6_noseqsimilarity.py: for Euclidean based variants without using domain knowledge.
  * Eucli_visualization: visualize the spatial layout of target embeddings under given drug and disease based on Euclidean KGC methods.
  * Hypo_ECFP6_noontology_noppi.py: for hyperbolic based variants using ECFP6 + self-contained target embedding look-up table.
  * Hypo_ECFP6_ontology_ppi.py: for hyperbolic based variants using ECFP6 + GO/PPI graph.
  * Hypo_ECFP6_ontology_ppi_detached1.py: for pre-training hyperbolic target embeddings using GO/PPI graph.
  * Hypo_ECFP6_ontology_ppi_detached2.py: mainly for hyperbolic based variants using ECFP6 + pre-trained target embeddings from detached1.py or using ECFP6 + target sequence similarity.
  * Hypo_noECFP6_noontology_noppi.py: for hyperbolic based variants using self-contained drug and target embedding look-up tables.
  * Hypo_noECFP6_ontology_ppi.py: for hyperbolic based variants using self-contained drug embedding look-up table + GO/PPI graph.
  * Hypo_visualization: visualize the spatial layout of target embeddings under given drug and disease based on hyperbolic KGC methods.
3. __Files in ./DDT_triple_prediction/DDT/outputs:
  * hypo_coordinate_set_original_fold4_515_449.pickle and hypo_coordinate_set_transformed_fold4_515_449.pickle: 2-D coordinates of the hyperbolic spatial layout in Fig. 2. (C) and (D).
  * eucli_coordinate_set_original_fold4_515_449.pickle and eucli_coordinate_set_transformed_fold4_515_449.pickle: 2-D coordinates of the Euclidean spatial layout in Fig. 2. (C) and (D).
  
  
  

