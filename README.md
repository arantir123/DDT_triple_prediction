# FLONE: fully Lorentz network embedding for inferring novel drug targets (under review)

How to use it:

1. __Unfold the file ./DDT_triple_prediction/DDT/data/Cross_validation_split.rar, to obtain the required model input for each independent repeat (based on DTINet).__
2. __Follow the instruction in each following .py file to execute different variants described in the manuscript on each independent repeat:__
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
3. __Files in ./DDT_triple_prediction/DDT/outputs:__
  * __hypo_coordinate_set_original_fold4_515_449.pickle and hypo_coordinate_set_transformed_fold4_515_449.pickle:__ 2-D coordinates of the hyperbolic spatial layout in Fig. 2. (A) and (B) in the manuscript (based on Hypo_visualization.py).
  * __eucli_coordinate_set_original_fold4_515_449.pickle and eucli_coordinate_set_transformed_fold4_515_449.pickle:__ 2-D coordinates of the Euclidean spatial layout in Fig. 2. (C) and (D) in the manuscript (based on Eucli_visualization.py).
  
  
  

