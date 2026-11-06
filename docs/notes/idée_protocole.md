> etapes pour from scratch
PUIS
etapes pour distilatiion
Dans les grandes lignes


## From scratch

1. **Préparer les données** : télécharger DALI v2 (annotations et audio), séparer la voix de chaque chanson, récupérer le découpage train/validation de jhuang448.
2. **Audio → spectrogramme** : réutiliser leur `data.py`, qui convertit chaque extrait en spectrogramme mel.
3. **Écrire le modèle acoustique** : un modèle **causal** (LSTM unidirectionnel, convolutions qui ne regardent que le passé), avec 72 classes en sortie.
4. **Entraîner** : copie de leur `train.py` avec votre modèle à la place du leur. La « réponse attendue » est la suite de phonèmes annotée dans DALI, avec la perte CTC.
5. **Produire les postériogrammes** : faire passer les chansons de test (Jamendo, MUSDB) dans le modèle entraîné.
6. **Évaluer** : passer ces postériogrammes dans leur `eval.py` et comparer avec le modèle pré-entraîné de jhuang448.

Si l'écart avec le modèle pré-entraîné est faible, c'est terminé. S'il est important, on passe à la distillation.

## Distillation

1. **Données** : les mêmes qu'en from scratch, rien à refaire.
2. **Charger le professeur** : le modèle pré-entraîné de jhuang448 (`checkpoints/checkpoint_Baseline`). On ne l'entraîne pas, on s'en sert seulement pour prédire.
3. **Calculer les corrigés** : faire passer tous les extraits de DALI dans le professeur et sauvegarder ses postériogrammes sur le disque. C'est fait une seule fois.
4. **Modèle** : le même modèle causal qu'en from scratch.
5. **Entraîner** : la même copie de `train.py`, mais la « réponse attendue » devient le postériogramme sauvegardé du professeur, avec la perte KL au lieu de la CTC.
6. **Produire les postériogrammes et évaluer** : exactement comme en from scratch, sur Jamendo et MUSDB, puis comparer les trois modèles.

D'une méthode à l'autre, seules les étapes 3 et 5 de la distillation changent : on calcule les corrigés du professeur, puis on entraîne avec ces corrigés au lieu des annotations.
