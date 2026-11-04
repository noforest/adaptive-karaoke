Le CRNN en une phrase : un CNN reconnaît des motifs sonores locaux dans chaque
petit morceau du spectrogramme, puis un LSTM les relie dans le temps pour
décider quel phonème est chanté, frame par frame.

**Les étapes :**

- Entrée : le spectrogramme mel. Une image du son, avec le temps en largeur
  (une colonne par frame) et 80 bandes de fréquence en hauteur.
- CNN (3 couches de convolution). Chaque couche fait glisser de petits filtres
  sur le spectrogramme pour repérer des motifs : attaques de consonnes,
  formants de voyelles, etc. Les couches successives combinent les motifs
  simples en motifs plus riches (32, puis 64, puis 128 filtres).
- Convolution causale. Le filtre ne regarde que la frame actuelle et les
  précédentes, jamais les suivantes. Pour garder la même longueur, on ajoute
  des zéros au début (padding à gauche).
- Pooling sur la fréquence seulement. On divise le nombre de bandes par 2 après
  les couches (80 → 40 → 20) pour compacter l’information, mais on ne touche
  pas au temps : on garde une sortie par frame.
- Aplatissement. À chaque frame, on met bout à bout les 128 canaux × 20 bandes,
  ce qui donne un vecteur de 2560 valeurs.
- LSTM unidirectionnel. Il lit ces vecteurs frame par frame, de gauche à
  droite, en gardant une mémoire du passé. Sans cette mémoire, une frame de 10
  ms isolée est ambiguë. Avec, le modèle sait par exemple qu’il vient
  d’entendre « hel » et que « lo » est probable.
- Linéaire + softmax. Pour chaque frame, on transforme la sortie du LSTM en une
  probabilité par phonème, avec une somme égale à 1.
- Sortie : le postériogramme. Un tableau phonèmes × temps, qu’on aligne ensuite
  sur les paroles connues.

Le LSTM simplement :

- C’est un réseau qui lit une séquence pas à pas et garde une mémoire qu’il met
  à jour à chaque pas.
- Un réseau récurrent simple oublie vite : l’information ancienne se dilue à
  chaque pas.
- Le LSTM ajoute une mémoire longue et trois « portes » apprises qui décident
  quoi oublier, quoi écrire et quoi lire. Il peut ainsi garder une information
  utile sur beaucoup de frames.

Pourquoi tout est causal :

- On veut suivre le chanteur en direct, donc la sortie à l’instant t ne peut
  dépendre que de ce qui a déjà été chanté.
- Convolutions causales et LSTM unidirectionnel garantissent exactement ça.
- Un LSTM bidirectionnel serait souvent plus précis, mais il a besoin de la fin
  de la chanson pour produire sa sortie, donc il est impossible à utiliser en
  temps réel.
