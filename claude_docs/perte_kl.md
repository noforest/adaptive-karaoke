# La perte KL (vs CTC)

**KL** : écart entre deux distributions.
`KL(p‖q) = Σ p·log(p/q)` → 0 si l'élève `q` copie le professeur `p`, frame par frame.

**CTC** : cible = texte sans temps → somme sur tous les alignements possibles.

## Pourquoi un écart entre deux distributions ?

Parce que le prof donne une **distribution** par frame, pas une seule étiquette.

- Le but de la distillation : l'élève **imite** le prof. Ça revient à rendre sa distribution identique à celle du prof.
- Les probas secondaires du prof (ex. 70 % /a/, 20 % /e/) disent que /a/ et /e/ se ressemblent. L'argmax seul (/a/) perd cette info.
- La KL est la mesure standard de cet écart. Elle vaut 0 quand les deux distributions sont identiques, donc la minimiser = copier le prof.
