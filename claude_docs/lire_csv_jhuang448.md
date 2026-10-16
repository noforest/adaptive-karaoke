Votre intuition est juste : **une frame correspond à un seul phonème** (ou au symbole *blank*). La confusion vient du fait qu'une ligne du CSV n'est pas une frame. C'est un **mot qui s'étend sur plusieurs frames**.

Je dois aussi corriger ce que j'ai dit dans ma réponse précédente : une frame ne dure pas 11,6 ms. Dans `test.py:23`, `resolution = 256 / 22050 * 3` ≈ **34,8 ms par frame**.

### Exemple concret

```
32.531, 32.705, ['θ', 'ɹ', 'uː']   → "through"
```

Le mot dure 32.705 − 32.531 = 0,174 s, soit **5 frames** de 34,8 ms. Le chemin d'alignement attribue un seul état à chaque frame, par exemple :

| frame | 934 | 935 | 936 | 937 | 938 |
| ----- | --- | --- | --- | --- | --- |
| état  | θ   | ɹ   | ɹ   | uː  | uː  |

Il y a 3 phonèmes répartis sur 5 frames, et chaque frame n'a qu'un seul phonème.

### Comment l'alignement est fait

Le softmax ne sert pas à prendre directement le phonème le plus probable à chaque frame. Il produit, pour chaque frame, une **distribution de log-probabilités** sur tous les phonèmes. Ensuite, `utils.alignment` (`utils.py:150-190`) fait une programmation dynamique de type CTC/Viterbi :

- On connaît déjà la séquence de phonèmes des paroles : `θ ɹ uː _ d eɪ z _ …`.
- Les états sont `blank, p1, blank, p2, blank, …` (d'où les indices `2*i+1`).
- À chaque frame, le chemin peut **rester** sur le même phonème, **passer au suivant**, ou passer par un *blank*.
- On choisit le chemin monotone qui maximise la somme des log-probabilités du phonème imposé à chaque frame.

Résultat : chaque frame est assignée à exactement un état, et chaque phonème occupe au moins une frame. Le modèle ne trouve donc pas plusieurs phonèmes dans une frame. Le CSV regroupe simplement les frames consécutives qui appartiennent à un même mot (du premier phonème jusqu'au séparateur suivant). Il n'affiche que le début et la fin de ce bloc, accompagnés de la liste des phonèmes du mot.

Pour obtenir l'alignement frame par frame, il suffit de récupérer la variable `path` dans `utils.alignment`. Chaque paire `[frame, état]` donne directement le phonème attribué à cette frame.
