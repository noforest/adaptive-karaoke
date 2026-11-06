# La perte CTC : c'est quoi, pourquoi elle ?

CTC = *Connectionist Temporal Classification* (Graves et al., 2006).

## Le problème

Le modèle acoustique sort **une distribution sur 72 classes par frame** (une frame = 34,8 ms). Pour l'entraîner avec une cross-entropy classique, il faudrait connaître **le phonème de chaque frame**.

Ce qu'on a :
- **DALI** donne le début et la fin (en secondes) de chaque **note** (≈ syllabe), **mot**, **ligne** et **paragraphe**. Aucun temps au niveau des phonèmes.
- **Le phonémiseur** donne les phonèmes de chaque mot, et `phone_idx` dit quels indices appartiennent à quel mot (`through` → indices 0 à 2, l'indice 3 étant l'espace `␣`).

En combinant les deux, on sait que `θ ɹ uː` tient dans la fenêtre de « through ». On ne sait pas comment ils s'y répartissent :

| frame         | 1 | 2 | 3  | 4  | 5  |
|---------------|---|---|----|----|----|
| possibilité A | θ | ɹ | uː | uː | uː |
| possibilité B | θ | θ | ɹ  | ɹ  | uː |

Couper le mot en parts égales serait faux : en chant, une consonne dure quelques dizaines de ms, alors qu'une voyelle tenue peut durer des secondes. Il manque donc la position des phonèmes **à l'intérieur** du mot.

## L'idée

1. On ajoute une classe **blank ε** (la n°71) qui veut dire « rien de nouveau ici ».
2. Un *chemin* est une suite d'une classe par frame. On le lit en **fusionnant les répétitions**, puis en **supprimant les blanks** :

   `ε n n ε aɪ aɪ aɪ t ε` → `n aɪ t`

3. Plusieurs chemins se lisent comme la même cible. La CTC n'en choisit aucun : elle **additionne leurs probabilités**.

```
probabilité d'un chemin = produit des probabilités de sa classe à chaque frame
P(cible)                = somme des probabilités de tous les chemins qui se lisent comme la cible
perte CTC               = -log P(cible)
```

**Mini-exemple** : cible `a`, 2 frames. Les chemins valides sont `a a`, `a ε` et `ε a`. Si on note p1(a) la probabilité de `a` à la frame 1 :

```
P(a) = p1(a)·p2(a) + p1(a)·p2(ε) + p1(ε)·p2(a)
```

Sur une vraie chanson, le nombre de chemins explose. La somme se calcule quand même efficacement par **programmation dynamique** (forward-backward, en T × L opérations), sur les états `ε, p1, ε, p2, ε, …` décrits dans `transitions_ctc_alignement.md`.

## Pourquoi c'est le bon choix ici

- **Pas besoin du temps des phonèmes** : la suite de phonèmes suffit, et on l'a.
- **Durées variables gérées toutes seules** : un « aɪ » tenu 1 s ou chanté en 100 ms donne la même cible, seul le chemin change.
- **Tolérante aux erreurs de DALI** : le README reconnaît des décalages d'alignement. La CTC n'impose pas de bornes frame par frame, donc un décalage léger ne fausse pas la réponse attendue.
- **Cohérence avec l'alignement final** : à l'inférence, on garde les mêmes états mais on prend le **meilleur** chemin au lieu de la somme (Viterbi), ce qui donne le temps de chaque phonème.
- **Standard** : c'est la perte du modèle de jhuang448, et elle existe telle quelle dans PyTorch.

Les temps des mots ou des lignes restent utiles pour **découper l'audio en extraits courts**. La CTC aligne ensuite les phonèmes dans chaque extrait.

## Limites à connaître

- **Indépendance entre frames** : chaque frame est prédite sans regarder les autres sorties. Le contexte doit venir du réseau (LSTM, convolutions).
- **Sorties « en pics »** : le modèle met ε presque partout et un pic court par phonème (voir `transitions_ctc_alignement.md`). Les débuts sont bien placés, les fins moins.
- **Il faut assez de frames** : au moins une par phonème, plus une pour le ε obligatoire entre deux phonèmes identiques consécutifs. Sinon la perte est infinie, d'où `zero_infinity=True`.

## En PyTorch

```python
ctc = torch.nn.CTCLoss(blank=71, zero_infinity=True)
log_probs = model(x).log_softmax(-1)          # (T, N, 72), attention à l'ordre des axes
loss = ctc(log_probs, targets, input_lengths, target_lengths)
```

Pour la distillation, la CTC est remplacée par une perte KL : le professeur fournit directement une distribution par frame, donc le problème d'alignement disparaît.

## A quoi sert Espilon
**Alors un a prolongé, c'est a a a a ou a ε ε ε ?**

Les deux sont valides et veulent dire la même chose. La CTC accepte les deux à
l'entraînement. Le modèle du dépôt a appris de lui-même à préférer a ε ε ε (le
pic, puis des ε), c'est le comportement « en pics ». C'est pour ça que ① est
rare : 56 fois sur Embers, contre 5 600 pour ε → même ε.

Ton intuition est juste sur un point : a ε peut être un a prolongé. Mais ce
peut aussi être un a court suivi d'un silence. Le ε ne fait pas la différence,
il veut juste dire « rien de nouveau ».

**Prolongé vs Répété**
┌─────────────────┬──────────────────────┬──────────┐
│ ce qu'on chante │   sortie du modèle   │ lu comme │
├─────────────────┼──────────────────────┼──────────┤
│ a prolongé      │ a ε ε ε (ou a a a a) │ un a     │
├─────────────────┼──────────────────────┼──────────┤
│ a répété        │ a ε a                │ deux a   │
└─────────────────┴──────────────────────┴──────────┘
