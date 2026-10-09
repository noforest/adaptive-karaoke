## Notes

### Techniques
- **From scratch** : modèle entraîné sur DALI sans professeur
- **Distillation** : modèle entraîné à imiter le modèle pré-entraîné du repo suivant [https://github.com/jhuang448/LyricsAlignment-Multilingual/](https://github.com/jhuang448/LyricsAlignment-Multilingual/)

### Ordre recommandé
1. Entraîner from scratch, puis brancher le DTW
2. Comparer au modèle pré-entraîné
3. Distillation seulement si écart est important

### Liens
- [https://github.com/jhuang448/LyricsAlignment-Multilingual/](https://github.com/jhuang448/LyricsAlignment-Multilingual/)
- [https://github.com/charlesbrazier/opera_tracker/blob/master/fast_dtw.pyx](https://github.com/charlesbrazier/opera_tracker/blob/master/fast_dtw.pyx)
