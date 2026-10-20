
# Lab 2 : PyTorch, CNN et Fine-tuning sur MNIST

Prise en main de PyTorch (mode GPU) : CNN from scratch et fine-tuning d'AlexNet et VGG16 sur MNIST.

## Contenu
- CNN (2 convolutions, 2 max-pooling, 2 couches fully connected, Dropout)
- Fine-tuning d'AlexNet et VGG16 (couches gelées vs entraînement complet)
- Comparaison : accuracy, F1, loss, temps, paramètres

## Résultats

| Modèle | Accuracy | F1 macro | Temps (s) | Paramètres (M) |
|---|---|---|---|---|
| CNN (from scratch) | 0,9929 | 0,9928 | 142,2 | 0,42 |
| AlexNet (gelé) | 0,9750 | 0,9751 | 59,6 | 57,04 |
| AlexNet (complet) | 0,9845 | 0,9840 | 71,8 | 57,04 |
| VGG16 (gelé) | 0,9770 | 0,9772 | 228,5 | 134,30 |
| VGG16 (complet) | 0,9850 | 0,9849 | 534,2 | 134,30 |

![Comparaison](images/comparaison.png)

## Conclusion
Sur MNIST, un petit CNN surpasse AlexNet et VGG16 pour un coût bien inférieur. Le transfer learning est plus pertinent pour des datasets complexes ou de petite taille.

## Lancer le projet
```bash
pip install -r requirements.txt
jupyter notebook lab2_mnist.ipynb
```
Un GPU est recommandé (Google Colab ou Kaggle).
