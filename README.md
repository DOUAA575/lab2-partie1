# Lab 2 : PyTorch, CNN, Fine-tuning et Vision Transformer sur MNIST

Prise en main de PyTorch en mode GPU pour la vision par ordinateur : CNN from scratch, fine-tuning d'AlexNet et VGG16, et Vision Transformer (ViT) implémenté from scratch. Tous les modèles sont comparés sur MNIST.

## Contenu

**Partie 1 : CNN et fine-tuning**
- CNN from scratch : 2 convolutions (3×3, padding 1), 2 max-pooling, 2 couches fully connected, Dropout 0,5, Adam (lr = 0,001, weight decay 1e-4)
- Fine-tuning d'AlexNet et VGG16 pré-entraînés (ImageNet) : couches gelées vs entraînement complet

**Partie 2 : Vision Transformer**
- ViT from scratch : 49 patches de 4×4, token `[CLS]`, encodage positionnel sinusoïdal, 2 blocs Transformer (4 têtes d'attention, d = 32)

**Comparaison** : accuracy, F1 macro, loss, temps d'entraînement, nombre de paramètres.

> Note : la question sur Faster R-CNN n'est pas traitée dans ce dépôt, car c'est un modèle de détection d'objets, peu adapté à la classification MNIST.

## Résultats

| Modèle | Accuracy | F1 macro | Loss | Temps (s) | Paramètres (M) |
|---|---|---|---|---|---|
| **CNN (from scratch)** | **0,9926** | **0,9926** | 0,0272 | 137,1 | 0,422 |
| AlexNet (gelé) | 0,9750 | 0,9751 | 0,0584 | 59,1 | 57,045 |
| AlexNet (complet) | 0,9845 | 0,9840 | 0,0355 | 71,5 | 57,045 |
| VGG16 (gelé) | 0,9770 | 0,9772 | 0,0244 | 209,8 | 134,302 |
| VGG16 (complet) | 0,9870 | 0,9869 | 0,0411 | 476,7 | 134,302 |
| ViT (from scratch) | 0,9674 | 0,9671 | 0,0773 | 156,1 | 0,026 |

Les losses ne sont pas strictement comparables : le CNN et le ViT sont entraînés sur 60 000 images, AlexNet et VGG16 sur un sous-ensemble de 10 000 images (224×224).


## Conclusions

- **CNN** : le meilleur compromis sur MNIST (99,26 %) avec seulement 0,42 M de paramètres. Ses biais inductifs (localité, partage des poids) sont adaptés aux images.
- **Fine-tuning** : le fine-tuning complet apporte environ +1 point par rapport au gel des couches, mais AlexNet et VGG16 restent plus lourds et ne dépassent pas le CNN sur un dataset aussi simple.
- **ViT** : 96,74 % avec seulement 26 k paramètres. Il est le plus compact, mais reste derrière le CNN, car il doit apprendre à partir des données ce que le CNN suppose déjà. La loss baissait encore à la fin : plus d'epochs ou un modèle plus grand l'amélioreraient.
- Le transfer learning et les Transformers sont plus pertinents pour des datasets volumineux, complexes ou de petite taille, pas pour MNIST.

## Structure du dépôt

```
lab2-pytorch-mnist/
├── lab2_mnist_vit.ipynb   # notebook complet (Partie 1 + Partie 2)
├── requirements.txt
├── README.md           
```

## Lancer le projet

```bash
git clone https://github.com/VOTRE-PSEUDO/lab2.git
cd lab2-pytorch-mnist
pip install -r requirements.txt
jupyter notebook lab2-tp.ipynb
```

Un GPU est fortement recommandé (Google Colab : *Runtime → Change runtime type → T4 GPU*, ou Kaggle). MNIST est téléchargé automatiquement par `torchvision`. Exécutez les cellules dans l'ordre, de haut en bas.

## Environnement

- Python 3.10+, PyTorch, GPU NVIDIA T4 (Colab/Kaggle)
- Graine fixée : `torch.manual_seed(42)` (de petites variations restent possibles d'une exécution à l'autre)

## Références

- Dosovitskiy et al., *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*, 2020
- Dataset MNIST : https://www.kaggle.com/datasets/hojjatk/mnist-dataset
