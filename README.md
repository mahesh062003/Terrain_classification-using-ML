# Terrain Classification Web App

A CNN that classifies terrain photographs into **grassy**, **marshy**, **rocky**
or **sandy**, served through a Flask upload interface.

Built for route and traversability assessment: the terrain under a vehicle or
robot determines whether a path is passable, and it can be read from a single
downward photograph with no additional sensor.

---

## Results

**90.4% test accuracy** on a held-out 778-image test set, four classes.

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| grassy | 0.923 | 0.923 | 0.923 | 91 |
| marshy | 0.869 | 0.894 | 0.881 | 170 |
| rocky | 0.823 | 0.940 | 0.878 | 183 |
| sandy | 0.974 | 0.883 | 0.926 | 334 |
| **accuracy** | | | **0.904** | 778 |
| macro avg | 0.897 | 0.910 | 0.902 | 778 |
| weighted avg | 0.909 | 0.904 | 0.905 | 778 |

Performance holds up across classes despite an imbalanced training set —
`sandy` has nearly four times the training images of `grassy` (779 vs 212), yet
macro-F1 (0.902) and accuracy (0.904) are within two thousandths of each other,
and `grassy` has the highest per-class F1.

The weakest cell is `rocky` precision (0.823) against its recall (0.940): the
model over-predicts `rocky`, pulling in some `marshy` and `sandy`. `sandy` shows
the mirror image at 0.974 precision and 0.883 recall, so the main confusion runs
in both directions between rocky and sandy.

### Training distribution

| Class | Train | Test |
|---|---|---|
| sandy | 779 (42%) | 334 |
| rocky | 426 (23%) | 183 |
| marshy | 396 (21%) | 170 |
| grassy | 212 (11%) | 91 |

---

## Usage

```bash
pip install -r requirements.txt
# download model.h5 from the v1.0.0 release into this directory first
python app.py        # http://localhost:5000
```

Upload an image on the Upload page and the predicted terrain is rendered back.
`app.py` resizes to 224 × 224, rescales to [0, 1], and takes the argmax over
the four-class softmax.

> Evaluating the model yourself: the test generator must be built with
> `shuffle=False`. `model.predict()` consumes the generator in order while
> `.classes` returns labels in directory order, so a shuffled generator
> misaligns the two and scores near chance.

## Model weights

The trained weights are **not committed** — download `model.h5` (255 MB) from
the [v1.0.0 release](https://github.com/mahesh062003/Terrain_classification-using-ML/releases/tag/v1.0.0)
and place it in the repository root before running `app.py`.

Release assets are used rather than Git LFS because the free LFS tier allows
1 GB of bandwidth per month, which a file this size exhausts in four clones.

## Dataset

Four class directories under `dataset-1/{train,test}/`. Not committed —
[download from Drive](https://drive.google.com/drive/folders/1OOc-ceo7eF1stmN8JTUL4JNnnqNDklQ-?usp=drive_link).

```
dataset-1/
  train/  grassy/ marshy/ rocky/ sandy/
  test/   grassy/ marshy/ rocky/ sandy/
```

## Stack

Python · TensorFlow/Keras · Flask · NumPy
