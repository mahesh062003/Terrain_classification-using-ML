# Terrain Classification Web App

A CNN that classifies terrain photographs into **grassy**, **marshy**, **rocky**
or **sandy**, served through a Flask upload interface.

Built for route and traversability assessment: the terrain type under a vehicle
or robot determines whether a path is passable, and it can be read from a
single downward photograph without any additional sensor.

---

## Result

**90.4% test accuracy** on a held-out 778-image test set, four classes.
Measured by re-running the trained `model.h5` against `dataset-1/test`:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| grassy | 0.923 | 0.923 | 0.923 | 91 |
| marshy | 0.869 | 0.894 | 0.881 | 170 |
| rocky | 0.823 | 0.940 | 0.878 | 183 |
| sandy | 0.974 | 0.883 | 0.926 | 334 |
| **accuracy** | | | **0.904** | 778 |
| macro avg | 0.897 | 0.910 | 0.902 | 778 |
| weighted avg | 0.909 | 0.904 | 0.905 | 778 |

**Performance is even across classes despite an imbalanced training set.**
`sandy` has nearly four times the training images of `grassy` (779 vs 212),
so the obvious worry is a model that simply learns to answer "sandy". That is
not what happened: macro-F1 (0.902) and accuracy (0.904) are within two
thousandths of each other, and `grassy` — the smallest class — has the
*highest* per-class F1 at 0.923. The imbalance is real in the data but did not
translate into biased predictions.

The weakest cell is `rocky` precision at 0.823 against 0.940 recall: the model
over-predicts `rocky`, pulling in some `marshy` and `sandy` images. `sandy`
shows the mirror image — 0.974 precision, 0.883 recall — so the main confusion
is rocky/sandy in both directions.

### A correction to an earlier reported figure

An evaluation cell in `notebooks/model-final.ipynb` previously reported **30.3%
accuracy with a macro-F1 of 0.25** — barely above the 25% chance baseline for
four classes. That number was wrong, and the cause was a bug rather than a
model failure:

```python
test_generator = test_datagen.flow_from_directory(...)   # shuffle defaults to True
predictions   = model.predict(test_generator, ...)       # consumed in SHUFFLED order
true_classes  = test_generator.classes                   # returned in DIRECTORY order
```

`flow_from_directory` shuffles by default, so `predict()` walked the images in
one order while `.classes` reported labels in another. The two arrays were
compared element-wise while misaligned, which scores at approximately chance no
matter how good the model is.

This was confirmed rather than assumed. Deliberately misaligning the labels
against the same predictions reproduces **31.1%** — matching the original 30.3%
and establishing the mechanism. The same predictions scored correctly give
90.4%.

`model.evaluate()` was never affected: it pairs each batch's inputs with that
same batch's labels, so shuffling cannot desynchronise them.

The notebook now passes `shuffle=False` on the evaluation generator, so the
classification report and confusion matrix it produces are meaningful.

### Training distribution

| Class | Train | Test |
|---|---|---|
| sandy | 779 (42%) | 334 |
| rocky | 426 (23%) | 183 |
| marshy | 396 (21%) | 170 |
| grassy | 212 (11%) | 91 |

---

## Running it

```bash
pip install -r requirements.txt
python app.py        # http://localhost:5000
```

Upload an image on the Upload page; the predicted terrain is rendered back.

`app.py` resizes to 224x224, rescales to [0,1], and takes the argmax over the
four-class softmax.

## Model weights

`model.h5` (267 MB) is tracked with **Git LFS** (see `.gitattributes`). Note
that GitHub's free LFS tier allows 1 GB of storage and 1 GB of bandwidth per
month — a 267 MB file consumes that bandwidth quickly across clones. Attaching
the weights to a GitHub Release instead would avoid the limit entirely.

## Dataset

Four class directories under `dataset-1/{train,test}/`. Not committed —
[download from Drive](https://drive.google.com/drive/folders/1OOc-ceo7eF1stmN8JTUL4JNnnqNDklQ-?usp=drive_link).

```
dataset-1/
  train/  grassy/ marshy/ rocky/ sandy/
  test/   grassy/ marshy/ rocky/ sandy/
```

## Notes on repository hygiene

Two things fixed while preparing this:

- `Requirements.txt` was capitalised while the install instruction said
  `requirements.txt`. That works on Windows and **fails on Linux and macOS**,
  where filenames are case-sensitive. Renamed to lowercase.
- A 536 MB `NH-HAZE_params_0100000.pt` file sat in the project directory. It is
  a pretrained **image-dehazing** model, is not referenced by any code here,
  and is not mine — it is now gitignored rather than published.

## Stack

Python · TensorFlow/Keras · Flask · NumPy
