# Loading and Preparing Datasets with TensorFlow Datasets (TFDS)

## 1. Project Title

**Efficient Dataset Loading and Preprocessing using TensorFlow Datasets**

---

## 2. Problem Statement and Goal of Project

Access to standardized, ready-to-use datasets is critical for deep learning research and prototyping.
This project demonstrates how to **load, inspect, preprocess, and prepare datasets** using TensorFlow Datasets (TFDS), enabling quick experimentation without manual dataset handling.

---

## 3. Solution Approach

The notebook follows a clear workflow:

1. **Loading datasets from TFDS** – Use `tfds.load()` to fetch datasets with optional train/test splits.
2. **Inspecting dataset metadata** – View dataset info, features, and label mappings.
3. **Preprocessing** – Apply image resizing, normalization, and type conversion.
4. **Batching, shuffling, and caching** – Build optimized input pipelines for training.
5. **Visualization** – Display sample images with labels for verification.

---

## 4. Technologies & Libraries

From the code:

* **TensorFlow** – Model compatibility, preprocessing, and pipeline integration.
* **TensorFlow Datasets (TFDS)** – Dataset loading and metadata management.
* **Matplotlib** – Visualization of images and labels.
* **NumPy** – Optional numerical handling.

---

## 5. Description about Dataset

The notebook uses datasets from **TensorFlow Datasets (TFDS)** — dataset choice (e.g., MNIST, CIFAR-10) depends on `tfds.load()` parameters in the code.
No manual dataset download or external file handling is required.

---

## 6. Installation & Execution Guide

**Requirements:**

```bash
pip install tensorflow tensorflow-datasets matplotlib numpy
```

**Run the notebook:**

```bash
jupyter notebook tfds.ipynb
```

or in JupyterLab:

```bash
jupyter lab tfds.ipynb
```

---

## 7. Key Results / Performance

* Successfully loaded a TFDS dataset with both training and testing splits.
* Visualized sample images with correct labels for dataset validation.
* Built an **optimized input pipeline** using batching, shuffling, caching, and prefetching.

Example snippet:

```python
train_ds, test_ds = tfds.load('mnist', split=['train', 'test'], as_supervised=True)
train_ds = train_ds.shuffle(1024).batch(32).prefetch(tf.data.AUTOTUNE)
```

---

## 8. Screenshots / Sample Out

**Sample dataset visualization:**

```
Image: <tf.Tensor: shape=(28, 28, 1), dtype=uint8>
Label: 7
```

*(Accompanied by plotted image using Matplotlib in the notebook)*

---

## 9. Additional Learnings / Reflections

* TFDS provides **instant access** to a wide variety of datasets with minimal code.
* Integrating TFDS with `tf.data` transformations ensures optimal GPU utilization.
* Always inspect a dataset visually before training to verify preprocessing correctness.
* TFDS is highly useful for benchmarking and educational purposes.

> 💡 *Some interactive outputs (e.g., plots, widgets) may not display correctly on GitHub. If so, please view this notebook via [nbviewer.org](https://nbviewer.org) for full rendering.*

---

## 👤 Author

**Mehran Asgari**
**Email:** [imehranasgari@gmail.com](mailto:imehranasgari@gmail.com)
**GitHub:** [https://github.com/imehranasgari](https://github.com/imehranasgari)

---

## 📄 License

This project is licensed under the Apache 2.0 License – see the `LICENSE` file for details.

---

