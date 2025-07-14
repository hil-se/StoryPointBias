import numpy as np
import scipy
import os
from sentence_transformers import SentenceTransformer
from sklearn.svm import LinearSVC
import pandas as pd
import tensorflow as tf
from metrics import Metrics
from pdb import set_trace

def loadData(dataName="jirasoftware_filtered"):
    path = "../Data/"
    df = pd.read_csv(path+dataName+".csv")
    return df

def process(dataName="jirasoftware_filtered", sensitive="is_internal"):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    data = loadData(dataName=dataName)
    embeddings = model.encode(data["text"])
    embedded = pd.DataFrame({"X": embeddings.tolist(), "Y": data["storypoint"], "A": data[sensitive], "split_mark": data["split_mark"]})
    return embedded


def build_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_dim,)),

        # tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5)),
        # tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.Dropout(0.3),
        #
        # tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5)),
        # tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.Dropout(0.2),
        #
        # tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5)),
        # tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dense(1, activation="linear")
    ])

    return model

class ComparativeModel(tf.keras.Model):
    def __init__(self, encoder, w_loss="hinge", **kwargs):
        super(ComparativeModel, self).__init__(**kwargs)
        self.encoder = encoder
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.w_loss = w_loss

    @property
    def metrics(self):
        return [self.loss_tracker]

    def call(self, features, trainable=True):
        encodings_A = self.encoder(features["A"], training=trainable)
        encodings_B = self.encoder(features["B"], training=trainable)
        return tf.subtract(encodings_A, encodings_B)

    def compute_loss(self, y, diff):
        y = tf.cast(y, tf.float32)
        loss = tf.reduce_mean(tf.math.maximum(0.0, 1.0 - (y * tf.squeeze(diff))))
        return loss

    def compute_loss_square(self, y, diff):
        y = tf.cast(y, tf.float32)
        loss = tf.reduce_mean(tf.square(tf.math.maximum(0.0, 1.0 - (y * tf.squeeze(diff)))))
        return loss

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            diff = self(x)
            if self.w_loss == "hinge":
                loss = self.compute_loss(y, diff)
            else:
                loss = self.compute_loss_square(y, diff)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        x, y = data
        diff = self(x)
        loss = self.compute_loss(y, diff)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def predict(self, X):
        """Predicts preference between two items."""
        return np.array(self.encoder(np.array(X.tolist())))

def generate_comparative_judgments(train_list, N=1):
    m = len(train_list)
    train_list.index = range(m)
    features = {"A": [], "B": [], "Label": []}
    seen = set()
    for i in range(m):
        n = 0
        while n < N:
            j = np.random.randint(0, m)
            if (i,j) in seen or (j,i) in seen:
                continue
            if train_list["Y"][i] > train_list["Y"][j]:
                features["A"].append(train_list["X"][i])
                features["B"].append(train_list["X"][j])
                features["Label"].append(1.0)
                n += 1
            elif train_list["Y"][i] < train_list["Y"][j]:
                features["A"].append(train_list["X"][i])
                features["B"].append(train_list["X"][j])
                features["Label"].append(-1.0)
                n += 1
            seen.add((i, j))
    features = {key: np.array(features[key]) for key in features}
    return features


def generate_all(train_list):
    m = len(train_list)
    train_list.index = range(m)
    features = {"A": [], "B": [], "Label": []}
    seen = set()
    for i in range(m):
        for j in range(m):
            if (i, j) in seen or (j, i) in seen:
                continue
            if train_list["Y"][i] > train_list["Y"][j]:
                features["A"].append(train_list["X"][i])
                features["B"].append(train_list["X"][j])
                features["Label"].append(1.0)
            elif train_list["Y"][i] < train_list["Y"][j]:
                features["A"].append(train_list["X"][i])
                features["B"].append(train_list["X"][j])
                features["Label"].append(-1.0)
            seen.add((i, j))
    features = {key: np.array(features[key]) for key in features}
    return features

def comparative_learning(train_x, test_x, features, loss = "hinge", val = None):
    encoder = build_model((train_x.shape[1]))
    de = ComparativeModel(encoder=encoder, w_loss = loss)

    checkpoint_path = "checkpoint/STD_comp.keras"
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    # de.compile(optimizer="SGD", loss=tf.keras.losses.Hinge())
    de.compile(optimizer="adam")
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=100, factor=0.3, min_lr=1e-6, verbose=1)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor="val_loss", save_best_only=True,
                                                    save_weights_only=True, verbose=1, )

    if val:
        val_data = (val, val["Label"])
    else:
        val_data = None

    # Train model
    history = de.fit(features, features["Label"], validation_data=val_data, epochs=300, batch_size=32, callbacks=[reduce_lr, checkpoint],
                     verbose=1)
    print("\nLoading best checkpoint model...")
    de.load_weights(checkpoint_path)

    preds_test = de.predict(test_x).flatten()
    preds_train = de.predict(train_x).flatten()
    return preds_test, preds_train

def LinearSVM(train_x, test_x, features, loss = "hinge"):
    train_feature = features["A"] - features["B"]
    model = LinearSVC(loss = loss)
    model.fit(train_feature, features["Label"])
    preds_test = model.decision_function(test_x).flatten()
    preds_train = model.decision_function(train_x).flatten()
    return preds_test, preds_train

def train_and_test(dataname, N=1):

    repeats = 1

    data = process(dataName=dataname, sensitive="is_internal")
    train_x = np.array(data[data["split_mark"] == "train"]["X"].tolist())
    train_y = np.array(data[data["split_mark"] == "train"]["Y"].tolist())
    test_x = np.array(data[data["split_mark"] == "test"]["X"].tolist())
    test_y = np.array(data[data["split_mark"] == "test"]["Y"].tolist())
    features = generate_comparative_judgments(data[data["split_mark"] == "train"], N=N)
    features_val = generate_comparative_judgments(data[data["split_mark"] == "val"], N=N)

    train_results = []
    test_results = []

    # Comparative learning
    for i in range(repeats):
        preds_test, preds_train = comparative_learning(train_x, test_x, features, val = features_val)

        m_train = Metrics(train_y, preds_train)
        m_test = Metrics(test_y, preds_test)
        result_train = {"Data": dataname, "N": N, "Treatment": "CL",
                        "Pearson": m_train.pearsonr().statistic, "Spearman": m_train.spearmanr().statistic,
                        # "Isep": m_train.Isep(np.array(data[data["split_mark"] == "train"]["A"])),
                        # "Csep": m_train.Csep(np.array(data[data["split_mark"] == "train"]["A"])),
                        # "gAOD": m_train.gAOD(np.array(data[data["split_mark"] == "train"]["A"]))
                        }
        result_test = {"Data": dataname, "N": N, "Treatment": "CL",
                       "Pearson": m_test.pearsonr().statistic, "Spearman": m_test.spearmanr().statistic,
                       # "Isep": m_test.Isep(np.array(data[data["split_mark"] == "test"]["A"])),
                       # "Csep": m_test.Csep(np.array(data[data["split_mark"] == "test"]["A"])),
                       # "gAOD": m_test.gAOD(np.array(data[data["split_mark"] == "test"]["A"]))
                       }

        train_results.append(result_train)
        test_results.append(result_test)


    # # Comparative learning2
    # for i in range(repeats):
    #     preds_test, preds_train = comparative_learning(train_x, test_x, features, loss = "hinge_square")
    #
    #     m_train = Metrics(train_y, preds_train)
    #     m_test = Metrics(test_y, preds_test)
    #     result_train = {"Data": dataname, "N": N, "MAE": m_train.mae(), "Treatment": "CL2",
    #                     "Pearson": m_train.pearsonr().statistic, "Spearman": m_train.spearmanr().statistic,
    #                     # "Isep": m_train.Isep(np.array(data[data["split_mark"] != "test"]["A"])),
    #                     # "Csep": m_train.Csep(np.array(data[data["split_mark"] != "test"]["A"])),
    #                     # "gAOD": m_train.gAOD(np.array(data[data["split_mark"] != "test"]["A"]))
    #                     }
    #     result_test = {"Data": dataname, "N": N, "MAE": m_test.mae(), "Treatment": "CL2",
    #                    "Pearson": m_test.pearsonr().statistic, "Spearman": m_test.spearmanr().statistic,
    #                    # "Isep": m_test.Isep(np.array(data[data["split_mark"] == "test"]["A"])),
    #                    # "Csep": m_test.Csep(np.array(data[data["split_mark"] == "test"]["A"])),
    #                    # "gAOD": m_test.gAOD(np.array(data[data["split_mark"] == "test"]["A"]))
    #                    }
    #     train_results.append(result_train)
    #     test_results.append(result_test)

    # Linear SVM
    # preds_test, preds_train = LinearSVM(train_x, test_x, features, loss = "hinge")
    # m_train = Metrics(train_y, preds_train)
    # m_test = Metrics(test_y, preds_test)
    # result_train = {"Data": dataname, "N": N, "MAE": m_train.mae(), "Treatment": "lsvm",
    #                 "Pearson": m_train.pearsonr().statistic, "Spearman": m_train.spearmanr().statistic,
    #                 # "Isep": m_train.Isep(np.array(data[data["split_mark"] == "train"]["A"])),
    #                 # "Csep": m_train.Csep(np.array(data[data["split_mark"] == "train"]["A"])),
    #                 # "gAOD": m_train.gAOD(np.array(data[data["split_mark"] == "train"]["A"]))
    #                  }
    # result_test = {"Data": dataname, "N": N, "MAE": m_test.mae(), "Treatment": "lsvm",
    #                "Pearson": m_test.pearsonr().statistic, "Spearman": m_test.spearmanr().statistic,
    #                # "Isep": m_test.Isep(np.array(data[data["split_mark"] == "test"]["A"])),
    #                # "Csep": m_test.Csep(np.array(data[data["split_mark"] == "test"]["A"])),
    #                # "gAOD": m_test.gAOD(np.array(data[data["split_mark"] == "test"]["A"]))
    #                 }
    # train_results.append(result_train)
    # test_results.append(result_test)

    # for i in range(repeats):
    #     preds_test, preds_train = LinearSVM(train_x, test_x, features, loss="squared_hinge")
    #     m_train = Metrics(train_y, preds_train)
    #     m_test = Metrics(test_y, preds_test)
    #     result_train = {"Data": dataname, "N": N, "MAE": m_train.mae(), "Treatment": "lsvm2",
    #                      "Pearson": m_train.pearsonr().statistic, "Spearman": m_train.spearmanr().statistic,
    #                      # "Isep": m_train.Isep(np.array(data[data["split_mark"] != "test"]["A"])),
    #                      # "Csep": m_train.Csep(np.array(data[data["split_mark"] != "test"]["A"])),
    #                      # "gAOD": m_train.gAOD(np.array(data[data["split_mark"] != "test"]["A"]))
    #                      }
    #     result_test = {"Data": dataname, "N": N, "MAE": m_test.mae(), "Treatment": "lsvm2",
    #                     "Pearson": m_test.pearsonr().statistic, "Spearman": m_test.spearmanr().statistic,
    #                     # "Isep": m_test.Isep(np.array(data[data["split_mark"] == "test"]["A"])),
    #                     # "Csep": m_test.Csep(np.array(data[data["split_mark"] == "test"]["A"])),
    #                     # "gAOD": m_test.gAOD(np.array(data[data["split_mark"] == "test"]["A"]))
    #                     }
    #     train_results.append(result_train)
    #     test_results.append(result_test)

    return train_results, test_results

if __name__ == "__main__":
    data = "jirasoftware_filtered"
    results_train = []
    results_test = []
    # for n in [1, 2, 3, 4, 5, 10]:
    for n in [1]:
        for _ in range(10):
            result_train, result_test = train_and_test(data, N=n)
            results_train.extend(result_train)
            results_test.extend(result_test)
    results_train = pd.DataFrame(results_train)
    print(results_train)
    results_train.to_csv("../Results/STD_compval_train.csv", index=False)
    results_test = pd.DataFrame(results_test)
    print(results_test)
    results_test.to_csv("../Results/STD_compval_test.csv", index=False)




