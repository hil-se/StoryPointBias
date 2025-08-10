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

    # initial_learning_rate = 0.001
    # lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    #     initial_learning_rate, decay_steps=5000, alpha=0.0001
    # )
    # optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

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

    # def test_step(self, features):
    #     encodings_A, encodings_B = self(features)
    #     loss = self.compute_loss(encodings_A, encodings_B, features["Label"])
    #     self.loss_tracker.update_state(loss)
    #     return {"loss": self.loss_tracker.result()}

    def predict(self, X):
        """Predicts preference between two items."""
        return np.array(self.encoder(np.array(X.tolist())))

def generate_comparative_judgments(train_list, N=10):
    m = len(train_list)
    features = {"A": [], "B": [], "Label": []}
    seen = set()
    n = 0
    while n < N:
        i = np.random.randint(0, m)
        j = np.random.randint(0, m)
        if (i,j) in seen or (j,i) in seen or i == j:
            continue
        if train_list["Y"][i] > train_list["Y"][j]:
            features["A"].append(train_list["X"][i])
            features["B"].append(train_list["X"][j])
            features["Label"].append(1.0)
            # features.append({"A": train_list["A"][i], "B": train_list["A"][j], "Label": 1.0})
            n += 1
            seen.add((i, j))
        elif train_list["Y"][i] < train_list["Y"][j]:
            features["A"].append(train_list["X"][i])
            features["B"].append(train_list["X"][j])
            features["Label"].append(-1.0)
            # features.append({"A": train_list["A"][i], "B": train_list["A"][j], "Label": -1.0})
            n += 1
            seen.add((i, j))
    features = {key: np.array(features[key]) for key in features}
    return features, seen

def comparative_learning(train_x, test_x, features, loss = "hinge"):
    encoder = build_model((train_x.shape[1]))
    de = ComparativeModel(encoder=encoder, w_loss = loss)

    checkpoint_path = "checkpoint/STD_comp.keras"
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    # de.compile(optimizer="SGD", loss=tf.keras.losses.Hinge())
    de.compile(optimizer="adam")
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=100, factor=0.3, min_lr=1e-6, verbose=1)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor="loss", save_best_only=True,
                                                    save_weights_only=True, verbose=1, )
    # Train model
    history = de.fit(features, features["Label"], epochs=100, batch_size=3200, callbacks=[reduce_lr, checkpoint],
                     verbose=1)
    print("\nLoading best checkpoint model...")
    de.load_weights(checkpoint_path)

    preds_test = de.predict(test_x).flatten()
    preds_train = de.predict(train_x).flatten()
    return preds_test, preds_train

def acquire(seen, preds_train, y, step = 10):
    pairs = {}
    n = len(preds_train)
    for i in range(n):
        for j in range(i+1, n):
            if (i,j) in seen or (j,i) in seen or y[i]==y[j]:
                continue
            pairs[(i,j)] = np.abs(preds_train[i] - preds_train[j])
    return list(dict(sorted(pairs.items(), key=lambda item: item[1])).keys())[:step]

def active(dataname, init = 10, step = 10):

    data = process(dataName=dataname, sensitive="is_internal")
    train_x = np.array(data[data["split_mark"] != "test"]["X"].tolist())
    train_data = data[data["split_mark"] != "test"]
    test_x = np.array(data[data["split_mark"] == "test"]["X"].tolist())
    test_y = np.array(data[data["split_mark"] == "test"]["Y"].tolist())
    features, seen = generate_comparative_judgments(train_data, N=init)
    n = len(seen)
    results = []

    while n < len(train_data):
        # Comparative learning
        preds_test, preds_train = comparative_learning(train_x, test_x, features)
        m_test = Metrics(test_y, preds_test)
        result_test = {"Data": dataname, "N": n, "MAE": m_test.mae(), "Treatment": "CL",
                       "Pearson": m_test.pearsonr().statistic, "Spearman": m_test.spearmanr().statistic
                       }
        results.append(result_test)
        to_add = acquire(seen, preds_train, train_data["Y"], step = step)
        for tup in to_add:
            if train_data["Y"][tup[0]] > train_data["Y"][tup[1]]:
                features["A"] = np.append(features["A"], [train_data["X"][tup[0]]], axis = 0)
                features["B"] = np.append(features["B"], [train_data["X"][tup[1]]], axis = 0)
                features["Label"] = np.append(features["Label"], 1.0)
                # features.append({"A": train_list["A"][i], "B": train_list["A"][j], "Label": 1.0})
                n += 1
            elif train_data["Y"][tup[0]] < train_data["Y"][tup[1]]:
                features["A"] = np.append(features["A"], [train_data["X"][tup[0]]], axis = 0)
                features["B"] = np.append(features["B"], [train_data["X"][tup[1]]], axis = 0)
                features["Label"] = np.append(features["Label"], -1.0)
                # features.append({"A": train_list["A"][i], "B": train_list["A"][j], "Label": -1.0})
                n += 1
            seen.add(tup)


    return results

if __name__ == "__main__":
    data = "jirasoftware_filtered"
    results = active(data, init=10, step = 10)
    results = pd.DataFrame(results)
    print(results)
    results.to_csv("../Results/STD_comp_active.csv", index=False)




