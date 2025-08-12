import numpy as np
import os
from sentence_transformers import SentenceTransformer
import pandas as pd
import tensorflow as tf
from metrics import Metrics
from density_balance import DensityBalance
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
        tf.keras.layers.Dense(1, activation="linear")
    ])

    model.compile(
        optimizer='adam',
        loss="mae",
        # loss=tf.keras.losses.Huber(delta=1.0),
        metrics=['mae']
    )

    return model


def active(dataname, treatment = "None", init = 10, step = 10, delta = 5.0):
    data = process(dataname, "is_internal")
    train_x = np.array(data[data["split_mark"] != "test"]["X"].tolist())
    train_y = np.array(data[data["split_mark"] != "test"]["Y"].tolist())
    test_x = np.array(data[data["split_mark"] == "test"]["X"].tolist())
    test_y = np.array(data[data["split_mark"] == "test"]["Y"].tolist())
    pool = list(range(len(train_y)))
    inds = list(np.random.choice(pool, init, replace=False))
    pool = list(set(pool) - set(inds))
    n = init
    results = []

    model = build_model((train_x.shape[1]))
    while pool:
        model = train(model, train_x[inds], train_y[inds], treatment = treatment)
        preds_test = model.predict(test_x).flatten()
        m_test = Metrics(test_y, preds_test)
        result = {"N": n, "MAE": m_test.mae(), "Treatment": treatment, "MAE": m_test.mae(),
                       "Pearson": m_test.pearsonr().statistic, "Spearman": m_test.spearmanr().statistic}
        results.append(result)
        to_add = list(np.random.choice(pool, min((step,len(pool))), replace=False))
        inds = list(set(inds)|set(to_add))
        pool = list(set(pool)-set(to_add))
        n = len(inds)
    return results




def train(model, train_x, train_y, treatment = "None"):

    if treatment=="FairReweighing":
        db = DensityBalance()
        weight = db.weight(np.array(data[data["split_mark"] != "test"]["A"]), train_y)
    else:
        weight = None



    checkpoint_path = "checkpoint/STD.keras"
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=100, factor=0.3, min_lr=1e-6, verbose=1)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor="loss", save_best_only=True,
                                                    save_weights_only=True, verbose=1)
    # early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=150, verbose=1,
    #                                                   restore_best_weights=True)

    history = model.fit(
        train_x, train_y,
        sample_weight=weight,
        batch_size=10,
        epochs=200,
        callbacks=[reduce_lr, checkpoint],
        verbose=1
    )

    print("\nLoading best checkpoint model...")
    model.load_weights(checkpoint_path)
    return model

if __name__ == "__main__":
    data = "jirasoftware_filtered"
    # treatments = ["None"]
    seed = 10
    tf.random.set_seed(seed)
    np.random.seed(seed)
    results = active(data, init=10, step = 10, delta=0.5)
    results = pd.DataFrame(results)
    print(results)
    results.to_csv("../Results/STD_random_cont.csv", index=False)




