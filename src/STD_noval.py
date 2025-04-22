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

        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5)),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dense(1, activation=None)
    ])

    # initial_learning_rate = 0.001
    # lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    #     initial_learning_rate, decay_steps=5000, alpha=0.0001
    # )
    # optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    model.compile(
        optimizer='SGD',
        loss="mae",
        # loss=tf.keras.losses.Huber(delta=1.0),
        metrics=['mae']
    )

    return model

def train_and_test(dataname, treatment = "None"):

    data = process(dataname, "is_internal")
    train_x = np.array(data[data["split_mark"]!="test"]["X"].tolist())
    train_y = np.array(data[data["split_mark"]!="test"]["Y"].tolist())
    test_x = np.array(data[data["split_mark"] == "test"]["X"].tolist())
    test_y = np.array(data[data["split_mark"] == "test"]["Y"].tolist())

    if treatment=="FairReweighing":
        db = DensityBalance()
        weight = db.weight(np.array(data[data["split_mark"] != "test"]["A"]), train_y)
    else:
        weight = None


    model = build_model((train_x.shape[1]))

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
        validation_data=None,
        batch_size=32,
        epochs=500,
        callbacks=[reduce_lr, checkpoint],
        verbose=1
    )

    print("\nLoading best checkpoint model...")
    model.load_weights(checkpoint_path)
    # model.fit(train_x, train_y)
    preds_test = model.predict(test_x).flatten()
    preds_train = model.predict(train_x).flatten()
    m_train = Metrics(train_y, preds_train)
    m_test = Metrics(test_y, preds_test)
    result_train = {"Data": dataname, "Treatment": treatment, "MAE": m_train.mae(),
                    "Pearson": m_train.pearsonr().statistic, "Spearman": m_train.spearmanr().statistic,
                    "Isep": m_train.Isep(np.array(data[data["split_mark"] != "test"]["A"])),
                    "Csep": m_train.Csep(np.array(data[data["split_mark"] != "test"]["A"]))}
    result_test = {"Data": dataname, "Treatment": treatment, "MAE": m_test.mae(),
                   "Pearson": m_test.pearsonr().statistic, "Spearman": m_test.spearmanr().statistic,
                   "Isep": m_test.Isep(np.array(data[data["split_mark"] == "test"]["A"])),
                   "Csep": m_test.Csep(np.array(data[data["split_mark"] == "test"]["A"]))}
    # result_train = {"Data": dataname, "Treatment": treatment, "MAE": m_train.mae(),
    #                 "Pearson": "(%.2f) %.2f" % (m_train.pearsonr().pvalue, m_train.pearsonr().statistic),
    #                 "Spearman": "(%.2f) %.2f" % (m_train.spearmanr().pvalue, m_train.spearmanr().statistic),
    #                 "Isep": m_train.Isep(np.array(data[data["split_mark"] != "test"]["A"])),
    #                 "Csep": m_train.Csep(np.array(data[data["split_mark"] != "test"]["A"]))}
    # result_test = {"Data": dataname, "Treatment": treatment, "MAE": m_test.mae(),
    #                "Pearson": "(%.2f) %.2f" % (m_test.pearsonr().pvalue, m_test.pearsonr().statistic),
    #                "Spearman": "(%.2f) %.2f" % (m_test.spearmanr().pvalue, m_test.spearmanr().statistic),
    #                "Isep": m_test.Isep(np.array(data[data["split_mark"] == "test"]["A"])),
    #                "Csep": m_test.Csep(np.array(data[data["split_mark"] == "test"]["A"]))}
    return result_train, result_test

if __name__ == "__main__":
    data = "jirasoftware_filtered"
    treatments = ["None", "FairReweighing"]
    results_train = []
    results_test = []
    for treatment in treatments:
        for _ in range(20):
            result_train, result_test = train_and_test(data, treatment=treatment)
            results_train.append(result_train)
            results_test.append(result_test)
    results_train = pd.DataFrame(results_train)
    print(results_train)
    results_train.to_csv("../Results/STD_train.csv", index=False)
    results_test = pd.DataFrame(results_test)
    print(results_test)
    results_test.to_csv("../Results/STD_test.csv", index=False)




