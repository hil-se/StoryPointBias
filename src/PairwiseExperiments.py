import pandas as pd
from transformers import GPT2Tokenizer, GPT2PreTrainedModel, GPT2Config
# import re
from GPT2SPModel import GPT2SPModel
import torch
import numpy as np
import scipy
import DatasetCreator as dc
import math
# import gc
# from sklearn import preprocessing
from ScalerModel import ScalerModel
from FTSVMModel import FTSVMModel
import time


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)

def loadData(dataset="clover", num_to_add=1, modeltype="GPT2SP"):
    print("Loading data...")
    if modeltype=="GPT2SP":
        return dc.generateData(dataName=dataset, labelName="Storypoint", LM="GPT2", num_to_add=num_to_add, labels=2)
    elif modeltype=="FTSVM":
        return dc.generateData(dataName=dataset, labelName="Storypoint", LM="FastText", num_to_add=num_to_add, labels=2)

def custom_loss_list(predictions, labels):
    predictions = predictions.tolist()
    labels = labels.tolist()
    ln = len(predictions)
    loss = 0
    for i in range(ln):
        if labels[i]==0:
            loss+=(abs(labels[i]-predictions[i]))
        else:
            loss += max(0, 1 - (predictions[i]*labels[i]))
    loss/=ln
    return loss

def custom_loss_tensor(predictions, labels):
    mae_loss = torch.abs(predictions - labels)
    mae_mask = (labels == 0).float()

    hinge_loss = torch.clamp(1 - predictions * labels, min=0)
    hinge_mask = (labels != 0).float()

    total_loss = mae_mask * mae_loss + hinge_mask * hinge_loss

    return torch.mean(total_loss)


def trainModel(dataname,
               train,
               val,
               modeltype="GPT2SP",
               loss="hinge",
               batch_size=16,
               epochs=None):

    if modeltype=="GPT2SP":
        config = GPT2Config(num_labels=1, pad_token_id=50256)
        model = GPT2SPModel(config)
        model = GPT2SPModel.from_pretrained('gpt2', config=config)
    elif modeltype=="FTSVM":
        model = FTSVMModel(input_size=300)

    # model.load_state_dict(torch.load("../../Data/GPT2SP Data/Trained models/"+dataname+".pkl", weights_only=True), strict=False)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.00001)
    model = model.to(DEVICE)

    train_A = train["A"].tolist()
    train_B = train["B"].tolist()
    train_label = train["Label"].tolist()
    train_len = len(train.index)

    val_A = val["A"].tolist()
    val_B = val["B"].tolist()
    val_label = val["Label"].tolist()
    val_len = len(val.index)

    model_path = "../Data/GPT2SP Data/Models/"+dataname+"_"+modeltype+".pth"

    if loss=="hinge":
        loss_fn = torch.nn.MarginRankingLoss()
    elif loss=="mae":
        loss_fn = torch.nn.L1Loss()
    elif loss=="hinge_embedding":
        loss_fn = torch.nn.HingeEmbeddingLoss()

    batch_size = 16
    epochs_per_round = 1
    batch_start = 0
    ind_tr = 0
    ind_v = 0
    ind_ts = 0
    batch_end = (batch_size*ind_tr)+batch_size

    if epochs==None:
        epochs = epochs_per_round*int(train_len/batch_size)

    best_loss = 100
    best_loss_v = 100
    best_loss_avg = 100
    best_epoch = 0
    epochs_since_decrease = 0
    # early_stopping_epochs = batch_size*5
    early_stopping_epochs = 15

    print("Max epochs:", epochs)
    print("Training...\n")

    for epoch in range(epochs):
        # Batching
        batch_start = ind_tr*batch_size
        batch_end = batch_start + batch_size
        if batch_end>=train_len:
            batch_end = train_len
            ind_tr = 0
        train_A_batch = train_A[batch_start:batch_end]
        train_B_batch = train_B[batch_start:batch_end]
        train_label_batch = train_label[batch_start:batch_end]
        ind_tr+=1

        batch_start = ind_v * batch_size
        batch_end = batch_start + batch_size
        if batch_end >= val_len:
            batch_end = val_len
            ind_v = 0
        val_A_batch = val_A[batch_start:batch_end]
        val_B_batch = val_B[batch_start:batch_end]
        val_label_batch = val_label[batch_start:batch_end]
        ind_v += 1


        # Formatting
        train_A_batch = torch.Tensor(train_A_batch)
        train_B_batch = torch.Tensor(train_B_batch)
        if modeltype == "GPT2SP":
            train_A_batch = train_A_batch.to(torch.int)
            train_B_batch = train_B_batch.to(torch.int)
        train_label_batch = torch.Tensor(train_label_batch)
        train_label_batch = train_label_batch.to(DEVICE)

        val_A_batch = torch.Tensor(val_A_batch)
        val_B_batch = torch.Tensor(val_B_batch)
        if modeltype == "GPT2SP":
            val_A_batch = val_A_batch.to(torch.int)
            val_B_batch = val_B_batch.to(torch.int)
        val_label_batch = torch.Tensor(val_label_batch)
        val_label_batch = val_label_batch.to(DEVICE)

        train_A_batch = train_A_batch.to(DEVICE)
        train_B_batch = train_B_batch.to(DEVICE)
        val_A_batch = val_A_batch.to(DEVICE)
        val_B_batch = val_B_batch.to(DEVICE)

        # Training
        train_A_pred = model(train_A_batch)
        train_B_pred = model(train_B_batch)
        train_A_pred = train_A_pred.to(DEVICE)
        train_B_pred = train_B_pred.to(DEVICE)

        val_A_pred = model(val_A_batch)
        val_B_pred = model(val_B_batch)
        val_A_pred = val_A_pred.to(DEVICE)
        val_B_pred = val_B_pred.to(DEVICE)

        if loss=="hinge_embedding":
            train_pred = torch.sub(train_A_pred, train_B_pred, alpha=1)
            val_pred = torch.sub(val_A_pred, val_B_pred, alpha=1)
            train_pred = train_pred.to(DEVICE)
            val_pred = val_pred.to(DEVICE)

        # Loss
        # loss_tr = custom_loss_tensor(train_pred, train_label_batch)
        # loss_v = custom_loss_tensor(val_pred, val_label_batch)

        if train_A_pred.size()==torch.Size([]):
            train_A_pred = torch.unsqueeze(train_A_pred, 0)
        if train_B_pred.size()==torch.Size([]):
            train_B_pred = torch.unsqueeze(train_B_pred, 0)

        if val_A_pred.size()==torch.Size([]):
            val_A_pred = torch.unsqueeze(val_A_pred, 0)
        if val_B_pred.size()==torch.Size([]):
            val_B_pred = torch.unsqueeze(val_B_pred, 0)

        if loss=="hinge_embedding" or loss=="mae":
            loss_tr = loss_fn(train_pred, train_label_batch)
            loss_v = loss_fn(val_pred, val_label_batch)
        elif loss=="hinge":
            loss_tr = loss_fn(train_A_pred, train_B_pred, train_label_batch)
            loss_v = loss_fn(val_A_pred, val_B_pred, val_label_batch)

        avg_loss = (abs(loss_tr.item())+abs(loss_v.item()))/2


        print("Epoch:", epoch + 1, ", Training Loss:", loss_tr.item(), ", Val Loss:", loss_v.item(), " Avg. Loss:", avg_loss)

        if avg_loss < best_loss_avg:
            best_loss = loss_tr.item()
            best_loss_v = loss_v.item()
            best_loss_avg = avg_loss
            best_epoch = epoch+1
            epochs_since_decrease = 0
            torch.save(model.state_dict(), model_path)
        else:
            epochs_since_decrease+=1
            if epochs_since_decrease>=early_stopping_epochs:
                print("\nEarly stopping limit reached.")
                print("Best loss:", best_loss, ", Best validation loss:", best_loss_v, "Best avg loss:", best_loss_avg)
                print("Loading best weights from epoch:", best_epoch)
                model.load_state_dict(torch.load(model_path, weights_only=True))
                model.eval()
                return model
        loss_tr.backward()
        optimizer.step()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model

def trainScalerModel(dataname, train_list, val_list, pairwise_model, pairwise_model_type):
    model = ScalerModel()
    model.to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.00001)
    model_path = "../Data/GPT2SP Data/Models/" + dataname + "_Scaler.pth"
    loss_fn = torch.nn.L1Loss()

    train = torch.Tensor(train_list["A"].tolist())
    val = torch.Tensor(val_list["A"].tolist())
    if pairwise_model_type=="GPT2SP":
        train = train.to(torch.int)
        val = val.to(torch.int)
    train_y = train_list["Score"].tolist()
    val_y = val_list["Score"].tolist()
    train_len = len(train_y)
    val_len = len(val_y)

    batch_size = 16
    epochs_per_round = 1
    batch_start = 0
    ind_tr = 0
    ind_v = 0
    ind_ts = 0
    batch_end = (batch_size * ind_tr) + batch_size

    epochs = max(epochs_per_round * int(train_len / batch_size), 100)
    best_loss = 100
    best_loss_v = 100
    best_loss_avg = 100
    best_epoch = 0
    epochs_since_decrease = 0
    early_stopping_epochs = 15

    print("\n\nTraining on model outputs...")
    print("Max epochs:", epochs)

    for epoch in range(epochs):
        # Batching
        batch_start = ind_tr * batch_size
        batch_end = batch_start + batch_size
        if batch_end >= train_len:
            batch_end = train_len
            ind_tr = 0
        train_A_batch = train[batch_start:batch_end]
        train_label_batch = train_y[batch_start:batch_end]
        ind_tr += 1

        batch_start = ind_v * batch_size
        batch_end = batch_start + batch_size
        if batch_end >= val_len:
            batch_end = val_len
            ind_v = 0
        val_A_batch = val[batch_start:batch_end]
        val_label_batch = val_y[batch_start:batch_end]
        ind_v += 1

        if len(train_A_batch) == 0 or len(val_A_batch) == 0:
            continue

        train_A_batch = torch.Tensor(train_A_batch)
        train_label_batch = torch.Tensor(train_label_batch)
        train_label_batch = train_label_batch.to(DEVICE)
        val_A_batch = torch.Tensor(val_A_batch)
        val_label_batch = torch.Tensor(val_label_batch)
        val_label_batch = val_label_batch.to(DEVICE)

        train_A_batch = train_A_batch.to(DEVICE)
        val_A_batch = val_A_batch.to(DEVICE)

        # Outputs
        train_A_pred = pairwise_model(train_A_batch).detach()
        train_A_pred = train_A_pred.to(DEVICE).detach()

        val_A_pred = pairwise_model(val_A_batch)
        val_A_pred = val_A_pred.to(DEVICE)

        train_A_pred = model(train_A_pred).to(DEVICE)
        val_A_pred = model(val_A_pred).to(DEVICE)

        if train_A_pred.size() == torch.Size([]):
            train_A_pred = torch.unsqueeze(train_A_pred, 0)
        loss_tr = loss_fn(train_A_pred, train_label_batch)
        if val_A_pred.size() == torch.Size([]):
            val_A_pred = torch.unsqueeze(val_A_pred, 0)
        loss_v = loss_fn(val_A_pred, val_label_batch)
        avg_loss = (abs(loss_tr.item()) + abs(loss_v.item())) / 2

        print("Epoch:", epoch + 1, ", Training Loss:", loss_tr.item(), ", Val Loss:", loss_v.item(), " Avg. Loss:", avg_loss)
        if avg_loss < best_loss_avg:
            best_loss = loss_tr.item()
            best_loss_v = loss_v.item()
            best_loss_avg = avg_loss
            best_epoch = epoch + 1
            epochs_since_decrease = 0
            torch.save(model.state_dict(), model_path)
        else:
            epochs_since_decrease += 1
            if epochs_since_decrease >= early_stopping_epochs:
                print("\nEarly stopping limit reached.")
                print("Best loss:", best_loss, ", Best validation loss:", best_loss_v, "Best avg loss:", best_loss_avg)
                print("Loading best weights from epoch:", best_epoch)
                model.load_state_dict(torch.load(model_path, weights_only=True))
                model.eval()
                return model
        loss_tr.backward()
        optimizer.step()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model


def testModel(model,
              test_list,
              max_sp,
              min_sp,
              scalerModel,
              modeltype="GPT2SP",
              batch_size=16):

    test = torch.Tensor(test_list["A"].tolist())
    if modeltype=="GPT2SP":
        test = test.to(torch.int)
    test_sp = test_list["Score"].tolist()

    batch_start = 0
    ind_ts = 0
    batch_end = (batch_size * ind_ts) + batch_size

    all_test_pred = []
    test_len = len(test_sp)
    itrs = math.ceil(test_len/batch_size)

    for i in range(itrs):
        if batch_end==test_len:
            break
        batch_start = ind_ts * batch_size
        batch_end = batch_start + batch_size
        if batch_end >= test_len:
            batch_end = test_len
            ind_ts = 0
        test_batch = test[batch_start:batch_end]
        ind_ts += 1

        # Formatting
        test_batch = torch.Tensor(test_batch)
        if modeltype=='GPT2SP':
            test_batch = test_batch.to(torch.int)
        test_batch = test_batch.to(DEVICE)

        test_pred = model(test_batch).to(DEVICE)

        if scalerModel:
            test_pred = scalerModel(test_pred)

        test_pred = test_pred.to(torch.device("cpu"))

        test_pred = test_pred.tolist()

        if type(test_pred) is list:
            all_test_pred.extend(test_pred)
        else:
            all_test_pred.append(test_pred)

    
    MAEs = []
    for i in range(len(test_sp)):
        MAEs.append(abs(test_sp[i]-all_test_pred[i]))
    MAE = sum(MAEs)/len(MAEs)
    MdAE = np.median(MAEs)

    pearsons = scipy.stats.pearsonr(all_test_pred, test_sp)[0]

    all_test_pred.sort()
    test_sp.sort()
    all_test_pred[all_test_pred==np.inf]=0

    spearmans = scipy.stats.spearmanr(test_sp, all_test_pred).statistic

    return MAE, MdAE, pearsons, spearmans



def experiment_project(dataset,
                       num_to_add=1,
                       modeltype="GPT2SP",
                       scaling=False,
                       training_epochs=200,
                       batch_size=16,
                       loss="hinge"):

    print(dataset)

    train, val, test, testlist, max_sp, min_sp, train_list, val_list = loadData(dataset,
                                                                                num_to_add=num_to_add,
                                                                                modeltype=modeltype)
    model = trainModel(dataset,
                       train,
                       val,
                       modeltype=modeltype,
                       loss=loss,
                       batch_size=batch_size,
                       epochs=training_epochs)

    scaler_model = None
    if scaling:
        scaler_model = trainScalerModel(dataset,
                                        train_list,
                                        val_list,
                                        model,
                                        modeltype)

    print("\n\nTesting...")

    MAE, MdAE, pearsons, spearmans = testModel(model,
                                               train_list,
                                               # testlist,
                                               max_sp,
                                               min_sp,
                                               scalerModel=scaler_model,
                                               modeltype=modeltype,
                                               batch_size=batch_size)

    print(dataset, MAE, MdAE, pearsons, spearmans)
    print("\n\n")

    return {"Data": dataset, "Pearson's coefficient": pearsons, "Spearman's coefficient": spearmans, "MAE": MAE, "MdAE": MdAE}

def experiments():
    datas = ["appceleratorstudio", "aptanastudio", "bamboo", "clover", "datamanagement", "duracloud", "jirasoftware",
             "mesos", "moodle", "mule", "mulestudio", "springxd", "talenddataquality", "talendesb", "titanium",
             "usergrid"]

    num_to_add = 1
    # modeltype = "GPT2SP"
    modeltype="FTSVM"
    scaling = False
    training_epochs = None
    batch_size = 16
    loss = "hinge"
    iterations = 1

    print("\nModel:", modeltype)
    print("Scaling:", scaling)
    print("Batch size:", batch_size)
    print("Loss function:", loss)
    print("Iterations:", iterations, "\n")

    filename = "../Results/Issue Story Points ("+modeltype+")"
    if scaling:
        filename+="(Scaling)"

    # experiment_project("clover",
    #                   num_to_add=num_to_add,
    #                   modeltype=modeltype,
    #                   scaling=scaling,
    #                   training_epochs=training_epochs,
    #                   batch_size=batch_size,
    #                   loss=loss)

    results = []
    for d in datas:
        p = 0
        sp = 0
        mae = 0
        mdae = 0
        times = 0
        for itr in range(iterations):
            print("Iteration: "+str(itr+1))
            start = time.time()
            temp_results = experiment_project(d,
                                              num_to_add=num_to_add,
                                              modeltype=modeltype,
                                              scaling=scaling,
                                              training_epochs=training_epochs,
                                              batch_size=batch_size,
                                              loss=loss)
            end = time.time()
            p+=temp_results["Pearson's coefficient"]
            sp+=temp_results["Spearman's coefficient"]
            mae+=temp_results["MAE"]
            mdae+=temp_results["MdAE"]
            times+=(end-start)
        data_results = {"Data": d,
                        "Pearson's coefficient": p/iterations,
                        "Spearman's coefficient": sp/iterations,
                        "MAE": mae/iterations,
                        "MdAE": mdae/iterations,
                        "Time": times/iterations}
        print(data_results)
        results.append(data_results)
    print("\n\n")
    results = pd.DataFrame(results)
    results.to_csv(filename+"_"+str(num_to_add)+".csv", index=False)
    print(results)
    print("Average results:")
    print("Pearson's coefficient:",
          sum(results["Pearson's coefficient"].tolist()) / len(results["Pearson's coefficient"]))
    print("Spearman's coefficient:",
          sum(results["Spearman's coefficient"].tolist()) / len(results["Spearman's coefficient"]))
    print("MAE:", sum(results["MAE"].tolist()) / len(results["MAE"]))
    print("MdAE:", sum(results["MdAE"].tolist()) / len(results["MdAE"]))
    print("Time:", sum(results["Time"].tolist()))

experiments()

# def dataStats():
#     datas = ["appceleratorstudio", "aptanastudio", "bamboo", "clover", "datamanagement", "duracloud", "jirasoftware",
#              "mesos", "moodle", "mule", "mulestudio", "springxd", "talenddataquality", "talendesb", "titanium",
#              "usergrid"]
#     res = []
#     for d in datas:
#         res.append(dc.getProjectStatisticsSummary(d))
#     res = pd.DataFrame(res)
#     res.to_csv("../Results/Data_stats.csv", index=False)

# dataStats()




