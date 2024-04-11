import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import random


def set_seed(seed, cudnn=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if (seed is not None) and cudnn:
        torch.backends.cudnn.deterministic = True

def load_data(dataset, test_size, drop_rate, target_col, sensitive_col, seed, device):


	df = pd.read_csv(f"./datasets/{dataset}.csv")
	
	X = df.drop([target_col, sensitive_col], axis=1).values
	Y = df[target_col].values.reshape((-1, 1))
	S = df[sensitive_col].values.reshape((-1, 1))
	
	X_train, X_test, Y_train, Y_test, S_train, S_test = train_test_split(X, Y, S, test_size=test_size, random_state=seed)


	np.random.seed(seed)	
	dropped_idx = np.random.uniform(size=(len(S_train))) < drop_rate

	labeld_train_set = TensorDataset(
		torch.tensor(X_train[~dropped_idx], dtype=torch.float32).to(device),
		torch.tensor(Y_train[~dropped_idx], dtype=torch.float32).to(device),
		torch.tensor(S_train[~dropped_idx], dtype=torch.float32).to(device)
	)

	unlabeled_train_set = TensorDataset(
		torch.tensor(X_train[dropped_idx], dtype=torch.float32).to(device),
		torch.tensor(Y_train[dropped_idx], dtype=torch.float32).to(device)
	)

	unlabeled_train_set_with_label = TensorDataset(
		torch.tensor(X_train[dropped_idx], dtype=torch.float32).to(device),
		torch.tensor(Y_train[dropped_idx], dtype=torch.float32).to(device),
		torch.tensor(S_train[dropped_idx], dtype=torch.float32).to(device)
	)

	test_set = TensorDataset(
		torch.tensor(X_test, dtype=torch.float32).to(device),
		torch.tensor(Y_test, dtype=torch.float32).to(device),
		torch.tensor(S_test, dtype=torch.float32).to(device)
	)

	return labeld_train_set, unlabeled_train_set, unlabeled_train_set_with_label, test_set

def labeling_unlabeled_data(estimator, unlabeled_set):
	unlabeled_loader = DataLoader(unlabeled_set, batch_size=256, shuffle=True)

	features = []
	target_labels = []
	sensitive_labels = []

	estimator = estimator.eval()

	for X, Y in unlabeled_loader:
		S_pred = estimator(X)

		features.append(X)
		target_labels.append(Y)
		sensitive_labels.append((S_pred > 0.5).to(torch.float))
		
	X = torch.concat(features)
	Y = torch.concat(target_labels)
	S = torch.concat(sensitive_labels)

	return TensorDataset(
		X, Y, S
	)

def labeling_unlabeled_data_randomly(unlabeled_set):
	unlabeled_loader = DataLoader(unlabeled_set, batch_size=256, shuffle=True)

	features = []
	target_labels = []
	sensitive_labels = []


	for X, Y, in unlabeled_loader:
		S_pred = torch.rand_like(Y)

		features.append(X)
		target_labels.append(Y)
		sensitive_labels.append((S_pred > 0.5).to(torch.float))
		
	X = torch.concat(features)
	Y = torch.concat(target_labels)
	S = torch.concat(sensitive_labels)

	return TensorDataset(
		X, Y, S
	)