import pandas as pd
import torch
from torch.utils.data import ConcatDataset, random_split, TensorDataset, DataLoader
from .training import train_estimator
from .utils import labeling_unlabeled_data, load_data, labeling_unlabeled_data_randomly


class Runner:
	def __init__(self, dataset, test_size, drop_rate, seed, device):

		self.dataset = dataset
		if dataset == 'Adult':
			self.sensitive_col = 'gender'
			self.target_col = 'income'
		elif dataset == 'COMPAS':
			self.sensitive_col = 'race'
			self.target_col = 'two_year_recid'
		elif dataset == 'Crime':
			self.sensitive_col = 'S'
			self.target_col = 'ViolentCrimesPerPop'

		# self.sensitive_col = 'race'
		# self.target_col = 'two_year_recid'
		
		self.device = device

		self.labeled_train_set, self.unlabeled_train_set, self.unlabeled_train_set_with_label, self.test_set = load_data(dataset, test_size, drop_rate, self.target_col, self.sensitive_col, seed, device)

		self.col_names = pd.read_csv(f"./datasets/{dataset}.csv", nrows=1).columns
	
	def Ours(self, num_epochs, lr, metric='sp'):

		# Split a extra validation set to find tau
		train_len = int(0.7 * len(self.labeled_train_set))
		val_len = len(self.labeled_train_set) - train_len

		train_set, val_set = random_split(self.labeled_train_set, [train_len, val_len])

		estimator = train_estimator(train_set, self.test_set, num_epochs, lr, self.device)

		sorted_confidence, indices = torch.sort(estimator(val_set[:][0]).reshape(-1))
		sorted_s = val_set[:][2].reshape(-1)[indices]
		sorted_y = val_set[:][1].reshape(-1)[indices]

		cut_idx = torch.argmax((sorted_confidence >= 0.5).to(torch.int)).item() - 1

		min_idx_0 = 0
		min_val_0 = 1000000 
		for i in range(cut_idx + 1):
			if metric == 'sp':
				val = (sorted_s[0:i + 1] == 1).sum() + (sorted_s[i + 1:cut_idx + 1] == 0).sum()
			else:
				val = ((sorted_s[0:i + 1] == 1)
						& (sorted_y[0:i + 1] == 1)).sum() + \
					  ((sorted_s[i + 1:cut_idx + 1] == 0)
						& (sorted_y[i + 1:cut_idx + 1] == 1)).sum()

			if min_val_0 > val:
				min_val_0 = val
				min_idx_0 = i

		min_idx_1 = cut_idx + 1
		min_val_1 = 1000000 
		for i in range(cut_idx + 1, len(sorted_s)):
			if metric == 'sp':
				val = (sorted_s[cut_idx + 1:i + 1] == 1).sum() + (sorted_s[i + 1:] == 0).sum()
			else:
				val = ((sorted_s[cut_idx + 1:i + 1] == 1)
						& (sorted_y[cut_idx + 1:i + 1] == 1)).sum() + \
					  ((sorted_s[i + 1:] == 0)
						& (sorted_y[i + 1:] == 1)).sum()
				
			if min_val_1 >= val:
				min_val_1 = val
				min_idx_1 = i

		tau1 = min(sorted_confidence[min_idx_0], 0.5)
		tau2 = max(sorted_confidence[min_idx_1], 0.5)

		# Predict missing sensitive labels
		unlabeled_loader = DataLoader(self.unlabeled_train_set, batch_size=256, shuffle=True)

		features = []
		confidences = []
		target_labels = []
		sensitive_labels = []

		estimator = estimator.eval()

		for X, Y in unlabeled_loader:
			S_pred = estimator(X)

			features.append(X)
			target_labels.append(Y)
			confidences.append(S_pred)

			sensitive_labels.append((S_pred > 0.5).to(torch.float))
			
		X = torch.concat(features)
		Y = torch.concat(target_labels)
		S = torch.concat(sensitive_labels)
		S_confidence = torch.concat(confidences)

		# Randomize the unreliable predicted labels
		S[(S_confidence < 0.5) & (S_confidence > tau1)] = (torch.rand(S[(S_confidence < 0.5) & (S_confidence > tau1)].shape, device=self.device) < tau1).to(torch.float)
		S[(S_confidence > 0.5) & (S_confidence < tau2)] = (torch.rand(S[(S_confidence > 0.5) & (S_confidence < tau2)].shape, device=self.device) < tau2).to(torch.float)

		labeled_unlabel_data = TensorDataset(X, Y, S)
				
		return ConcatDataset([self.labeled_train_set, labeled_unlabel_data])
