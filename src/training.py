import torch
from tqdm import tqdm
from torch import nn
from torch.nn.init import kaiming_uniform_
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score

class Classifier(nn.Module):
	def __init__(self, in_features):
		super().__init__()
		self.layer1 = nn.Linear(in_features, 10)
		kaiming_uniform_(self.layer1.weight, nonlinearity='relu')
		self.activation1 = nn.ReLU()
		self.layer2 = nn.Linear(10, 1)
		kaiming_uniform_(self.layer2.weight, nonlinearity='relu')
		self.activation2 = nn.Sigmoid()
		
	def forward(self, x):
		x = self.layer1(x)
		x = self.activation1(x)
		x = self.layer2(x)
		return self.activation2(x) 
	
class Classifier_adv(nn.Module):
	def __init__(self, in_features):
		super().__init__()
		self.layer1 = nn.Linear(in_features, 10)
		kaiming_uniform_(self.layer1.weight, nonlinearity='relu')
		self.activation1 = nn.ReLU()
		self.layer2 = nn.Linear(10, 1)
		kaiming_uniform_(self.layer2.weight, nonlinearity='relu')
		self.activation2 = nn.Sigmoid()

	def forward(self, x):
		x = self.layer1(x)
		prev_output = self.activation1(x)
		x = self.layer2(prev_output)
		return self.activation2(x), prev_output

class Adversary(nn.Module):
	def __init__(self):

		super().__init__()
		self.layer1 = nn.Linear(10, 1)
		kaiming_uniform_(self.layer1.weight, nonlinearity='sigmoid')
		self.activation1 = nn.Sigmoid()
	
	def forward(self, prev_output):
		output = self.layer1(prev_output)
		return self.activation1(output)

def evaluate_model(classifier, test_loader, adv=False):
	with torch.no_grad():
		pred_list = []
		pred_score_list = []
		truth_list = []
		S_list = []
		
		for X, Y, S in test_loader:
			if adv:
				y_pred, _ = classifier(X)
			else:
				y_pred = classifier(X)

			pred_score_list.append(y_pred.cpu().numpy())
			pred_list.append((y_pred.cpu().numpy() > 0.5).astype(int))
			truth_list.append(Y.cpu().numpy().astype(int))
			S_list.append(S.cpu().numpy().astype(int))

		total_pred = np.concatenate(pred_list)
		total_y = np.concatenate(truth_list)
		total_S = np.concatenate(S_list)
		total_pred_score = np.concatenate(pred_score_list)

		acc = (total_pred == total_y).mean() * 100
		f1 = f1_score(total_y, total_pred) * 100
		sp = (total_pred[total_S == 1].mean() - total_pred[total_S == 0].mean()) * 100
		eo = (total_pred[(total_S == 1) & (total_y == 1)].mean() - total_pred[(total_S == 0) & (total_y == 1)].mean()) * 100
	return acc, f1, abs(sp), abs(eo)

def train_estimator(train_set, test_set, num_epochs, lr, device):

	num_input_features = len(train_set[0][0])

	estimator = Classifier(num_input_features).to(device)
	optimizer = torch.optim.Adam(estimator.parameters(), lr=lr)
	loss = nn.BCELoss()

	labeled_train_loader = DataLoader(train_set, batch_size=256, shuffle=True)
	test_loader = DataLoader(test_set, batch_size=256, shuffle=True)

	for epoch in tqdm(range(num_epochs), desc="training"):
		for X, _, S in labeled_train_loader:
			optimizer.zero_grad()
			S_pred = estimator(X)

			l = loss(S_pred, S)
			l.backward()
			optimizer.step()
		
	with torch.no_grad():
		pred_list = []
		truth_list = []
		
		for X, _, S in test_loader:
			S_pred = estimator(X)

			pred_list.append((S_pred.cpu().numpy() > 0.5).astype(int))
			truth_list.append(S.cpu().numpy().astype(int))

		acc = (np.concatenate(pred_list) == np.concatenate(truth_list)).mean()
		print(f"Accuracy: {acc: .5f}")
	

	return estimator

def original_train(total_data_set, test_set, device):

	num_input_features = len(total_data_set[0][0])
	lr=0.1

	classifier = Classifier(num_input_features).to(device)
	optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
	loss = nn.BCELoss()

	train_loader = DataLoader(total_data_set, batch_size=256, shuffle=True)
	test_loader = DataLoader(test_set, batch_size=256, shuffle=True)

	num_epochs = 50
	for epoch in tqdm(range(num_epochs), desc="Training original classifier"):
		for X, Y, _ in train_loader:

			optimizer.zero_grad()
			y_pred = classifier(X)
			l = loss(y_pred, Y)
			l.backward()
			optimizer.step()

	return evaluate_model(classifier, test_loader)

def fair_train_adversary(total_data, test_set, lambda_, device):

	input_feature_num = len(total_data[0][0])
	classifier = Classifier_adv(input_feature_num).to(device)
	adv = Adversary().to(device)
	loss = nn.BCELoss()

	clf_optimizer = torch.optim.Adam(
		[{"params": classifier.layer1.parameters(), "lr": 0.01},
		{"params": classifier.layer2.parameters(), "lr": 0.001},
		])

	adv_lr = 0.01
	adv_optimizer = torch.optim.Adam(adv.parameters(), lr=adv_lr)

	data_loader = DataLoader(total_data, batch_size=256, shuffle=True)
	test_loader = DataLoader(test_set, batch_size=256, shuffle=True)


	# Pre-train classifier and adversary
	pre_num_epochs = 10

	for epoch in tqdm(range(pre_num_epochs), desc="Pre-training classifier"):
		for X, Y, S in data_loader:

			clf_optimizer.zero_grad()
			y_pred, _ = classifier(X)
			l = loss(y_pred, Y)
			l.backward()
			clf_optimizer.step()

		
	for epoch in tqdm(range(pre_num_epochs), desc="Pre-training adversary"):
		for X, Y, S in data_loader:

			adv_optimizer.zero_grad()
			
			_, prev_output = classifier(X)
			S_pred = adv(prev_output)
			l = loss(S_pred, S)
			l.backward()
			adv_optimizer.step()

	with torch.no_grad():
		pred_list = []
		truth_list = []
		
		for X, Y, S in test_loader:
			y_pred, _ = classifier(X)

			pred_list.append((y_pred.cpu().numpy() > 0.5).astype(int))
			truth_list.append(Y.cpu().numpy().astype(int))

		acc = (np.concatenate(pred_list) == np.concatenate(truth_list)).mean()
		print(f"Y-Accuracy: {acc: .5f}")

	with torch.no_grad():
		pred_list = []
		truth_list = []
		
		for X, Y, S in test_loader:
			_, prev_output = classifier(X)
			S_pred = adv(prev_output)

			pred_list.append((S_pred.cpu().numpy() > 0.5).astype(int))
			truth_list.append(S.cpu().numpy().astype(int))

		acc = (np.concatenate(pred_list) == np.concatenate(truth_list)).mean()
		print(f"S-Accuracy: {acc: .5f}")

	num_epochs = 50

	for epoch in tqdm(range(num_epochs), desc="Training..."):

		# Train adversary for 1 epoch
		for X, Y, S in data_loader:

			adv_optimizer.zero_grad()
			
			_, prev_output = classifier(X)
			S_pred = adv(prev_output)
			l = loss(S_pred, S)
			l.backward()
			adv_optimizer.step()
		
		# Train classifier for 1 mini-batch
		# for X, Y, S in data_loader:
		X, Y, S = next(iter(data_loader))
		
		clf_optimizer.zero_grad()
		y_pred, prev_output = classifier(X)
		S_pred = adv(prev_output)

		l = loss(y_pred, Y)
		l_adv = loss(S_pred, S)

		(l - lambda_ * l_adv).backward()
		clf_optimizer.step()

	return evaluate_model(classifier, test_loader, adv=True)


def fair_train_DBC(total_data_set, test_set, lambda_, device):

	num_input_features = len(total_data_set[0][0])
	lr=0.01

	classifier = Classifier(num_input_features).to(device)
	optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
	loss = nn.BCELoss()

	train_loader = DataLoader(total_data_set, batch_size=256, shuffle=True)
	test_loader = DataLoader(test_set, batch_size=256, shuffle=True)

	num_epochs = 100

	for epoch in tqdm(range(num_epochs), desc="DBC: training"):
		for X, Y, S in train_loader:

			optimizer.zero_grad()
			y_pred = classifier(X)
			try:
				l1 = loss(y_pred, Y)
			except Exception:
				print(y_pred, Y, X)
				for param in classifier.parameters():
					print(param)


			p_1 = S.mean()

			# Transform S with values of 0/1 into that with values of -1/1
			S = 2 * S - 1
			if p_1 > 0.99 or p_1 < 0.01:
				l2 = 0
			else:
				l2 = (1 / (1e-4 + p_1 * (1 - p_1)) * ((S + 1) / 2 - p_1) * y_pred).mean().abs()


			(l1 + lambda_ * l2).backward()
			optimizer.step()


	return evaluate_model(classifier, test_loader)

def fair_train_reweight(total_data_set, test_set, ita, device):

	if isinstance(total_data_set, list):
		total_data_set = total_data_set[0]

	num_input_features = len(total_data_set[0][0])
	num_input_samples = len(total_data_set)
	lr=0.01

	classifier = Classifier(num_input_features).to(device)
	optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
	loss = nn.BCELoss(reduction='none')

	weights = torch.ones((num_input_samples, 1)).to(device)
	train_loader = DataLoader(total_data_set, batch_size=256, shuffle=True)
	test_loader = DataLoader(test_set, batch_size=256, shuffle=True)

	total_X, total_Y, total_S, total_logits = [], [], [], []
	with torch.no_grad():
		for X, Y, S in train_loader:
			total_X.append(X)
			total_Y.append(Y)
			total_S.append(S)

	total_X = torch.concat(total_X)
	total_Y = torch.concat(total_Y)
	total_S = torch.concat(total_S)
		
	total_data_set = TensorDataset(total_X, total_Y, total_S, weights)
	train_loader = DataLoader(total_data_set, batch_size=256, shuffle=True)
		

	num_epochs = 5
	# lambda_ = 0.5

	lambda_sp_1 = 0
	lambda_sp_0 = 0
	lambda_eo_1 = 0
	lambda_eo_0 = 0

	T = 100

	for epoch in range(50):
		for X, Y, S, W in train_loader:

			optimizer.zero_grad()
			y_pred = classifier(X)
			l1 = (loss(y_pred, Y) * W).mean()


			l1.backward()
			optimizer.step()

	print(evaluate_model(classifier, test_loader), ita)

	for t in tqdm(range(T), desc="Reweight: Training"):
		for epoch in range(num_epochs):
			for X, Y, S, W in train_loader:

				optimizer.zero_grad()
				y_pred = classifier(X)
				l1 = (loss(y_pred, Y) * W).mean()


				l1.backward()
				optimizer.step()

		# if t % (10 - 1) == 0:
		# 	print(evaluate_model(classifier, test_loader), ita)

		total_X, total_Y, total_S, total_logits = [], [], [], []
		with torch.no_grad():
			for X, Y, S, _ in train_loader:
				total_X.append(X)
				total_Y.append(Y)
				total_S.append(S)
				total_logits.append(classifier(X))
				
		total_X = torch.concat(total_X)
		total_Y = torch.concat(total_Y)
		total_S = torch.concat(total_S)
		total_logits = torch.concat(total_logits)
		total_pred = torch.where(total_logits > 1, torch.ones_like(total_logits), torch.zeros_like(total_logits))

		delta_sp_1 = (total_logits * (total_S / total_S.mean() - 1)).mean()
		delta_sp_0 = (total_logits * ((1 - total_S) / (1 - total_S).mean() - 1)).mean()

		delta_eo_1 = total_logits[(total_S == 1) & (total_Y == 1)].mean() - total_logits[total_Y == 1].mean()
		delta_eo_0 = total_logits[(total_S == 0) & (total_Y == 1)].mean() - total_logits[total_Y == 1].mean()


		lambda_sp_1 -= ita * delta_sp_1
		lambda_sp_0 -= ita * delta_sp_0

		lambda_eo_1 -= ita * delta_eo_1
		lambda_eo_0 -= ita * delta_eo_0

		weights_tilde = torch.exp(lambda_sp_1 * total_S + lambda_sp_0 * (1 - total_S) + lambda_eo_1 * total_S + lambda_eo_0 * (1 - total_S))
		weights = (weights_tilde * total_Y + 1 * (1 - total_Y))/(1 + weights_tilde)

		total_data_set = TensorDataset(total_X, total_Y, total_S, weights)
		train_loader = DataLoader(total_data_set, batch_size=256, shuffle=True)

	return evaluate_model(classifier, test_loader)