import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--dataset", default='COMPAS')
parser.add_argument("--test_size", default=0.2)
parser.add_argument("--sensitive_drop_rate", default=0.8, type=float)
parser.add_argument("--num_epochs", default=100)
parser.add_argument("--lr", default=0.01)
parser.add_argument("--device", default='cpu')
parser.add_argument("--method", default='ADV')


args = parser.parse_args()