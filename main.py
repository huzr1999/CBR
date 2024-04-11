from src.Runner import Runner
from src.training import fair_train_adversary, fair_train_DBC, fair_train_reweight
from args import args
from src.utils import set_seed


seed = 23213


lambda_dict = {"COMPAS": {"ADV": 1.5, "DBC": 2, "REW": 0.3}, 
               "Adult": {"ADV": 2.5, "DBC": 0.25, "REW": 0.3},
               "Crime": {"ADV": 0.5, "DBC": 0.15, "REW": 0.1}}

set_seed(seed)
runner = Runner(args.dataset, args.test_size, args.sensitive_drop_rate, seed, args.device)

total_data_set = runner.Ours(args.num_epochs, args.lr)
test_set = runner.test_set

lambda_ = lambda_dict[args.dataset][args.method]

if args.method == 'ADV':
    acc, f1, sp, eo = fair_train_adversary(total_data_set, test_set, lambda_, args.device)
elif args.method == 'DBC':
    acc, f1, sp, eo = fair_train_DBC(total_data_set, test_set, lambda_, args.device)
else:
    acc, f1, sp, eo = fair_train_reweight(total_data_set, test_set, lambda_, args.device)

print(f"acc: {acc:.2f}")
print(f"f1: {f1:.2f}")
print(f"sp: {sp:.2f}")
print(f"eo: {eo:.2f}")