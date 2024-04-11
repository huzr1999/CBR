for dataset in COMPAS Adult Crime; do
	for method in ADV DBC REW; do
		python main.py --dataset $dataset --method $method
	done
done