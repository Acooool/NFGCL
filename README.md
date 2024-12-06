# NFGCL
NFGCL: A Negative-sampling-free Graph Contrastive Learning Framework for Recommendation

Our model is in "recbole_gnn/model/general_recommender/nfgcl.py"

If you want to run the model, you can add our code files to the Recbole-GNN-main (an open-source library: **https://github.com/RUCAIBox/RecBole-GNN.git**) and then edit the file **run_recbole_gnn.py**. 

For example (dataset=Gowalla):

```
import argparse

from recbole_gnn.quick_start import run_recbole_gnn


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='NFGCL', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='Gowalla', help='name of datasets')
    parser.add_argument('--config_files', type=str, default='recbole_gnn/properties/overall.yaml recbole_gnn/properties/dataset/gowalla.yaml', help='config files')

    args, _ = parser.parse_known_args()

    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    run_recbole_gnn(model=args.model, dataset=args.dataset, config_file_list=config_file_list)
```

Thank you for your attention! :)
