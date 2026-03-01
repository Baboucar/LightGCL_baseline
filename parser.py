import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Model Params')

    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--decay', default=0.99, type=float, help='learning rate decay (optional)')
    parser.add_argument('--batch', default=256, type=int, help='user batch size (eval)')
    parser.add_argument('--inter_batch', default=4096, type=int, help='interaction batch size (train)')

    parser.add_argument('--note', default=None, type=str, help='note')

    parser.add_argument('--lambda1', default=0.2, type=float, help='weight of cl loss')
    parser.add_argument('--lambda2', default=1e-7, type=float, help='l2 reg weight')

    parser.add_argument('--epoch', default=100, type=int, help='number of epochs')
    parser.add_argument('--d', default=64, type=int, help='embedding size')
    parser.add_argument('--q', default=5, type=int, help='rank')
    parser.add_argument('--gnn_layer', default=2, type=int, help='number of gnn layers')
    parser.add_argument('--data', default='yelp', type=str, help='name of dataset')

    parser.add_argument('--dropout', default=0.0, type=float, help='rate for edge dropout')
    parser.add_argument('--temp', default=0.2, type=float, help='temperature in cl loss')
    parser.add_argument('--cuda', default='0', type=str, help='the gpu to use')

    # NEW: evaluation mode
    parser.add_argument(
        '--eval_mode',
        default='full',
        type=str,
        choices=['full', 'neg99'],
        help="full = rank against all items; neg99 = 1pos+99neg (ml1m.test.negative)"
    )
    parser.add_argument('--eval_k', default=10, type=int, help='K for HR/NDCG in neg99 mode')

    return parser.parse_args()

args = parse_args()