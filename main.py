import math
import random
import torch
import numpy as np
from tqdm import tqdm
from time import time
from prettytable import PrettyTable
from utils.parser import parse_args
from utils.data_loader import load_data
from module.H2GR import Recommender
from utils.evaluate import test
from utils.helper import early_stopping

from torch.utils.tensorboard import SummaryWriter
import os 


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
n_users = 0
n_items = 0
n_entities = 0
n_nodes = 0
n_relations = 0


def get_feed_data(train_entity_pairs, train_user_set):

    def negative_sampling(user_item, train_user_set):
        neg_items = list()
        for user, _ in user_item.cpu().numpy():
            user = int(user)
            each_negs = list()
            neg_item = np.random.randint(low=0, high=n_items, size=args.num_neg_sample) #negative pair number = 200
            if len(set(neg_item) & set(train_user_set[user]))==0: #🌟🌟 no intersection
                each_negs += list(neg_item)
            else: #🌟🌟 intersection: not all random items are not in user-item interact matrix
                neg_item = list(set(neg_item) - set(train_user_set[user]))
                each_negs += neg_item
                while len(each_negs)<args.num_neg_sample: #🌟🌟 complement neg-item until it is full
                    n1 = np.random.randint(low=0, high=n_items, size=1)[0]
                    if n1 not in train_user_set[user]:
                        each_negs += [n1]
            neg_items.append(each_negs)
        return neg_items

    feed_dict = {}
    entity_pairs = train_entity_pairs
    feed_dict['users'] = entity_pairs[:, 0] #[1380510]
    feed_dict['pos_items'] = entity_pairs[:, 1] #[1380510]
    feed_dict['neg_items'] = torch.LongTensor(negative_sampling(entity_pairs,train_user_set)) #[1380510, 200]
    # print(feed_dict['users'].shape) 
    # print(feed_dict['pos_items'].shape)
    # print(feed_dict['neg_items'].shape)
    return feed_dict

if __name__ == '__main__':
    """fix the random seed"""
    seed = 2020
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    """read args"""
    global args, device
    args = parse_args()
    device = torch.device("cuda:"+str(args.gpu_id)) if args.cuda else torch.device("cpu")

    """build dataset"""
    train_cf, test_cf, uie_dict, n_params, global_graph, mat_list = load_data(args)


    #🌟train_cf: 1380510(alibaba), 80% of Interactions
    #🌟test_cf: 400583(alibaba), 20% of Interactions

    # train_cf = train_cf[:10]
    # test_cf = test_cf[:10]

    adj_mat_list, mean_mat_list = mat_list

    n_users = n_params['n_users']
    n_items = n_params['n_items']
    n_entities = n_params['n_entities']
    n_relations = n_params['n_relations']
    n_nodes = n_params['n_nodes']

    """cf data"""
    train_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in train_cf], np.int32)) #🌟[1380510,2]
    test_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in test_cf], np.int32)) #🌟[400583,2]

    """define model"""
    model = Recommender(n_params, args, global_graph, mean_mat_list).to(device)

    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total/1e6))




    writer = SummaryWriter()

    """define optimizer"""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    cur_best = 0
    stopping_step = 0
    should_stop = False

    print("start training ...")
    iter = math.ceil(len(train_cf_pairs) / args.batch_size) 

    for epoch in range(args.epoch):
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
        if epoch%10 == 1 or epoch==0:
            # shuffle training data
            index = np.arange(len(train_cf))
            np.random.shuffle(index)
            train_cf_pairs = train_cf_pairs[index]
            print("start prepare feed data...")
            all_feed_data = get_feed_data(train_cf_pairs, uie_dict['train_user_set'])  # {'user': [n,], 'pos_item': [n,], 'neg_item': [n, n_sample]}

        """training"""
        model.train()
        loss, s, con_loss = 0, 0, 0
        train_s_t = time()
        for i in tqdm(range(iter)):
            batch = dict()
            batch['users'] = all_feed_data['users'][i*args.batch_size:(i+1)*args.batch_size].to(device) #🌟[batch_size]
            batch['pos_items'] = all_feed_data['pos_items'][i*args.batch_size:(i+1)*args.batch_size].to(device) #🌟[batch_size]
            batch['neg_items'] = all_feed_data['neg_items'][i*args.batch_size:(i+1)*args.batch_size,:].to(device) #🌟[batch_size, 200]
            # batch['pos_agg_items'] = all_feed_data['pos_agg_items'].to(device) #🌟n_users, 3]

            batch_loss, loss2 = model(batch)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            loss += batch_loss.item()
            con_loss += loss2.item()
            s += args.batch_size


        train_e_t = time()
        print("contrastive loss: ", con_loss)
        if epoch % 5 == 0 or epoch == 1:
            """testing"""
            model.eval()
            test_s_t = time()
            with torch.no_grad():
                ret = test(model, uie_dict, n_params)
            test_e_t = time()
            torch.save(model.state_dict(), args.out_dir + 'model_' + args.dataset + str(epoch) +'.pth')
            train_res = PrettyTable()
            train_res.field_names = ["Epoch", "training time", "tesing time", "Loss", "recall", "ndcg", "precision", "hit_ratio"]
            train_res.add_row(
                [epoch, train_e_t - train_s_t, test_e_t - test_s_t, loss, ret['recall'], ret['ndcg'], ret['precision'], ret['hit_ratio']]
            )
            print(train_res)
            writer.add_scalar('Testing/recall', ret['recall'][0], epoch)
            writer.add_scalar('Testing/ndcg', ret['ndcg'][0], epoch)
            writer.add_scalar('Testing/precision', ret['precision'][0], epoch)
            writer.add_scalar('Testing/hit_ratio', ret['hit_ratio'][0], epoch)
            writer.add_scalar('Training/Loss', loss, epoch)
            f = open('./result/{}.txt'.format(args.dataset), 'a+')
            f.write(str(train_res) + '\n')
            f.close()
            # *********************************************************
            # cur_best, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best,
            #                                                             stopping_step, expected_order='acc',
            #                                                             flag_step=20)
            # if should_stop:
            #     break

            # """save weight"""
            # if ret['recall'][0] == cur_best and args.save:
            #     torch.save(model.state_dict(), args.out_dir + 'model_' + args.dataset + str(epoch) +'.pth')

        else:
            writer.add_scalar('Training/Loss', loss, epoch)
            print('using time %.4f, training loss at epoch %d: %.4f' % (train_e_t - train_s_t, epoch, loss))

    print('early stopping at %d, recall@20:%.4f' % (epoch, cur_best))
