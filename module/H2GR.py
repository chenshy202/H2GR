import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean
from scipy.sparse import csr_matrix
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pickle
from .hyperbolic import *
from .hyperboloid import Hyperboloid
import time
import math


class Aggregator(nn.Module):
    """
    Relational Path-aware Convolution Network
    """
    def __init__(self, n_users, n_items):
        super(Aggregator, self).__init__()
        self.manifold = Hyperboloid()
        self.n_users = n_users
        self.n_items = n_items

    def Lon_KG_forward(self, entity_look_up_emb, edge_index, edge_type,
                   relation_lookup_emb):
        c = 1
        head, tail = edge_index
        # exclude interact in IDs, remap [1, n_relations) to [0, n_relations-1)
        relation_type = edge_type - 1
        # [n_entities_, emb_dim]
        n_entities = entity_look_up_emb.shape[0]

        head_emb = entity_look_up_emb[head]
        tail_emb = entity_look_up_emb[tail]
        relation_emb = relation_lookup_emb[relation_type]

        # Mobius addition
        # map to hyperbolic point
        hyper_head_emb = self.manifold.expmap0(head_emb, c)

        # map to local hyperbolic point
        hyper_tail_emb = self.manifold.expmap(tail_emb, hyper_head_emb, c)
        hyper_relation_emb = self.manifold.expmap(relation_emb, hyper_head_emb, c)

        # Mobius addition in hyperbolic space
        res = project(self.manifold.add(hyper_tail_emb, hyper_relation_emb, c))
        # map to local tangent point
        res = self.manifold.logmap(res, hyper_head_emb, c)

        entity_agg = scatter_mean(src=res, index=head, dim_size=n_entities, dim=0)

        return entity_agg
    
    def Pon_KG_forward(self, entity_look_up_emb, edge_index, edge_type,
                   relation_lookup_emb):
        head, tail = edge_index
        # exclude interact in IDs, remap [1, n_relations) to [0, n_relations-1)
        relation_type = edge_type - 1
        # [n_entities_, emb_dim]
        n_entities = entity_look_up_emb.shape[0]

        head_emb = entity_look_up_emb[head]
        tail_emb = entity_look_up_emb[tail]
        relation_emb = relation_lookup_emb[relation_type]

        # Mobius addition
        # map to hyperbolic point
        hyper_head_emb = expmap0(head_emb)

        # map to local hyperbolic point
        hyper_tail_emb = expmap(tail_emb, hyper_head_emb)
        hyper_relation_emb = expmap(relation_emb, hyper_head_emb)

        # Mobius addition in hyperbolic space
        res = project(mobius_add(hyper_tail_emb, hyper_relation_emb))
        # map to local tangent point
        res = logmap(res, hyper_head_emb)

        entity_agg = scatter_mean(src=res, index=head, dim_size=n_entities, dim=0)

        return entity_agg

    def forward(self, entity_emb, user_emb, item_emb_cf,
                edge_index, edge_type, interact_mat,
                relation_weight, flag):
        
        """KG aggregate"""
        if flag == "Pon":
            entity_agg = self.Pon_KG_forward(entity_emb, edge_index, edge_type, relation_weight)
        elif flag == "Lon":
            entity_agg = self.Lon_KG_forward(entity_emb, edge_index, edge_type, relation_weight)


        """user aggregate"""
        mat_row = interact_mat._indices()[0, :]
        mat_col = interact_mat._indices()[1, :]
        mat_val = interact_mat._values()
        
        item_user_mat = torch.sparse.FloatTensor(torch.cat([mat_col, mat_row]).view(2, -1), torch.ones_like(mat_val),
                                                 size=[self.n_items, self.n_users])
        
        item_agg_cf = torch.sparse.mm(item_user_mat, user_emb)

        user_item_mat = torch.sparse.FloatTensor(torch.cat([mat_row, mat_col]).view(2, -1), torch.ones_like(mat_val),
                                                 size=[self.n_users, self.n_items])
        item_emb_kg = entity_emb[:self.n_items]

        item_emb_fusion = item_emb_cf + item_emb_kg
        user_agg = torch.sparse.mm(user_item_mat, item_emb_fusion)
        
        return entity_agg, user_agg, item_agg_cf
    

    


class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    """
    def __init__(self, channel, n_users,
                 n_items, n_relations, interact_mat,
                 device, node_dropout_rate=0.5, mess_dropout_rate=0.1):
        super(GraphConv, self).__init__()
        self.interact_mat = interact_mat
        self.n_relations = n_relations
        self.n_users = n_users
        self.n_items = n_items
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate
        self.device = device
        self.convs = Aggregator(n_users=n_users, n_items=n_items).to(self.device)

        relation_weight = nn.init.xavier_uniform_(torch.empty(n_relations - 1, channel))  # not include interact
        self.relation_weight = nn.Parameter(relation_weight)  # [n_relations - 1, in_channel]
        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout

    def _edge_sampling(self, edge_index, edge_type, rate=0.5):
        # edge_index: [2, -1]
        # edge_type: [-1]
        n_edges = edge_index.shape[1]
        random_indices = np.random.choice(n_edges, size=int(n_edges * rate), replace=False)
        return edge_index[:, random_indices], edge_type[random_indices]

    def _sparse_dropout(self, x, rate=0.5):
        noise_shape = x._nnz()

        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def Dr_Norm(self, user_emb, entity_emb, item_emb_cf, mess_dropout, dropout_rate):
        for i in range(3):
            if mess_dropout:
                Dropout = nn.Dropout(dropout_rate)
                entity_emb = Dropout(entity_emb)
                user_emb = Dropout(user_emb)
                item_emb_cf = Dropout(item_emb_cf)
            entity_emb = F.normalize(entity_emb)
            user_emb = F.normalize(user_emb)
            item_emb_cf = F.normalize(item_emb_cf)
            if i == 0:
                entity_agg, user_agg, item_agg_cf = entity_emb, user_emb, item_emb_cf
            else:
                entity_agg = torch.add(entity_agg, entity_emb)
                user_agg = torch.add(user_agg, user_emb)
                item_agg_cf = torch.add(item_agg_cf, item_emb_cf) 
        return entity_agg, user_agg, item_agg_cf


    def forward(self, user_emb, entity_emb, item_emb_cf, edge_index, edge_type,
                interact_mat, mess_dropout=True, node_dropout=False):

        """node dropout"""
        if node_dropout:
            edge_index, edge_type = self._edge_sampling(edge_index, edge_type, 0.5)
            interact_mat = self._sparse_dropout(interact_mat, 0.5)

        entity_res_emb = entity_emb  # [n_entity, channel]
        user_res_emb = user_emb  # [n_users, channel]
        item_emb_cf_res = item_emb_cf

        flag = "Pon"
        entity_emb, user_emb, item_emb_cf = self.convs(entity_emb, user_emb, item_emb_cf,
                                            edge_index, edge_type, interact_mat,
                                            self.relation_weight, flag)
        entity_agg, user_agg, item_agg_cf = self.Dr_Norm(user_emb, entity_emb, item_emb_cf, mess_dropout, 0.03)
            

        entity_res_emb = torch.add(entity_res_emb, entity_agg)
        user_res_emb = torch.add(user_res_emb, user_agg)
        item_emb_cf_res = torch.add(item_emb_cf_res, item_agg_cf)

  
        flag = "Pon"
        entity_emb, user_emb, item_emb_cf = self.convs(entity_agg, user_agg, item_agg_cf,
                                            edge_index, edge_type, interact_mat,
                                            self.relation_weight, flag)                                 
        entity_agg, user_agg, item_agg_cf = self.Dr_Norm(user_emb, entity_emb, item_emb_cf, mess_dropout, 0.03)

        # entity_res_emb = torch.add(entity_res_emb, entity_agg)
        # user_res_emb = torch.add(user_res_emb, user_agg)
        # item_emb_cf_res = torch.add(item_emb_cf_res, item_agg_cf)


        flag = "Pon"
        entity_emb, user_emb, item_emb_cf = self.convs(entity_agg, user_agg, item_agg_cf,
                                            edge_index, edge_type, interact_mat,
                                            self.relation_weight, flag)                                
        entity_agg, user_agg, item_agg_cf = self.Dr_Norm(user_emb, entity_emb, item_emb_cf, mess_dropout, 0.03)

        # entity_res_emb = torch.add(entity_res_emb, entity_agg)
        # user_res_emb = torch.add(user_res_emb, user_agg)
        # item_emb_cf_res = torch.add(item_emb_cf_res, item_agg_cf)


        flag = "Pon"
        entity_emb, user_emb, item_emb_cf = self.convs(entity_agg, user_agg, item_agg_cf,
                                            edge_index, edge_type, interact_mat,
                                            self.relation_weight, flag)                                 
        entity_agg, user_agg, item_agg_cf = self.Dr_Norm(user_emb, entity_emb, item_emb_cf, mess_dropout, 0.03)

        entity_res_emb = torch.add(entity_res_emb, entity_agg)
        user_res_emb = torch.add(user_res_emb, user_agg)
        item_emb_cf_res = torch.add(item_emb_cf_res, item_agg_cf)

        return entity_res_emb, user_res_emb, item_emb_cf_res



class Recommender(nn.Module):
    def __init__(self, data_config, args_config, graph, adj_mat):
        super(Recommender, self).__init__()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_relations = data_config['n_relations']
        self.n_entities = data_config['n_entities']  # include items
        self.n_nodes = data_config['n_nodes']  # n_users + n_entities

        self.margin_ccl = args_config.margin
        self.num_neg_sample = args_config.num_neg_sample

        self.decay = args_config.l2
        self.emb_size = args_config.dim
        self.node_dropout = args_config.node_dropout
        self.node_dropout_rate = args_config.node_dropout_rate
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate
        self.loss_f = args_config.loss_f
        self.device = torch.device("cuda:" + str(args_config.gpu_id)) if args_config.cuda \
                                                                      else torch.device("cpu")
        self.dropout_p = 0.5
        self.adj_mat = adj_mat
        self.edge_index, self.edge_type = self._get_edges(graph)

        self._init_weight()
        self._init_loss_function()

        self.gcn = self._init_model()

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.all_embed = initializer(torch.empty(self.n_nodes, self.emb_size))
        self.all_embed = nn.Parameter(self.all_embed)
        self.item_emb_cf = nn.Parameter(initializer(torch.empty(self.n_items, self.emb_size)))

        # [n_users, n_entities]
        self.interact_mat = self._convert_sp_mat_to_sp_tensor(self.adj_mat).to(self.device)


    def _init_model(self):
        return GraphConv(channel=self.emb_size,
                         n_users=self.n_users,
                         n_relations=self.n_relations,
                         n_items=self.n_items,
                         interact_mat=self.interact_mat,
                         device=self.device,
                         node_dropout_rate=self.node_dropout_rate,
                         mess_dropout_rate=self.mess_dropout_rate)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)



    def _get_edges(self, graph):
        graph_tensor = torch.tensor(list(graph.edges))  # [-1, 3]
        index = graph_tensor[:, :-1]  # [-1, 2]
        type = graph_tensor[:, -1]  # [-1, 1]
        return index.t().long().to(self.device), type.long().to(self.device)

    def _init_loss_function(self):
        if self.loss_f == "inner_bpr":
            self.loss = self.create_inner_bpr_loss
        elif self.loss_f == "dis_bpr":
            self.loss = self.create_dis_bpr_loss
        elif self.loss_f == 'contrastive_loss':
            self.loss = self.create_contrastive_loss
        else:
            raise NotImplementedError


    def forward(self, batch=None):
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items'].view(-1)

        user_emb = self.all_embed[:self.n_users, :]
        entity_emb = self.all_embed[self.n_users:, :]
        # entity_gcn_emb: [n_entity, channel]
        # user_gcn_emb: [n_users, channel]
        entity_gcn_emb, user_gcn_emb, item_gcn_emb_cf = self.gcn(user_emb,
                                                     entity_emb,
                                                     self.item_emb_cf,
                                                     self.edge_index,
                                                     self.edge_type,
                                                     self.interact_mat,
                                                     mess_dropout=self.mess_dropout,
                                                     node_dropout=self.node_dropout)

        u_e = user_gcn_emb[user]
        pos_e, neg_e = entity_gcn_emb[pos_item], entity_gcn_emb[neg_item]
        pos_e_cf, neg_e_cf = item_gcn_emb_cf[pos_item], item_gcn_emb_cf[neg_item]
        loss1 = self.loss(u_e, pos_e, neg_e, pos_e_cf, neg_e_cf)

        loss = loss1

        return loss,loss1

    def generate(self):
        user_emb = self.all_embed[:self.n_users, :]
        entity_emb = self.all_embed[self.n_users:, :]
        # entity_gcn_emb, user_gcn_emb, item_gcn_emb_cf = entity_emb, user_emb, self.item_emb_cf
        entity_gcn_emb, user_gcn_emb, item_gcn_emb_cf = self.gcn(user_emb,
                                                    entity_emb,
                                                    self.item_emb_cf,
                                                    self.edge_index,
                                                    self.edge_type,
                                                    self.interact_mat,
                                                    mess_dropout=False, node_dropout=False)

        entity_gcn_emb[:self.n_items] += item_gcn_emb_cf
        return entity_gcn_emb, user_gcn_emb



    def rating(self, u_g_embeddings, i_g_embeddings):
        if self.loss_f == "inner_bpr":
            return torch.matmul(u_g_embeddings, i_g_embeddings.t()).detach().cpu()

        elif self.loss_f == 'contrastive_loss':
            # u_g_embeddings = F.normalize(u_g_embeddings)
            # i_g_embeddings = F.normalize(i_g_embeddings)
            return torch.cosine_similarity(u_g_embeddings.unsqueeze(1), i_g_embeddings.unsqueeze(0), dim=2).detach().cpu()

        else:
            n_user = len(u_g_embeddings)
            n_item = len(i_g_embeddings)
            hyper_rate_matrix = np.zeros(shape=(n_user, n_item))

            hyper_u_g_embeddings = expmap0(u_g_embeddings)
            hyper_i_g_embeddings = expmap0(i_g_embeddings)

            for i in range(n_user):
                # [1, dim]
                one_hyper_u = hyper_u_g_embeddings[i, :]
                # [n_item, dim]
                one_hyper_u = one_hyper_u.expand(n_item, -1)
                one_hyper_score = -1 * sq_hyp_distance(one_hyper_u, hyper_i_g_embeddings)
                hyper_rate_matrix[i, :] = one_hyper_score.squeeze().detach().cpu()

            return hyper_rate_matrix

    def create_contrastive_loss(self, u_e, pos_e, neg_e, pos_e_cf, neg_e_cf):
        batch_size = u_e.shape[0]

        u_e = F.normalize(u_e) #🌟[batch_size, emb_size]
        pos_e = F.normalize(pos_e) #🌟[batch_size, emb_size]
        neg_e = F.normalize(neg_e) #🌟[200*batch_size, emb_size]
        pos_e_cf = F.normalize(pos_e_cf)
        neg_e_cf = F.normalize(neg_e_cf)


        ui_pos = torch.relu(2 - (torch.cosine_similarity(u_e, pos_e, dim=1) + torch.cosine_similarity(u_e, pos_e_cf, dim=1)))
        #🌟ui_pos.shape: [batch_size] #the score from user u evaluate on item i: user[j]->item[j]
        users_batch = torch.repeat_interleave(u_e, self.num_neg_sample, dim=0)
        # print(ui_pos.shape)

        ui_neg1 = torch.relu(torch.cosine_similarity(users_batch, neg_e, dim=1) - self.margin_ccl)
        #🌟ui_neg1.shape: [200*batch_size]
        # print(ui_neg1.shape)
        ui_neg1 = ui_neg1.view(batch_size, -1)
        x = ui_neg1>0
        ui_neg_loss1 = torch.sum(ui_neg1,dim=-1)/(torch.sum(x, dim=-1) + 1e-5)

        ui_neg2 = torch.relu(torch.cosine_similarity(users_batch, neg_e_cf, dim=1) - self.margin_ccl)
        ui_neg2 = ui_neg2.view(batch_size, -1)
        x = ui_neg2 > 0
        ui_neg_loss2 = torch.sum(ui_neg2, dim=-1) / (torch.sum(x, dim=-1) + 1e-5)

        loss = ui_pos + ui_neg_loss1 + ui_neg_loss2

        return loss.mean()


    def create_inner_bpr_loss(self, users, pos_items, neg_items):
        batch_size = users.shape[0]
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        cf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))
        # cul regularizer
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size

        return cf_loss + emb_loss

    def create_dis_bpr_loss(self, users, pos_items, neg_items):
        hyper_users = expmap0(users)
        hyper_pos_items = expmap0(pos_items)
        hyper_neg_items = expmap0(neg_items)

        hyper_pos_dis = sq_hyp_distance(hyper_users, hyper_pos_items)
        hyper_neg_dis = sq_hyp_distance(hyper_users, hyper_neg_items)
        # hyper_pos_dis = hyp_distance(hyper_users, hyper_pos_items)
        # hyper_neg_dis = hyp_distance(hyper_users, hyper_neg_items)

        cf_loss = -1 * torch.mean(nn.LogSigmoid()
                                  (hyper_neg_dis - hyper_pos_dis))
        return cf_loss


    # def plot(self, ax, u_emb, item_emb, item_list, flag, isflag):
    #     L = 3000
    #     i_emb = torch.cat((item_emb[:L],u_emb),0)
    #     tsne = TSNE(n_components=2)
    #     tsne.fit_transform(i_emb.detach().cpu())
    #     x = tsne.embedding_[:, 0]
    #     y = tsne.embedding_[:, 1]

    #     for i in range(L):
    #         if i in item_list:
    #             ax.scatter(x[i], y[i], c='b', s=1000, edgecolor = 'black')
    #         else:
    #             ax.scatter(x[i], y[i], c='deepskyblue', alpha = 0.3, s=500, edgecolor = 'black')
    #     ax.scatter(x[L], y[L], c='red', s=1500, edgecolor = 'black')
    #     if isflag:
    #         ax.set_title(flag, fontsize=90, color='black')
    #     plt.yticks([])
    #     plt.xticks([])  


    # def visualize(self, user_set):
    #     user_emb = self.all_embed[:self.n_users, :]
    #     entity_emb = self.all_embed[self.n_users:, :]
    #     with torch.no_grad():
    #         entity_emb[:self.n_items] += self.item_emb_cf
    #     entity_gcn_emb, user_gcn_emb = self.generate()

    #     user = 7
    #     item_list = user_set[user]
    #     print(item_list)
    #     u_emb1 = user_emb[user].view(1,-1)
    #     u_emb2 = user_gcn_emb[user].view(1,-1)
    #     plt.figure(figsize = (90, 40), dpi = 300)
    #     ax1 = plt.subplot(231)
    #     self.plot(ax1, u_emb1, entity_emb[:self.n_items], item_list, "User "+str(user), True)
    #     ax2 = plt.subplot(234)
    #     self.plot(ax2, u_emb2, entity_gcn_emb[:self.n_items], item_list, "User "+str(user), False)

    #     user = 11
    #     item_list = user_set[user]
    #     print(item_list)
    #     u_emb1 = user_emb[user].view(1,-1)
    #     u_emb2 = user_gcn_emb[user].view(1,-1)
    #     ax3 = plt.subplot(232)
    #     self.plot(ax3, u_emb1, entity_emb[:self.n_items], item_list, "User "+str(user), True)
    #     ax4 = plt.subplot(235)
    #     self.plot(ax4, u_emb2, entity_gcn_emb[:self.n_items], item_list, "User "+str(user), False)


    #     user = 12
    #     item_list = user_set[user]
    #     print(item_list)
    #     u_emb1 = user_emb[user].view(1,-1)
    #     u_emb2 = user_gcn_emb[user].view(1,-1)
    #     ax5 = plt.subplot(233)
    #     self.plot(ax5, u_emb1, entity_emb[:self.n_items], item_list, "User "+str(user), True)
    #     ax6 = plt.subplot(236)
    #     self.plot(ax6, u_emb2, entity_gcn_emb[:self.n_items], item_list, "User "+str(user), False)

    #     plt.tight_layout(pad=12,h_pad=12,w_pad=36.0)  
    #     plt.savefig('./visualization/'+'Last-Fm.jpg')
    #     plt.close()


    def plot(self, ax, u_emb, item_emb, item_list, flag, isflag):
        L = 8000
        i_emb = torch.cat((item_emb[:L],u_emb),0)
        tsne = TSNE(n_components=2)
        tsne.fit_transform(i_emb.detach().cpu())
        x = tsne.embedding_[:, 0]
        y = tsne.embedding_[:, 1]

        for i in range(L):
            if i in item_list:
                ax.scatter(x[i], y[i], c='b', s=500, edgecolor = 'black')
            else:
                ax.scatter(x[i], y[i], c='deepskyblue', alpha = 0.1, s=100, edgecolor = 'black')
        ax.scatter(x[L], y[L], c='red', s=800, edgecolor = 'black')
        if isflag:
            ax.set_title(flag, fontsize=60, color='black')
        plt.yticks([])
        plt.xticks([])  


    # def visualize(self, user_set):
    #     user_emb = self.all_embed[:self.n_users, :]
    #     entity_emb = self.all_embed[self.n_users:, :]
    #     with torch.no_grad():
    #         entity_emb[:self.n_items] += self.item_emb_cf
    #     entity_gcn_emb, user_gcn_emb = self.generate()

    #     for i in range(20):
    #         user = i
    #         item_list = user_set[user]
    #         print(item_list)
    #         u_emb1 = user_emb[user].view(1,-1)
    #         u_emb2 = user_gcn_emb[user].view(1,-1)
    #         plt.figure(figsize = (15, 25), dpi = 300)
    #         ax1 = plt.subplot(211)
    #         self.plot(ax1, u_emb1, entity_emb[:self.n_items], item_list, "User "+str(user), True)
    #         ax2 = plt.subplot(212)
    #         self.plot(ax2, u_emb2, entity_gcn_emb[:self.n_items], item_list, "User "+str(user), False)
    #         plt.savefig('./visualization/train/'+str(user)+'.jpg')
    #         plt.close()


    def visualize1(self, ax, user, user_set):
        user_emb = self.all_embed[:self.n_users, :]
        entity_emb = self.all_embed[self.n_users:, :]
        with torch.no_grad():
            entity_emb[:self.n_items] += self.item_emb_cf

        item_list = user_set[user]
        print(item_list)
        u_emb = user_emb[user].view(1,-1)
        self.plot(ax, u_emb, entity_emb[:self.n_items], item_list, "User "+str(user), False)



    def visualize2(self, ax, user, user_set):
        entity_gcn_emb, user_gcn_emb = self.generate()

        item_list = user_set[user]
        print(item_list)
        u_emb = user_gcn_emb[user].view(1,-1)
        self.plot(ax, u_emb, entity_gcn_emb[:self.n_items], item_list, "User "+str(user), False)







