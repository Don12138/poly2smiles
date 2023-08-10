import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import pdb

def index_scatter(sub_data, all_data, index):
    d0, d1 = all_data.size()
    buf = torch.zeros_like(all_data).scatter_(0, index.repeat(d1, 1).t(), sub_data)
    mask = torch.ones(d0, device=all_data.device).scatter_(0, index, 0)

    return all_data * mask.unsqueeze(-1) + buf


def index_select_ND(source, dim, index):    #选出source的dim纬度中，索引为index的值
    index_size = index.size()
    suffix_dim = source.size()[1:]
    final_size = index_size + suffix_dim
    target = source.index_select(dim, index.view(-1))

    return target.view(final_size)

class NumericalTokenRegression(nn.Module):
    # property为list[dict]
    # 一个dict中包含：起始位置，性质长度，小数点位置,最大值最小值（以平衡loss）
    # beta越大，算出的带有梯度的预测token越准，但是梯度信息越少，初始为10
    def __init__(self, args, beta, vocab):
        super().__init__()
        self.args = args
        self.beta = beta
        self.vocab = vocab
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vector_tokens = torch.tensor(np.arange(len(vocab)),dtype=torch.float,device=self.device)
        # 初始化权重的字典
        weight_dict = {
            1: 10,
            2: 1,
            3: 0.1,
            5: 0.05,
            7: 10,
            8: 1,
            9: 0.1,
            11: 0.05,
            13: 10,
            14: 1,
            15: 0.1,
            17: 0.05
        }

        # 创建权重张量
        self.weight = torch.zeros(args.predict_max_len,device = "cuda")
        for index, value in weight_dict.items():
            self.weight[index] = value
        # self.property_num = len(property_info)
        # all_keys = ["start","len","point","max","min"]
        # self.property_info_list = [[data_dict[key] for data_dict in property_info] for key in all_keys]
        # self.weight = self.get_weight(args.predict_max_len,self.property_info_list)
        # print(self.result_lists)


    def forward(self,x,target):
        # target = target.to(self.device)
        # x => [batch_size, vocab_size, str_length]
        # target => [batch_size, str_length]
        # pdb.set_trace()
        x = self.softmax_beta(x,beta=self.beta,dim=1)
        x = torch.matmul(x.permute(0,2,1), self.vector_tokens)
        # print(((x - target) * self.weight[:x.shape[1]]))
        loss = torch.sqrt(torch.mean(((x - target) * self.weight[:x.shape[1]]) ** 2))
        return loss


    def softmax_beta(self, x, beta, dim=None):
        x = x * beta
        e_x = torch.exp(x - torch.max(x, dim=dim, keepdim=True)[0])
        return e_x / e_x.sum(dim=dim, keepdim=True)

    # def get_weight(self, weight_length, info):
    #     weight = torch.zeros(weight_length,device=self.device)
    #     # ["start","len","point","max","min"]
    #     for property_iter in range(len(info[0])):
    #         start_index = info[0][property_iter]
    #         point = info[2][property_iter]
    #         property_length = info[1][property_iter]
    #         # dif = self.property_info_list[3][property_iter] - self.property_info_list[4][property_iter]
    #         dif = 1
    #         for i in range(point):
    #             weight[start_index + i] = (10. ** (point - i - 1)) / dif
    #         weight[start_index + point] = 1. / dif
    #         for i in range(property_length - point - 1):
    #             weight[start_index + i + point + 1] = ( 10 ** (-1 - i)) / dif
    #     return weight

class WeightCrossEntropyLoss(nn.Module):

    def __init__(self, args,ignore_index,reduction):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        # 初始化权重的字典
        weight_dict = {
            0: 2.,
            1: 0.5,
            2: 0.5,
            3: 0.5,
            4: 2.,
            5: 0.5,
            6: 2.,
            7: 0.5,
            8: 0.5,
            9: 0.5,
            10: 2.,
            11: 0.5,
            12: 2.,
            13: 0.5,
            14: 0.5,
            15: 0.5,
            16: 2.,
            17: 0.5,
            18: 2.,
        }
        # 创建权重张量
        self.weight = torch.ones(args.predict_max_len,device = "cuda")
        for index, value in weight_dict.items():
            self.weight[index] = value
        
    def forward(self, input,target):
        # pdb.set_trace()
        return F.cross_entropy(input, target,ignore_index=self.ignore_index,weight=self.weight[:input.shape[1]],reduction=self.reduction)