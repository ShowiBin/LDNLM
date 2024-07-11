from torch.utils.data import Dataset,DataLoader
import torch
from torch import nn
import cv2
import matplotlib.pyplot as plt
import numpy as np
from linear_attention_transformer import LinearAttentionTransformer

rep_V=[]# for visualization
w=[]



def merge_img(img_sets, size):
    '''将多个图像按照size拼起来'''#stand_width是patch的标准边长

    res = np.zeros([(int(size[0]/img_sets[0].shape[0])+1)*img_sets[0].shape[0],\
        (int(size[1]/img_sets[0].shape[1])+1)*img_sets[0].shape[1]])


    #补齐操作
    tmp_sets = []
    W,H = img_sets[0].shape[0],img_sets[0].shape[1]
    for img in img_sets:
        tmp_img = np.zeros([W,H])
        tmp_img[:img.shape[0],:img.shape[1]] = img
        tmp_sets.append(tmp_img)
    img_sets = tmp_sets
    # plt.imshow(img_sets[2],'gray');plt.show()
    single_img_w = img_sets[0].shape[-1]
    
    
    num_imgs_line = int(size[-1]/single_img_w) + 1
    # print(num_imgs_line)
    for i in range(num_imgs_line):
        for j in range(num_imgs_line):
            
            res[i*single_img_w:(i+1)*single_img_w,\
                j*single_img_w:(j+1)*single_img_w] = img_sets[int(i*num_imgs_line+j)]
            # print(i,j,num_imgs_line)
            # plt.imshow(res,'gray');plt.show()
#     def notallsam():
#         a = img_sets[0].shape[0]
#         for i in img_sets:
#             if i.shape[0] != a:
#                 return True
#         return False
#     if notallsam():
#         num_imgs_line +=  1
        
        
# #     print(num_imgs_line)
#     for pos,i in enumerate(img_sets):
#         line = int(pos/num_imgs_line)
#         col = int(pos%num_imgs_line)
# #         if stand_width not in i.size:
# #             res
#         try:
#             res[line*single_img_w: (line+1)*single_img_w,\
#             col*single_img_w: (col+1)*single_img_w]=i
#             # print('0',line,single_img_w)
#             # plt.imshow(img_sets[pos],'gray')
#             # plt.show()
#         except:
#             try:
#                 res[line*single_img_w: ,\
#                     col*single_img_w: (col+1)*single_img_w]=i
#                 print('1',line,single_img_w)
#             except:
#                 try:
#                     res[line*single_img_w: (line+1)*single_img_w,\
#                         col*single_img_w:]=i
#                     print('2',line,single_img_w)
#                 except:
#                     plt.imshow(res,'gray')
#                     plt.show()
#                     print(line*single_img_w,col*single_img_w)
#                     res[line*single_img_w:,\
#                         col*single_img_w:]=i
#         # plt.imshow(res[:size[0],:size[1]],'gray')
#         plt.show()
    return res[:size[0],:size[1]]
        

def get_key_patch(x_nei,kernel_size):
    '''通过领域获取所有相似矩阵'''
#     print(x_nei.shape,kernel_size)
    i,j = (x_nei.shape[-2] - 1)/2 , (x_nei.shape[-1] - 1)/2
    temp = []
    for m in range(x_nei.shape[-2] - 2 * kernel_size):
        for n in range(x_nei.shape[-1]  - 2 * kernel_size):
            key_patch = x_nei[:,:,m + kernel_size -kernel_size:m + kernel_size+kernel_size + 1\
                      ,n + kernel_size -kernel_size:n+kernel_size+kernel_size + 1]
            #batch_size, nei_num, 然后才是一个nei
            temp.append(key_patch)
            del key_patch
    temp_ = torch.stack(temp).permute(1,2,0,3,4)
    del temp
    return temp_



def scaleDotproductAtt(q, k ,v):
    d_q = len(q[0])
#     print(q.shape,k.shape)
    out = torch.matmul(q,k.transpose(-1,-2))#b*l*q @ b*q*l => [b*l*l]
    # w.append(out)
    out = out/((d_q)**0.5)
    out = nn.Softmax(dim=-1)(out)#对最后一维度做变化，因为这维度要去和下一个V计算的
    out =  torch.matmul(out,v)#b*l*l @ b*l*v [b*l*d_v]
    return out#[b*l*d_v]


def get_sincos_enc_posEmb(n_position, d_model):
    '''
    generate a positional embedding by sin cos formular
    '''
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)
    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_model)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.FloatTensor(sinusoid_table)



class MultiHeadATT(nn.Module):
    def __init__(self,d_model,d_k=64,d_q=64,d_v=64,n_head=1):
        super(MultiHeadATT,self).__init__()
        
        self.n_head = n_head
        self.d_q = d_q
        self.d_k = d_k
        self.d_v = d_v
        
        
#         size = (2*kernel_size + 1)#9
        
#         d_model = size**2
        
        self.W_k = nn.Linear(d_model,d_k*n_head)
        self.W_q = nn.Linear(d_model,d_q*n_head)
        self.W_v = nn.Linear(d_model,d_v*n_head)
        self.L = nn.Linear(d_v*n_head, d_model)
        
        self.LN = nn.LayerNorm(d_model)
        
    def forward(self,Q,K,V):
        residual = Q
        batch_size = Q.shape[0]
        
#         print('K,Q',K.shape,Q.shape)
        K = self.W_k(K).view(batch_size,-1,self.n_head,self.d_k).transpose(1,2)
        Q = self.W_q(Q).view(batch_size,-1,self.n_head,self.d_q).transpose(1,2)
        V = self.W_v(V).view(batch_size,-1,self.n_head,self.d_v).transpose(1,2)
        
        out = scaleDotproductAtt(Q,K,V)
        
        out = out.transpose(1,2).contiguous().view(batch_size,-1,self.n_head*self.d_v)
        out = self.L(out)
        
        out = self.LN(out+residual)
        return out

class ATT(nn.Module):
    def __init__(self,d_model,d_k=64,d_q=64,d_v=64,n_head=1):
        super(ATT,self).__init__()
        
        self.n_head = n_head
        self.d_v = d_v
        

        
        self.MHATT = MultiHeadATT(d_model,d_k,d_q,d_v,n_head)
        # self.LMHATT = LinearMultiHeadAttention(d_model, n_head, dropout=0.1)
        self.LN = nn.LayerNorm(d_model)
        
        self.posw_ffn = nn.Sequential(
            nn.Linear(d_model, d_model*2),
            nn.ReLU(),
            nn.Linear(d_model*2,d_model)
        )
        
    def forward(self,patchs):
        
        out = self.MHATT(patchs,patchs,patchs)   
        # out = self.LMHATT(query=patchs,key=patchs,value=patchs)  
        residual = out
        out = self.posw_ffn(out)
        return self.LN(out + residual)

    

class ATT_NLM(nn.Module):
    def __init__(self,nei_size,kernel_size,n_head=2,n_layers = 1, d_model=None,d_k=64,d_q=64,d_v=64,pos_cod_len=2048):
        super(ATT_NLM,self).__init__()
        self.nei_size = nei_size
        
        self.kernel_size = kernel_size
        
#         self.PAD = torch.nn.ConstantPad2d(kernel_size,0.0)#这里只pad kernel_size
        size = 2*kernel_size + 1
        if d_model == None:
            d_model = size**2
            while d_model % n_head != 0:## 2023年12月5日，为了满足可以被n_head个头分解开，所以使得其可以被模
                d_model += 1
            
        
        # 构建一个CNN，用来提取特征
        self.CNN = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=size**2,kernel_size=2*kernel_size+1,stride=1,padding=kernel_size),
            nn.ReLU(),
        )
        
        self.L_transfer = nn.Linear(size**2,d_model)
        
        self.pos_emb = nn.Embedding.from_pretrained(\
        get_sincos_enc_posEmb(pos_cod_len,d_model),freeze=False)#不冻结，让他学
    #这里的初始位置编码只有左右位置信息，无上下位置信息。
    #之后可以研究下，怎么为初始化位置编码加上上下位置信息
    
        # self.ATT = nn.ModuleList([ATT(d_model,d_k,d_q,d_v,n_head) for _ in range(n_layers)])
#             ATT(d_model,d_k,d_q,d_v,n_head)#双头
        self.ATT_Linear = LinearAttentionTransformer(# 2023年12月6日， 用下述线性att代替att
            dim = d_model,
            heads = n_head,
            depth = n_layers,
            max_seq_len = 18000,
            local_attn_window_size = 2*nei_size+1,
            n_local_attn_heads = n_head
        ).cuda()
        self.PRE = nn.Sequential(
            nn.Linear(d_model,d_model),
            nn.ReLU(),
            nn.Linear(d_model,1)#regression
        )
        
    def forward(self,x):
        # x: [batch_size,h,w]
        x = x.float()
        x = x.unsqueeze(1)# [batch_size,1,h,w]
        patchs = self.CNN(x)# [batch_size,size**2,h,w]
        patchs = patchs.flatten(-2)# [batch_size,size**2,w*h]
        patchs = patchs.transpose(1,2)# [batch_size,w*h, size**2]
        
        patchs = self.L_transfer(patchs)
        
        #add position encoding 
        out  = patchs+ self.pos_emb(torch.LongTensor([list(range(patchs.shape[1]))]).to(x.device))
        
        # for layer in self.ATT:
        #     out = layer(out)#逐个通过transformer encoder layer
        out = self.ATT_Linear(out)
        rep_V.append(out)#可解释性
        out = self.PRE(out)#做个回归好了
        
#         out = self.LN(out+residual)

        #把out切割一下
#         print(out.shape)
        out = out.unsqueeze(-1).reshape(patchs.shape[0],2*self.nei_size+1,\
                                        2*self.nei_size+1)
#         out = out[:,self.kernel_size:-self.kernel_size,self.kernel_size:-self.kernel_size]#只pad了一层，所以这里不用恢复了
        return out



class ATTNLM():
    def __init__(self,nei_size,kernel_size,model_path,nhead=1,nlayer=1,pos_cod_len=2048,device = 'cuda:0'):

        self.nei_size = nei_size
        self.kernel_size = kernel_size
        

        self.model = ATT_NLM(nei_size,kernel_size,n_head=nhead,n_layers=nlayer,pos_cod_len=pos_cod_len).to(device)

        # model = ATT_NLM(nei_size,kernel_size,n_head=2,n_layers=1).to(device)
        # self.model = ImageLinearAttention(
        #   chan = 1,
        #   heads = nhead,
        #   kernel_size=2*kernel_size+1,
        #   padding=kernel_size,
        #   key_dim = 64       # can be decreased to 32 for more memory savings
        # ).to(device)
        ckpt = torch.load(model_path)#加载参数

        self.model.load_state_dict(ckpt)#将参数用到model上

        self.PAD = torch.nn.ConstantPad2d(kernel_size,0.0)#.to(device)\

    def denoise(self,blur_test,batch_size = 128):
        
        nei_size = self.nei_size
        kernel_size = self.kernel_size

        # print(nei_size,kernel_size)
        shape = blur_test.shape

        blur_test = self.PAD(blur_test)
        
        test_img_sets = []
        i = nei_size+kernel_size
        gap = 2*nei_size + 1

        # position_bias = 0#记录下边界的特殊patch相对于sets中最后一个的相对位置
        edge_patches = dict()#{'position':
                                #{'patch':patch,'old_size':old_size} }

        while(i-nei_size<=blur_test.shape[0] - (kernel_size)):
            j = nei_size + kernel_size
            
            while(j-nei_size  <= blur_test.shape[1] - (kernel_size) ):
        #         print(j-nei_size-kernel_size,j+nei_size+kernel_size+1)
                temp = blur_test[i-nei_size-kernel_size:i+nei_size+kernel_size+1,j-nei_size-kernel_size:j+nei_size+kernel_size+1]
                #判断temp是不是边界的patch
                if j+nei_size+kernel_size+1 >= blur_test.shape[1]:
                #将patch移动位置，使得满足size的要求
                    old_size = list(temp.detach().numpy().shape)
                    if i +nei_size+kernel_size+1 >= blur_test.shape[0]:#这是最右下的patch
                        
                        old_size[0] -= 2*kernel_size
                        old_size[1] -= 2*kernel_size
                        temp = blur_test[blur_test.shape[0] - 2*(kernel_size+nei_size) - 1:blur_test.shape[0],\
                                        blur_test.shape[1] - 2*(kernel_size+nei_size) - 1:blur_test.shape[1]]
                    else:
                        old_size[1] -= 2*kernel_size
                        temp = blur_test[i-nei_size-kernel_size:i+nei_size+kernel_size+1,\
                                        blur_test.shape[1] - 2*(kernel_size+nei_size) - 1:blur_test.shape[1]]
                    edge_patches[len(test_img_sets) + len(edge_patches)] = {'old_size':old_size,'patch':temp}
                    # print(edge_patches.keys())
                    
                elif i+nei_size+kernel_size+1 >= blur_test.shape[0]:#这是下边界的
        #             print('I',len(test_img_sets) + len(edge_patches))
                    old_size = list(temp.size())
                    old_size[0] -= 2*kernel_size
                    temp = blur_test[blur_test.shape[0] - 2*(kernel_size+nei_size) - 1:blur_test.shape[0],\
                                        j-nei_size-kernel_size:j+nei_size+kernel_size+1]
                    edge_patches[len(test_img_sets) + len(edge_patches)] = {'old_size':old_size,'patch':temp}
                    
                    # print(old_size,j)
                    
                
                #将此些特殊的patch存储到dict中
                #
                else:
        #             print(i,j)
                    test_img_sets.append(temp)
                
                j+=gap
            i+=gap
        keys = edge_patches.keys()
        edge_sets = [edge_patches[i]['patch'] for i in keys]
        all_sets = (test_img_sets + edge_sets)
        # edge_sets = [i['patch'] for i in edge_patches]
        res = []
        with torch.no_grad():
            print(len(all_sets))
            import time
            t1 = time.time()
            for i in range(0,len(all_sets),batch_size):
                if i+batch_size > len(all_sets):
                    x = torch.stack(all_sets[i:]).float()
                else:
                    x = torch.stack(all_sets[i:i+batch_size]).float()
                # x = x.unsqueeze(1)
                
                x = x[:,kernel_size:-kernel_size,kernel_size:-kernel_size]
                
                y = self.model(x.to('cuda'))    
                res.extend(y.cpu())
                torch.cuda.empty_cache()
            t2 = time.time()
            print('time:',t2-t1)
        #给特殊的patch做filter
        norm_res,edge_res = res[:len(test_img_sets)],res[len(test_img_sets):]
        # print(keys)
        for i in range(len(keys)):
            k = list(keys)[i]
            size = edge_patches[k]['old_size']
            
            norm_res.insert(k,edge_res[i][-1*size[0]:,-1*size[1]:])
        
        # return norm_res
        res_img = merge_img(norm_res,shape)

        return res_img,norm_res,rep_V,w