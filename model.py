import torch
import torch.nn as nn
import torch.nn.init
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import grad
import torch.backends.cudnn as cudnn
from torch.nn.modules.activation import ReLU
from torch.nn.modules.linear import Linear
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import numpy as np
from loss import TripletLoss
from basic.bigfile import BigFile
from collections import OrderedDict


def get_we_parameter(vocab, w2v_file):
    w2v_reader = BigFile(w2v_file)
    ndims = w2v_reader.ndims

    we = []
    # we.append([0]*ndims)
    for i in range(len(vocab)):
        try:
            vec = w2v_reader.read_one(vocab.idx2word[i])
        except:
            vec = np.random.uniform(-1, 1, ndims)
        we.append(vec)
    print('getting pre-trained parameter for word embedding initialization', np.shape(we)) 
    return np.array(we)


def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X


def xavier_init_fc(fc):
    """Xavier initialization for the fully connected layer
    """
    r = np.sqrt(6.) / np.sqrt(fc.in_features +
                             fc.out_features)
    fc.weight.data.uniform_(-r, r)
    fc.bias.data.fill_(0)

class MFC(nn.Module):
    """
    Multi Fully Connected Layers
    """
    def __init__(self, fc_layers, dropout, have_dp=True, have_bn=False, have_last_bn=False):
        super(MFC, self).__init__()
        # fc layers
        self.n_fc = len(fc_layers)
        if self.n_fc > 1:
            if self.n_fc > 1:
                self.fc1 = nn.Linear(fc_layers[0], fc_layers[1])
            if self.n_fc > 2:
                self.fc2 = nn.Linear(fc_layers[1], fc_layers[2])
            if self.n_fc > 3:
                self.fc3 = nn.Linear(fc_layers[2], fc_layers[3])
            if self.n_fc > 4:
                self.fc4 = nn.Linear(fc_layers[3], fc_layers[4])

            self.relu = ReLU()

            # dropout
            self.have_dp = have_dp
            if self.have_dp:
                self.dropout = nn.Dropout(p=dropout)

            # batch normalization
            self.have_bn = have_bn
            self.have_last_bn = have_last_bn
            if self.have_bn:
                if self.n_fc > 1 and self.have_last_bn:
                    self.bn_1 = nn.BatchNorm1d(fc_layers[-1])

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        if self.n_fc > 1:
            xavier_init_fc(self.fc1)
        if self.n_fc > 2:
            xavier_init_fc(self.fc2)
        if self.n_fc > 3:
            xavier_init_fc(self.fc3)
        if self.n_fc > 4:
            xavier_init_fc(self.fc4)

    def forward(self, inputs):

        if self.n_fc <= 1:
            features = inputs

        elif self.n_fc > 1:
            features = self.fc1(inputs)
            if self.n_fc > 2:
                features = features + self.relu(self.fc2(features))
            if self.n_fc > 3:
                features = features + self.relu(self.fc3(features))
            if self.n_fc > 4:
                features = features + self.relu(self.fc4(features))
            # batch normalization
            if self.have_bn and self.have_last_bn:
                features = self.bn_1(features)
            if self.have_dp:
                features = self.dropout(features)

        return features


class Video_multilevel_encoding(nn.Module):
    """
    Section 3.1. Video-side Multi-level Encoding
    """
    def __init__(self, opt):
        super(Video_multilevel_encoding, self).__init__()

        self.rnn_output_size = opt.visual_rnn_size*2
        self.dropout = nn.Dropout(p=opt.dropout)
        self.concate = opt.concate
        self.gru_pool = opt.gru_pool
        self.tag_vocab_size = opt.tag_vocab_size
        self.loss_fun = opt.loss_fun

        # visual bidirectional rnn encoder
        self.rnn = nn.GRU(opt.visual_feat_dim, opt.visual_rnn_size, batch_first=True, bidirectional=True)

        # visual 1-d convolutional network
        self.convs1 = nn.ModuleList([
                nn.Conv2d(1, opt.visual_kernel_num, (window_size, self.rnn_output_size), padding=(window_size - 1, 0)) 
                for window_size in opt.visual_kernel_sizes
                ])

        
    def forward(self, videos):
        """Extract video feature vectors."""
        videos, videos_origin, lengths, videos_mask = videos
        
        # Level 1. Global Encoding by Mean Pooling According
        org_out = videos_origin

        # Level 2. Temporal-Aware Encoding by biGRU
        gru_init_out, _ = self.rnn(videos)
        if self.gru_pool == 'mean':
            mean_gru = Variable(torch.zeros(gru_init_out.size(0), self.rnn_output_size)).cuda()
            for i, batch in enumerate(gru_init_out):
                mean_gru[i] = torch.mean(batch[:lengths[i]], 0)
            gru_out = mean_gru
        elif self.gru_pool == 'max':
            gru_out = torch.max(torch.mul(gru_init_out, videos_mask.unsqueeze(-1)), 1)[0]
        gru_out = self.dropout(gru_out)

        # Level 3. Local-Enhanced Encoding by biGRU-CNN
        videos_mask = videos_mask.unsqueeze(2).expand(-1,-1,gru_init_out.size(2)) # (N,C,F1)
        gru_init_out = gru_init_out * videos_mask
        con_out = gru_init_out.unsqueeze(1)
        con_out = [F.relu(conv(con_out)).squeeze(3) for conv in self.convs1]
        con_out = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in con_out]
        con_out = torch.cat(con_out, 1)
        con_out = self.dropout(con_out)

        # concatenation
        if self.concate == 'full':
            features = torch.cat((gru_out,con_out,org_out), 1)
        elif self.concate == 'reduced':  # level 2+3
            features = torch.cat((gru_out,con_out), 1)

        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(Video_multilevel_encoding, self).load_state_dict(new_state)

class Video_multilevel_encoding_with_support(nn.Module):
    """
    Section 3.1. Video-side Multi-level Encoding
    """
    def __init__(self, opt):
        super(Video_multilevel_encoding_with_support, self).__init__()

        self.style = opt.style
        self.rnn_output_size = opt.visual_rnn_size*2
        self.dropout = nn.Dropout(p=opt.dropout)
        self.concate = opt.concate
        self.gru_pool = opt.gru_pool
        self.tag_vocab_size = opt.tag_vocab_size
        self.loss_fun = opt.loss_fun

        # visual bidirectional rnn encoder
        self.rnn = nn.GRU(opt.visual_feat_dim, opt.visual_rnn_size, batch_first=True, bidirectional=True)

        # visual 1-d convolutional network
        self.convs1 = nn.ModuleList([
                nn.Conv2d(1, opt.visual_kernel_num, (window_size, self.rnn_output_size), padding=(window_size - 1, 0)) 
                for window_size in opt.visual_kernel_sizes
                ])
        
        self.k = nn.Linear(opt.visual_mapping_layers[0], opt.hidden_size, bias=True) 
        self.q = nn.Linear(opt.visual_mapping_layers[0], opt.hidden_size, bias=True)

        
    def forward(self, videos, support_videos, caption_ids, video_ids):
        """Extract video feature vectors."""
        videos, videos_origin, lengths, videos_mask = videos
        feature = self.encode_video(videos, videos_origin, lengths, videos_mask)

        if support_videos == None:
            return feature
        support_videos, support_videos_origin, support_lengths, support_videos_mask = support_videos
        s_feature = []
        for i in range(support_videos.size(1)):
            s_feature.append(self.encode_video(support_videos[:,i,:,:], support_videos_origin[:,i,:], support_lengths[:,i], support_videos_mask[:,i,:]))
        s_feature = torch.stack(s_feature,1)
        key = self.k(s_feature)
        query = self.q(feature)
        w = F.softmax(torch.bmm(key,query.unsqueeze(2)),dim=1)
        w = w.repeat(1,1,s_feature.shape[2])
        gated_feature = torch.sum(w * s_feature,dim=1)

        return feature + gated_feature

        
    def encode_video(self, videos, videos_origin, lengths, videos_mask):

        # Level 1. Global Encoding by Mean Pooling According
        org_out = videos_origin

        # Level 2. Temporal-Aware Encoding by biGRU
        gru_init_out, _ = self.rnn(videos)
        if self.gru_pool == 'mean':
            mean_gru = Variable(torch.zeros(gru_init_out.size(0), self.rnn_output_size)).cuda()
            for i, batch in enumerate(gru_init_out):
                mean_gru[i] = torch.mean(batch[:int(lengths[i])], 0)
            gru_out = mean_gru
        elif self.gru_pool == 'max':
            gru_out = torch.max(torch.mul(gru_init_out, videos_mask.unsqueeze(-1)), 1)[0]
        gru_out = self.dropout(gru_out)

        # Level 3. Local-Enhanced Encoding by biGRU-CNN
        videos_mask = videos_mask.unsqueeze(2).expand(-1,-1,gru_init_out.size(2)) # (N,C,F1)
        gru_init_out = gru_init_out * videos_mask
        con_out = gru_init_out.unsqueeze(1)
        con_out = [F.relu(conv(con_out)).squeeze(3) for conv in self.convs1]
        con_out = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in con_out]
        con_out = torch.cat(con_out, 1)
        con_out = self.dropout(con_out)

        # concatenation
        if self.concate == 'full':
            features = torch.cat((gru_out,con_out,org_out), 1)
        elif self.concate == 'reduced':  # level 2+3
            features = torch.cat((gru_out,con_out), 1)

        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(Video_multilevel_encoding_with_support, self).load_state_dict(new_state)

class Text_multilevel_encoding(nn.Module):
    """
    Section 3.2. Text-side Multi-level Encoding
    """
    def __init__(self, opt):
        super(Text_multilevel_encoding, self).__init__()
        self.word_dim = opt.word_dim
        self.we_parameter = opt.we_parameter
        self.rnn_output_size = opt.text_rnn_size*2
        self.dropout = nn.Dropout(p=opt.dropout)
        self.concate = opt.concate
        self.gru_pool = opt.gru_pool
        self.loss_fun = opt.loss_fun
        
        # visual bidirectional rnn encoder
        self.embed = nn.Embedding(opt.vocab_size, opt.word_dim)
        self.rnn = nn.GRU(opt.word_dim, opt.text_rnn_size, batch_first=True, bidirectional=True)

        # visual 1-d convolutional network
        self.convs1 = nn.ModuleList([
                nn.Conv2d(1, opt.text_kernel_num, (window_size, self.rnn_output_size), padding=(window_size - 1, 0)) 
                for window_size in opt.text_kernel_sizes
                ])

        self.init_weights()

    def init_weights(self):
        if self.word_dim == 500 and self.we_parameter is not None:
            self.embed.weight.data.copy_(torch.from_numpy(self.we_parameter))
        else:
            self.embed.weight.data.uniform_(-0.1, 0.1)


    def forward(self, text, *args):
        # Embed word ids to vectors
        cap_wids, cap_bows, lengths, cap_mask = text

        # Level 1. Global Encoding by Mean Pooling According
        org_out = cap_bows

        # Level 2. Temporal-Aware Encoding by biGRU

        # cap_wids = self.embed(cap_wids)
        # sorted_seq_lengths, indices = torch.sort(lengths, descending=True)
        # _, desorted_indices = torch.sort(indices, descending=False)
        # cap_wids = cap_wids[indices]
        # packed = pack_padded_sequence(cap_wids, sorted_seq_lengths.detach().cpu().numpy(), batch_first=True)
        # gru_init_out, _ = self.rnn(packed)
        # # Reshape *final* output to (batch_size, hidden_size)
        # padded, _ = pad_packed_sequence(gru_init_out, batch_first=True)
        # gru_init_out = padded[desorted_indices]

        cap_wids = self.embed(cap_wids)
        packed = pack_padded_sequence(cap_wids, lengths.cpu(), batch_first=True)
        gru_init_out, _ = self.rnn(packed)
        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(gru_init_out, batch_first=True)
        gru_init_out = padded[0]
    
        if self.gru_pool == 'mean':
            gru_out = Variable(torch.zeros(padded[0].size(0), self.rnn_output_size)).cuda()
            for i, batch in enumerate(padded[0]):
                gru_out[i] = torch.mean(batch[:int(lengths[i])], 0)
        elif self.gru_pool == 'max':
            gru_out = torch.max(torch.mul(gru_init_out, cap_mask.unsqueeze(-1)), 1)[0]
        gru_out = self.dropout(gru_out)

        # Level 3. Local-Enhanced Encoding by biGRU-CNN
        con_out = gru_init_out.unsqueeze(1)
        con_out = [F.relu(conv(con_out)).squeeze(3) for conv in self.convs1]
        con_out = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in con_out]
        con_out = torch.cat(con_out, 1)
        con_out = self.dropout(con_out)

        # concatenation
        if self.concate == 'full': # level 1+2+3
            features = torch.cat((gru_out,con_out,org_out), 1)
        elif self.concate == 'reduced': # level 2+3
            features = torch.cat((gru_out,con_out), 1)

        return features


class Text_multilevel_encoding_with_support(nn.Module):
    """
    Section 3.2. Text-side Multi-level Encoding
    """
    def __init__(self, opt):
        super(Text_multilevel_encoding_with_support, self).__init__()
        self.word_dim = opt.word_dim
        self.we_parameter = opt.we_parameter
        self.rnn_output_size = opt.text_rnn_size*2
        self.dropout = nn.Dropout(p=opt.dropout)
        self.concate = opt.concate
        self.gru_pool = opt.gru_pool
        self.loss_fun = opt.loss_fun
        self.style = opt.style
        
        # visual bidirectional rnn encoder
        self.embed = nn.Embedding(opt.vocab_size, opt.word_dim)
        self.rnn = nn.GRU(opt.word_dim, opt.text_rnn_size, batch_first=True, bidirectional=True)

        # visual 1-d convolutional network
        self.convs1 = nn.ModuleList([
                nn.Conv2d(1, opt.text_kernel_num, (window_size, self.rnn_output_size), padding=(window_size - 1, 0)) 
                for window_size in opt.text_kernel_sizes
                ])

        self.k = nn.Linear(opt.text_mapping_layers[0], opt.hidden_size, bias=True) 
        self.q = nn.Linear(opt.text_mapping_layers[0], opt.hidden_size, bias=True)

        self.init_weights()

    def init_weights(self):
        if self.word_dim == 500 and self.we_parameter is not None:
            self.embed.weight.data.copy_(torch.from_numpy(self.we_parameter))
        else:
            self.embed.weight.data.uniform_(-0.1, 0.1)


    def forward(self, text, support_text, *args):
        # Embed word ids to vectors
        cap_wids, cap_bows, lengths, cap_mask = text
        feature = self.encode_text(cap_wids, cap_bows, lengths, cap_mask)
        if support_text == None:
            return feature
        s_cap_wids, s_cap_bows, s_lengths, s_cap_mask = support_text
        s_feature = []
        for i in range(s_cap_wids.size(1)):
            s_feature.append(self.encode_text(s_cap_wids[:,i,:], s_cap_bows[:,i,:], s_lengths[:,i], s_cap_mask[:,i,:]))
        s_feature = torch.stack(s_feature,1)
        key = self.k(s_feature)
        query = self.q(feature)
        w = F.softmax(torch.bmm(key,query.unsqueeze(2)),dim=1)
        w = w.repeat(1,1,s_feature.shape[2])
        gated_feature = torch.sum(w * s_feature,dim=1)
        return feature + gated_feature

    def encode_text(self, cap_wids, cap_bows, lengths, cap_mask):

        # Level 1. Global Encoding by Mean Pooling According
        org_out = cap_bows

        # Level 2. Temporal-Aware Encoding by biGRU
        cap_wids = self.embed(cap_wids)
        sorted_seq_lengths, indices = torch.sort(lengths, descending=True)
        _, desorted_indices = torch.sort(indices, descending=False)
        cap_wids = cap_wids[indices]
        packed = pack_padded_sequence(cap_wids, sorted_seq_lengths.detach().cpu().numpy(), batch_first=True)
        gru_init_out, _ = self.rnn(packed)
        # Reshape *final* output to (batch_size, hidden_size)
        padded, _ = pad_packed_sequence(gru_init_out, batch_first=True)
        gru_init_out = padded[desorted_indices]
    
        if self.gru_pool == 'mean':
            gru_out = Variable(torch.zeros(padded.size(0), self.rnn_output_size)).cuda()
            for i, batch in enumerate(padded):
                gru_out[i] = torch.mean(batch[:int(lengths[i])], 0)
        elif self.gru_pool == 'max':
            gru_out = torch.max(torch.mul(gru_init_out, cap_mask.unsqueeze(-1)), 1)[0]
        gru_out = self.dropout(gru_out)

        # Level 3. Local-Enhanced Encoding by biGRU-CNN
        con_out = gru_init_out.unsqueeze(1)
        con_out = [F.relu(conv(con_out)).squeeze(3) for conv in self.convs1]
        con_out = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in con_out]
        con_out = torch.cat(con_out, 1)
        con_out = self.dropout(con_out)

        # concatenation
        if self.concate == 'full': # level 1+2+3
            features = torch.cat((gru_out,con_out,org_out), 1)
        elif self.concate == 'reduced': # level 2+3
            features = torch.cat((gru_out,con_out), 1)

        return features


class Latent_mapping(nn.Module):
    """
    Latent space mapping (Conference version)
    """
    def __init__(self, mapping_layers, dropout, l2norm=True):
        super(Latent_mapping, self).__init__()
        
        self.l2norm = l2norm
        # visual mapping
        self.mapping = MFC(mapping_layers, dropout, have_bn=True, have_last_bn=True)


    def forward(self, features):

        # mapping to latent space
        latent_features = self.mapping(features)
        if self.l2norm:
            latent_features = l2norm(latent_features)

        return latent_features



class BaseModel(object):

    def state_dict(self):
        state_dict = [None for i in range(12)]
        if hasattr(self,'teacher_T_text_encoding'):
            state_dict[0] = self.teacher_T_text_encoding.state_dict()
        if hasattr(self,'teacher_T_text_mapping'):
            state_dict[1] = self.teacher_T_text_mapping.state_dict()
        if hasattr(self,'teacher_T_vid_encoding'):
            state_dict[2] = self.teacher_T_vid_encoding.state_dict()
        if hasattr(self,'teacher_T_vid_mapping'):
            state_dict[3] = self.teacher_T_vid_mapping.state_dict()
        if hasattr(self,'teacher_V_text_encoding'):
            state_dict[4] = self.teacher_V_text_encoding.state_dict()
        if hasattr(self,'teacher_V_text_mapping'):
            state_dict[5] = self.teacher_V_text_mapping.state_dict()
        if hasattr(self,'teacher_V_vid_encoding'):
            state_dict[6] = self.teacher_V_vid_encoding.state_dict()
        if hasattr(self,'teacher_V_vid_mapping'):
            state_dict[7] = self.teacher_V_vid_mapping.state_dict()
        if hasattr(self,'student_text_encoding'):
            state_dict[8] = self.student_text_encoding.state_dict()
        if hasattr(self,'student_text_mapping'):
            state_dict[9] = self.student_text_mapping.state_dict()
        if hasattr(self,'student_vid_encoding'):
            state_dict[10] = self.student_vid_encoding.state_dict()
        if hasattr(self,'student_vid_mapping'):
            state_dict[11] = self.student_vid_mapping.state_dict()

        return state_dict

    def load_state_dict(self, state_dict):
        if hasattr(self,'teacher_T_text_encoding'):
            self.teacher_T_text_encoding.load_state_dict(state_dict[0])
        if hasattr(self,'teacher_T_text_mapping'):
            self.teacher_T_text_mapping.load_state_dict(state_dict[1])
        if hasattr(self,'teacher_T_vid_encoding'):
            self.teacher_T_vid_encoding.load_state_dict(state_dict[2])
        if hasattr(self,'teacher_T_vid_mapping'):
            self.teacher_T_vid_mapping.load_state_dict(state_dict[3])
        if hasattr(self,'teacher_V_text_encoding'):
            self.teacher_V_text_encoding.load_state_dict(state_dict[4])
        if hasattr(self,'teacher_V_text_mapping'):
            self.teacher_V_text_mapping.load_state_dict(state_dict[5])
        if hasattr(self,'teacher_V_vid_encoding'):
            self.teacher_V_vid_encoding.load_state_dict(state_dict[6])
        if hasattr(self,'teacher_V_vid_mapping'):
            self.teacher_V_vid_mapping.load_state_dict(state_dict[7])
        if hasattr(self,'student_text_encoding'):
            self.student_text_encoding.load_state_dict(state_dict[8])
        if hasattr(self,'student_text_mapping'):
            self.student_text_mapping.load_state_dict(state_dict[9])
        if hasattr(self,'student_vid_encoding'):
            self.student_vid_encoding.load_state_dict(state_dict[10])
        if hasattr(self,'student_vid_mapping'):
            self.student_vid_mapping.load_state_dict(state_dict[11])
        
    def state_dict_teacher_T(self):
        state_dict = [None for i in range(4)]
        if hasattr(self,'teacher_T_text_encoding'):
            state_dict[0] = self.teacher_T_text_encoding.state_dict()
        if hasattr(self,'teacher_T_text_mapping'):
            state_dict[1] = self.teacher_T_text_mapping.state_dict()
        if hasattr(self,'teacher_T_vid_encoding'):
            state_dict[2] = self.teacher_T_vid_encoding.state_dict()
        if hasattr(self,'teacher_T_vid_mapping'):
            state_dict[3] = self.teacher_T_vid_mapping.state_dict()

        return state_dict

    def load_state_dict_teacher_T(self, state_dict):
        if hasattr(self,'teacher_T_text_encoding'):
            self.teacher_T_text_encoding.load_state_dict(state_dict[0])
        if hasattr(self,'teacher_T_text_mapping'):
            self.teacher_T_text_mapping.load_state_dict(state_dict[1])
        if hasattr(self,'teacher_T_vid_encoding'):
            self.teacher_T_vid_encoding.load_state_dict(state_dict[2])
        if hasattr(self,'teacher_T_vid_mapping'):
            self.teacher_T_vid_mapping.load_state_dict(state_dict[3])

    def state_dict_teacher_V(self):
        state_dict = [None for i in range(4)]
        if hasattr(self,'teacher_V_text_encoding'):
            state_dict[0] = self.teacher_V_text_encoding.state_dict()
        if hasattr(self,'teacher_V_text_mapping'):
            state_dict[1] = self.teacher_V_text_mapping.state_dict()
        if hasattr(self,'teacher_V_vid_encoding'):
            state_dict[2] = self.teacher_V_vid_encoding.state_dict()
        if hasattr(self,'teacher_V_vid_mapping'):
            state_dict[3] = self.teacher_V_vid_mapping.state_dict()

        return state_dict

    def load_state_dict_teacher_V(self, state_dict):
        if hasattr(self,'teacher_V_text_encoding'):
            self.teacher_V_text_encoding.load_state_dict(state_dict[0])
        if hasattr(self,'teacher_V_text_mapping'):
            self.teacher_V_text_mapping.load_state_dict(state_dict[1])
        if hasattr(self,'teacher_V_vid_encoding'):
            self.teacher_V_vid_encoding.load_state_dict(state_dict[2])
        if hasattr(self,'teacher_V_vid_mapping'):
            self.teacher_V_vid_mapping.load_state_dict(state_dict[3])
        
        
    def train_start(self):
        """switch to train mode
        """
        if hasattr(self,'teacher_T_text_encoding'):
            self.teacher_T_text_encoding.train()
        if hasattr(self,'teacher_T_text_mapping'):
            self.teacher_T_text_mapping.train()
        if hasattr(self,'teacher_T_vid_encoding'):
            self.teacher_T_vid_encoding.train()
        if hasattr(self,'teacher_T_vid_mapping'):
            self.teacher_T_vid_mapping.train()
        if hasattr(self,'teacher_V_text_encoding'):
            self.teacher_V_text_encoding.train()
        if hasattr(self,'teacher_V_text_mapping'):
            self.teacher_V_text_mapping.train()
        if hasattr(self,'teacher_V_vid_encoding'):
            self.teacher_V_vid_encoding.train()
        if hasattr(self,'teacher_V_vid_mapping'):
            self.teacher_V_vid_mapping.train()
        if hasattr(self,'student_text_encoding'):
            self.student_text_encoding.train()
        if hasattr(self,'student_text_mapping'):
            self.student_text_mapping.train()
        if hasattr(self,'student_vid_encoding'):
            self.student_vid_encoding.train()
        if hasattr(self,'student_vid_mapping'):
            self.student_vid_mapping.train()

        
    def val_start(self):
        """switch to evaluate mode
        """
        if hasattr(self,'teacher_T_text_encoding'):
            self.teacher_T_text_encoding.eval()
        if hasattr(self,'teacher_T_text_mapping'):
            self.teacher_T_text_mapping.eval()
        if hasattr(self,'teacher_T_vid_encoding'):
            self.teacher_T_vid_encoding.eval()
        if hasattr(self,'teacher_T_vid_mapping'):
            self.teacher_T_vid_mapping.eval()
        if hasattr(self,'teacher_V_text_encoding'):
            self.teacher_V_text_encoding.eval()
        if hasattr(self,'teacher_V_text_mapping'):
            self.teacher_V_text_mapping.eval()
        if hasattr(self,'teacher_V_vid_encoding'):
            self.teacher_V_vid_encoding.eval()
        if hasattr(self,'teacher_V_vid_mapping'):
            self.teacher_V_vid_mapping.eval()
        if hasattr(self,'student_text_encoding'):
            self.student_text_encoding.eval()
        if hasattr(self,'student_text_mapping'):
            self.student_text_mapping.eval()
        if hasattr(self,'student_vid_encoding'):
            self.student_vid_encoding.eval()
        if hasattr(self,'student_vid_mapping'):
            self.student_vid_mapping.eval()


    def init_info(self):

        # init gpu
        if torch.cuda.is_available():
            if hasattr(self,'teacher_T_text_encoding'):
                self.teacher_T_text_encoding.cuda()
            if hasattr(self,'teacher_T_text_mapping'):
                self.teacher_T_text_mapping.cuda()
            if hasattr(self,'teacher_T_vid_encoding'):
                self.teacher_T_vid_encoding.cuda()
            if hasattr(self,'teacher_T_vid_mapping'):
                self.teacher_T_vid_mapping.cuda()
            if hasattr(self,'teacher_V_text_encoding'):
                self.teacher_V_text_encoding.cuda()
            if hasattr(self,'teacher_V_text_mapping'):
                self.teacher_V_text_mapping.cuda()
            if hasattr(self,'teacher_V_vid_encoding'):
                self.teacher_V_vid_encoding.cuda()
            if hasattr(self,'teacher_V_vid_mapping'):
                self.teacher_V_vid_mapping.cuda()
            if hasattr(self,'student_text_encoding'):
                self.student_text_encoding.cuda()
            if hasattr(self,'student_text_mapping'):
                self.student_text_mapping.cuda()
            if hasattr(self,'student_vid_encoding'):
                self.student_vid_encoding.cuda()
            if hasattr(self,'student_vid_mapping'):
                self.student_vid_mapping.cuda()
            cudnn.benchmark = True

        # init params
        params = []
        if hasattr(self,'teacher_T_text_encoding'):
            params += list(self.teacher_T_text_encoding.parameters())
        if hasattr(self,'teacher_T_text_mapping'):
            params += list(self.teacher_T_text_mapping.parameters())
        if hasattr(self,'teacher_T_vid_encoding'):
            params += list(self.teacher_T_vid_encoding.parameters())
        if hasattr(self,'teacher_T_vid_mapping'):
            params += list(self.teacher_T_vid_mapping.parameters())
        if hasattr(self,'teacher_V_text_encoding'):
            params += list(self.teacher_V_text_encoding.parameters())
        if hasattr(self,'teacher_V_text_mapping'):
            params += list(self.teacher_V_text_mapping.parameters())
        if hasattr(self,'teacher_V_vid_encoding'):
            params += list(self.teacher_V_vid_encoding.parameters())
        if hasattr(self,'teacher_V_vid_mapping'):
            params += list(self.teacher_V_vid_mapping.parameters())
        if hasattr(self,'student_text_encoding'):
            params += list(self.student_text_encoding.parameters())
        if hasattr(self,'student_text_mapping'):
            params += list(self.student_text_mapping.parameters())
        if hasattr(self,'student_vid_encoding'):
            params += list(self.student_vid_encoding.parameters())
        if hasattr(self,'student_vid_mapping'):
            params += list(self.student_vid_mapping.parameters())
        self.params = params
        
        # print structure
        if hasattr(self,'teacher_T_text_encoding'):
            print(self.teacher_T_text_encoding)
        if hasattr(self,'teacher_T_text_mapping'):
            print(self.teacher_T_text_mapping)
        if hasattr(self,'teacher_T_vid_encoding'):
            print(self.teacher_T_vid_encoding)
        if hasattr(self,'teacher_T_vid_mapping'):
            print(self.teacher_T_vid_mapping)
        if hasattr(self,'teacher_V_text_encoding'):
            print(self.teacher_V_text_encoding)
        if hasattr(self,'teacher_V_text_mapping'):
            print(self.teacher_V_text_mapping)
        if hasattr(self,'teacher_V_vid_encoding'):
            print(self.teacher_V_vid_encoding)
        if hasattr(self,'teacher_V_vid_mapping'):
            print(self.teacher_V_vid_mapping)
        if hasattr(self,'student_text_encoding'):
            print(self.student_text_encoding)
        if hasattr(self,'student_text_mapping'):
            print(self.student_text_mapping)
        if hasattr(self,'student_vid_encoding'):
            print(self.student_vid_encoding)
        if hasattr(self,'student_vid_mapping'):
            print(self.student_vid_mapping)


class Dual_Encoding(BaseModel):
    """
    dual encoding network
    """
    def __init__(self, opt):
        # Build Models

        self.style = opt.style
        self.associate_modal = opt.associate_modal

        # Model for inference
        self.student_text_encoding = Text_multilevel_encoding(opt)
        self.student_text_mapping = Latent_mapping(opt.text_mapping_layers, opt.dropout, opt.tag_vocab_size)
        self.student_vid_encoding = Video_multilevel_encoding(opt)
        self.student_vid_mapping = Latent_mapping(opt.visual_mapping_layers, opt.dropout, opt.tag_vocab_size)

        if self.style == 'distill':
            # if self.associate_modal == 'T' or self.associate_modal == 'V+T':
            if 'T' in self.associate_modal:
                self.teacher_T_text_encoding = Text_multilevel_encoding_with_support(opt)
                self.teacher_T_text_mapping = Latent_mapping(opt.text_mapping_layers, opt.dropout, opt.tag_vocab_size)
                self.teacher_T_vid_encoding = Video_multilevel_encoding(opt)
                self.teacher_T_vid_mapping = Latent_mapping(opt.visual_mapping_layers, opt.dropout, opt.tag_vocab_size)
            # if self.associate_modal == 'V' or self.associate_modal == 'V+T':
            if 'V' in self.associate_modal:
                self.teacher_V_text_encoding = Text_multilevel_encoding(opt)
                self.teacher_V_text_mapping = Latent_mapping(opt.text_mapping_layers, opt.dropout, opt.tag_vocab_size)
                self.teacher_V_vid_encoding = Video_multilevel_encoding_with_support(opt)
                self.teacher_V_vid_mapping = Latent_mapping(opt.visual_mapping_layers, opt.dropout, opt.tag_vocab_size)
            self.alpha_V = opt.alpha_V
            self.alpha_T = opt.alpha_T
            self.alpha_V_T = opt.alpha_V_T
            self.beta_V = opt.beta_V
            self.beta_T = opt.beta_T
            self.beta_V_T = opt.beta_V_T
            self.gamma_V = opt.gamma_V
            self.gamma_T = opt.gamma_T

            self.distill_type = opt.distill_type
            self.similarity_type = opt.similarity_type
            self.mseloss = nn.MSELoss(reduce=False)
            self.huberloss = nn.SmoothL1Loss(reduce=False)
            if opt.cost_style == 'sum':
                self.distill_criterion = nn.MSELoss(reduce=True, size_average=False)
                self.distill_kl = nn.KLDivLoss(reduce=True, size_average=False)
                self.similarity_loss = nn.SmoothL1Loss(reduce=True, size_average=False)
            elif opt.cost_style == 'mean':
                self.distill_criterion = nn.MSELoss(reduce=True, size_average=True)
                self.distill_kl = nn.KLDivLoss(reduce=True, size_average=True)
                self.similarity_loss = nn.SmoothL1Loss(reduce=True, size_average=True)  


        self.grad_clip = opt.grad_clip

        # Loss and Optimizer
        if opt.loss_fun == 'mrl':
            self.criterion = TripletLoss(margin=opt.margin,
                                            measure=opt.measure,
                                            max_violation=opt.max_violation,
                                            cost_style=opt.cost_style,
                                            direction=opt.direction)

        self.init_info()

        if opt.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.params, lr=opt.learning_rate)
        elif opt.optimizer == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(self.params, lr=opt.learning_rate)

        self.Eiters = 0


    def forward_emb(self, videos, support_videos, caption_id, data_ids, targets, support_text, volatile=False, *args):
        """Compute the video and caption embeddings
        """
        # video data
        frames, mean_origin, video_lengths, videos_mask = videos
        frames = Variable(frames, volatile=volatile)
        if torch.cuda.is_available():
            frames = frames.cuda()

        mean_origin = Variable(mean_origin, volatile=volatile)
        if torch.cuda.is_available():
            mean_origin = mean_origin.cuda()

        videos_mask = Variable(videos_mask, volatile=volatile)
        if torch.cuda.is_available():
            videos_mask = videos_mask.cuda()
        videos_data = (frames, mean_origin, video_lengths, videos_mask)

        # support video data
        if support_videos != None:
            support_frames, support_mean_origin, support_video_lengths, support_videos_mask = support_videos
            support_frames = Variable(support_frames, volatile=volatile)
            if torch.cuda.is_available():
                support_frames = support_frames.cuda()

            support_mean_origin = Variable(support_mean_origin, volatile=volatile)
            if torch.cuda.is_available():
                support_mean_origin = support_mean_origin.cuda()
            
            support_video_lengths = torch.Tensor(support_video_lengths)
            support_video_lengths = Variable(support_video_lengths, volatile=volatile)
            if torch.cuda.is_available():
                support_video_lengths = support_video_lengths.cuda()

            support_videos_mask = Variable(support_videos_mask, volatile=volatile)
            if torch.cuda.is_available():
                support_videos_mask = support_videos_mask.cuda()
            support_videos_data = (support_frames, support_mean_origin, support_video_lengths, support_videos_mask)

        # text data
        captions, cap_bows, lengths, cap_masks = targets
        if captions is not None:
            captions = Variable(captions, volatile=volatile)
            if torch.cuda.is_available():
                captions = captions.cuda()

        if cap_bows is not None:
            cap_bows = Variable(cap_bows, volatile=volatile)
            if torch.cuda.is_available():
                cap_bows = cap_bows.cuda()

        if lengths is not None:
            lengths = torch.Tensor(lengths)
            lengths = Variable(lengths, volatile=volatile)
            if torch.cuda.is_available():
                lengths = lengths.cuda()

        if cap_masks is not None:
            cap_masks = Variable(cap_masks, volatile=volatile)
            if torch.cuda.is_available():
                cap_masks = cap_masks.cuda()
        text_data = (captions, cap_bows, lengths, cap_masks)

        # support text data
        if support_text != None:
            s_captions, s_cap_bows, s_lengths, s_cap_masks = support_text
            if s_captions is not None:
                s_captions = Variable(s_captions, volatile=volatile)
                if torch.cuda.is_available():
                    s_captions = s_captions.cuda()

            if s_cap_bows is not None:
                s_cap_bows = Variable(s_cap_bows, volatile=volatile)
                if torch.cuda.is_available():
                    s_cap_bows = s_cap_bows.cuda()

            if s_lengths is not None:
                s_lengths = torch.Tensor(s_lengths)
                s_lengths = Variable(s_lengths, volatile=volatile)
                if torch.cuda.is_available():
                    s_lengths = s_lengths.cuda()

            if s_cap_masks is not None:
                s_cap_masks = Variable(s_cap_masks, volatile=volatile)
                if torch.cuda.is_available():
                    s_cap_masks = s_cap_masks.cuda()
            support_text_data = (s_captions, s_cap_bows, s_lengths, s_cap_masks)

        if self.style == 'distill':
            # if self.associate_modal == 'T' or self.associate_modal == 'V+T':
            if 'T' in self.associate_modal:
                teacher_T_vid_emb = self.teacher_T_vid_mapping(self.teacher_T_vid_encoding(videos_data))
                teacher_T_cap_emb = self.teacher_T_text_mapping(self.teacher_T_text_encoding(text_data, support_text_data))
            else:
                teacher_T_vid_emb = None
                teacher_T_cap_emb = None
            # if self.associate_modal == 'V' or self.associate_modal == 'V+T':
            if 'V' in self.associate_modal:
                teacher_V_vid_emb = self.teacher_V_vid_mapping(self.teacher_V_vid_encoding(videos_data, support_videos_data, caption_id, data_ids))
                teacher_V_cap_emb = self.teacher_V_text_mapping(self.teacher_V_text_encoding(text_data))
            else:
                teacher_V_vid_emb = None
                teacher_V_cap_emb = None
        else:
            teacher_T_vid_emb = None
            teacher_T_cap_emb = None
            teacher_V_vid_emb = None
            teacher_V_cap_emb = None

        student_vid_emb = self.student_vid_mapping(self.student_vid_encoding(videos_data))
        student_cap_emb = self.student_text_mapping(self.student_text_encoding(text_data))
        
        return teacher_T_vid_emb, teacher_T_cap_emb, teacher_V_vid_emb, teacher_V_cap_emb, student_vid_emb, student_cap_emb

    def embed_vis(self, vis_data, volatile=True):
        """Compute the video embeddings
        """
        # video data
        frames, mean_origin, video_lengths, videos_mask = vis_data
        frames = Variable(frames, volatile=volatile)
        if torch.cuda.is_available():
            frames = frames.cuda()

        mean_origin = Variable(mean_origin, volatile=volatile)
        if torch.cuda.is_available():
            mean_origin = mean_origin.cuda()

        videos_mask = Variable(videos_mask, volatile=volatile)
        if torch.cuda.is_available():
            videos_mask = videos_mask.cuda()
        vis_data = (frames, mean_origin, video_lengths, videos_mask)

        return self.student_vid_mapping(self.student_vid_encoding(vis_data))

    def embed_txt(self, txt_data, volatile=True):
        """Compute the caption embeddings
        """
        # text data
        captions, cap_bows, lengths, cap_masks = txt_data
        if captions is not None:
            captions = Variable(captions, volatile=volatile)
            if torch.cuda.is_available():
                captions = captions.cuda()

        if cap_bows is not None:
            cap_bows = Variable(cap_bows, volatile=volatile)
            if torch.cuda.is_available():
                cap_bows = cap_bows.cuda()

        if lengths is not None:
            lengths = torch.Tensor(lengths)
            lengths = Variable(lengths, volatile=volatile)
            if torch.cuda.is_available():
                lengths = lengths.cuda() 

        if cap_masks is not None:
            cap_masks = Variable(cap_masks, volatile=volatile)
            if torch.cuda.is_available():
                cap_masks = cap_masks.cuda()
        txt_data = (captions, cap_bows, lengths, cap_masks)

        return self.student_text_mapping(self.student_text_encoding(txt_data))

    def embed_vis_teacher_T(self, vis_data, volatile=True):
        """Compute the video embeddings
        """
        # video data
        frames, mean_origin, video_lengths, videos_mask = vis_data
        frames = Variable(frames, volatile=volatile)
        if torch.cuda.is_available():
            frames = frames.cuda()

        mean_origin = Variable(mean_origin, volatile=volatile)
        if torch.cuda.is_available():
            mean_origin = mean_origin.cuda()

        videos_mask = Variable(videos_mask, volatile=volatile)
        if torch.cuda.is_available():
            videos_mask = videos_mask.cuda()
        vis_data = (frames, mean_origin, video_lengths, videos_mask)

        return self.teacher_T_vid_mapping(self.teacher_T_vid_encoding(vis_data))

    def embed_txt_teacher_T(self, txt_data, support_txt_data, volatile=True):
        """Compute the caption embeddings
        """
        # text data
        captions, cap_bows, lengths, cap_masks = txt_data
        if captions is not None:
            captions = Variable(captions, volatile=volatile)
            if torch.cuda.is_available():
                captions = captions.cuda()

        if cap_bows is not None:
            cap_bows = Variable(cap_bows, volatile=volatile)
            if torch.cuda.is_available():
                cap_bows = cap_bows.cuda()

        if lengths is not None:
            lengths = torch.Tensor(lengths)
            lengths = Variable(lengths, volatile=volatile)
            if torch.cuda.is_available():
                lengths = lengths.cuda() 

        if cap_masks is not None:
            cap_masks = Variable(cap_masks, volatile=volatile)
            if torch.cuda.is_available():
                cap_masks = cap_masks.cuda()
        txt_data = (captions, cap_bows, lengths, cap_masks)

        # support text data
        if support_txt_data != None:
            s_captions, s_cap_bows, s_lengths, s_cap_masks = support_txt_data
            if s_captions is not None:
                s_captions = Variable(s_captions, volatile=volatile)
                if torch.cuda.is_available():
                    s_captions = s_captions.cuda()

            if s_cap_bows is not None:
                s_cap_bows = Variable(s_cap_bows, volatile=volatile)
                if torch.cuda.is_available():
                    s_cap_bows = s_cap_bows.cuda()

            if s_lengths is not None:
                s_lengths = torch.Tensor(s_lengths)
                s_lengths = Variable(s_lengths, volatile=volatile)
                if torch.cuda.is_available():
                    s_lengths = s_lengths.cuda()

            if s_cap_masks is not None:
                s_cap_masks = Variable(s_cap_masks, volatile=volatile)
                if torch.cuda.is_available():
                    s_cap_masks = s_cap_masks.cuda()
            support_text_data = (s_captions, s_cap_bows, s_lengths, s_cap_masks)

        return self.teacher_T_text_mapping(self.teacher_T_text_encoding(txt_data, support_text_data))

    def embed_txt_teacher_V(self, txt_data, volatile=True):
        """Compute the caption embeddings
        """
        # text data
        captions, cap_bows, lengths, cap_masks = txt_data
        if captions is not None:
            captions = Variable(captions, volatile=volatile)
            if torch.cuda.is_available():
                captions = captions.cuda()

        if cap_bows is not None:
            cap_bows = Variable(cap_bows, volatile=volatile)
            if torch.cuda.is_available():
                cap_bows = cap_bows.cuda()

        if lengths is not None:
            lengths = torch.Tensor(lengths)
            lengths = Variable(lengths, volatile=volatile)
            if torch.cuda.is_available():
                lengths = lengths.cuda() 

        if cap_masks is not None:
            cap_masks = Variable(cap_masks, volatile=volatile)
            if torch.cuda.is_available():
                cap_masks = cap_masks.cuda()
        txt_data = (captions, cap_bows, lengths, cap_masks)

        return self.teacher_V_text_mapping(self.teacher_V_text_encoding(txt_data))

    def embed_vis_teacher_V(self, vis_data, support_vis_data, data_ids, volatile=True):
        """Compute the video embeddings
        """
        # video data
        frames, mean_origin, video_lengths, videos_mask = vis_data
        frames = Variable(frames, volatile=volatile)
        if torch.cuda.is_available():
            frames = frames.cuda()

        mean_origin = Variable(mean_origin, volatile=volatile)
        if torch.cuda.is_available():
            mean_origin = mean_origin.cuda()

        videos_mask = Variable(videos_mask, volatile=volatile)
        if torch.cuda.is_available():
            videos_mask = videos_mask.cuda()
        vis_data = (frames, mean_origin, video_lengths, videos_mask)

        # support video data
        if support_vis_data != None:
            support_frames, support_mean_origin, support_video_lengths, support_videos_mask = support_vis_data
            support_frames = Variable(support_frames, volatile=volatile)
            if torch.cuda.is_available():
                support_frames = support_frames.cuda()

            support_mean_origin = Variable(support_mean_origin, volatile=volatile)
            if torch.cuda.is_available():
                support_mean_origin = support_mean_origin.cuda()
            
            support_video_lengths = torch.Tensor(support_video_lengths)
            support_video_lengths = Variable(support_video_lengths, volatile=volatile)
            if torch.cuda.is_available():
                support_video_lengths = support_video_lengths.cuda()

            support_videos_mask = Variable(support_videos_mask, volatile=volatile)
            if torch.cuda.is_available():
                support_videos_mask = support_videos_mask.cuda()
            support_videos_data = (support_frames, support_mean_origin, support_video_lengths, support_videos_mask)

        return self.teacher_V_vid_mapping(self.teacher_V_vid_encoding(vis_data, support_videos_data, data_ids))


    def forward_loss_distill_similarity(self, vid_emb, cap_emb, student_vid_emb, student_cap_emb, *args, **kwargs):
        s1 = vid_emb.mm(cap_emb.t())
        s2 = student_vid_emb.mm(student_cap_emb.t())
        if self.similarity_type=='svd':
            a,b,c = torch.svd(s1)
            s1 = torch.matmul(a,torch.matmul(torch.diag(torch.log(b)),c))
            a,b,c = torch.svd(s2)
            s2 = torch.matmul(a,torch.matmul(torch.diag(torch.log(b)),c))
            loss = self.similarity_loss(s1, s2)
        elif self.similarity_type == 'eig':
            a,b = torch.eig(s1,eigenvectors=True)
            s1 = torch.matmul(b,torch.matmul(torch.diag(a[:,0]),torch.inverse(b)))
            a,b = torch.eig(s2,eigenvectors=True)
            s2 = torch.matmul(b,torch.matmul(torch.diag(a[:,0]),torch.inverse(b)))
            loss = self.similarity_loss(s1, s2)
        elif self.similarity_type == 'diag':
            loss = torch.sum(torch.diagonal(self.huberloss(s1,s2)))
        elif self.similarity_type == 'maxdiag':
            loss = -torch.sum(torch.diagonal(s2))
        else:
            loss = self.similarity_loss(s1, s2)
        self.logger.update('similarity_distillation_loss', loss.item(), cap_emb.size(0))
        return loss

    def forward_loss_distill(self, student_cap_emb, cap_emb, *agrs, **kwargs):
        if self.distill_type == 'mse':
            loss = self.distill_criterion(student_cap_emb, cap_emb)
        elif self.distill_type == 'kl':
            loss = self.distill_kl(student_cap_emb, cap_emb)
        elif self.distill_type == 'mse+kl':
            loss = self.distill_criterion(student_cap_emb, cap_emb) + self.distill_kl(student_cap_emb, cap_emb)
        self.logger.update('distillationloss', loss.item(), cap_emb.size(0))

        return loss


    def forward_loss(self, cap_emb, vid_emb, *agrs, **kwargs):
        """Compute the loss given pairs of video and caption embeddings
        """
        loss = self.criterion(cap_emb, vid_emb)
        self.logger.update('Le', loss.item(), vid_emb.size(0))
        # self.logger.update('Le', loss.data[0], vid_emb.size(0))

        return loss


    def train_emb(self, i, epoch, opt, videos, support_videos, captions, support_captions, *args):
        """One training step given videos and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        teacher_T_vid_emb, teacher_T_cap_emb, teacher_V_vid_emb, teacher_V_cap_emb, student_vid_emb, student_cap_emb = self.forward_emb(videos, support_videos, args[2], args[3], captions, support_captions)
        loss_student = self.forward_loss(student_cap_emb, student_vid_emb)
        if self.style == 'distill':
            if 'T' in self.associate_modal:
                loss_T_triplet = self.forward_loss(teacher_T_cap_emb, teacher_T_vid_emb)
                loss_T_cap = self.forward_loss_distill(student_cap_emb, teacher_T_cap_emb.detach())
                loss_T_vid = self.forward_loss_distill(student_vid_emb, teacher_T_vid_emb.detach())
                loss_T_sim = self.forward_loss_distill_similarity(teacher_T_vid_emb.detach(), teacher_T_cap_emb.detach(), student_vid_emb, student_cap_emb)
            if 'V' in self.associate_modal:
                loss_V_triplet = self.forward_loss(teacher_V_cap_emb, teacher_V_vid_emb)
                loss_V_cap = self.forward_loss_distill(student_cap_emb, teacher_V_cap_emb.detach())
                loss_V_vid = self.forward_loss_distill(student_vid_emb, teacher_V_vid_emb.detach())
                loss_V_sim = self.forward_loss_distill_similarity(teacher_V_vid_emb.detach(), teacher_V_cap_emb.detach(), student_vid_emb, student_cap_emb)
            if self.associate_modal == 'V':
                if not opt.stop_teacher_V:
                    loss = self.gamma_V * loss_V_triplet
                else:
                    loss = loss_student + self.alpha_V * (loss_V_cap + loss_V_vid) + self.beta_V * loss_V_sim
            elif self.associate_modal == 'T':
                if not opt.stop_teacher_T:
                    loss = self.gamma_T * loss_T_triplet
                else:
                    loss = loss_student + self.alpha_T * (loss_T_cap + loss_T_vid) + self.beta_T * loss_T_sim
            elif self.associate_modal == 'V+T':
                if not opt.stop_teacher_V and opt.stop_teacher_T:
                    loss = self.gamma_V * loss_V_triplet
                elif not opt.stop_teacher_T and opt.stop_teacher_V:
                    loss = self.gamma_T * loss_T_triplet
                elif not opt.stop_teacher_V and not opt.stop_teacher_T:
                    loss = self.gamma_V * loss_V_triplet + self.gamma_T * loss_T_triplet
                else:
                    loss = loss_student + self.alpha_V * (loss_V_cap + loss_V_vid) + self.beta_V * loss_V_sim + self.alpha_T * (loss_T_cap + loss_T_vid) + self.beta_T * loss_T_sim
            elif self.associate_modal == 'VT':
                loss_V_T_cap = self.forward_loss_distill(teacher_V_cap_emb, teacher_T_cap_emb)
                loss_V_T_vid = self.forward_loss_distill(teacher_V_vid_emb, teacher_T_vid_emb)
                loss_V_T_sim = self.forward_loss_distill_similarity(teacher_T_vid_emb, teacher_T_cap_emb, teacher_V_vid_emb, teacher_V_cap_emb)
                if not opt.stop_teacher_V and not opt.stop_teacher_T:
                    loss = self.gamma_V * loss_V_triplet + self.gamma_T * loss_T_triplet + self.alpha_V_T * (loss_V_T_cap + loss_V_T_vid) + self.beta_V_T * loss_V_T_sim
                else:
                    loss = loss_student + self.alpha_V * (loss_V_cap + loss_V_vid) + self.beta_V * loss_V_sim + self.alpha_T * (loss_T_cap + loss_T_vid) + self.beta_T * loss_T_sim
        else:
            loss = loss_student

        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()

        if self.style == 'distill':
            if self.associate_modal == 'T':
                return student_vid_emb.size(0), loss_student.item(), loss_T_triplet.item(), loss_T_cap.item(), loss_T_vid.item(), loss_T_sim.item()
            elif self.associate_modal == 'V':
                return student_vid_emb.size(0), loss_student.item(), loss_V_triplet.item(), loss_V_cap.item(), loss_V_vid.item(), loss_V_sim.item()
            elif self.associate_modal == 'V+T':
                return student_vid_emb.size(0), loss_student.item(), loss_V_triplet.item(), loss_V_cap.item(), loss_V_vid.item(), loss_V_sim.item(), \
                                                                     loss_T_triplet.item(), loss_T_cap.item(), loss_T_vid.item(), loss_T_sim.item() 
            elif self.associate_modal == 'VT':
                return student_vid_emb.size(0), loss_student.item(), loss_V_triplet.item(), loss_V_cap.item(), loss_V_vid.item(), loss_V_sim.item(), \
                                                                     loss_T_triplet.item(), loss_T_cap.item(), loss_T_vid.item(), loss_T_sim.item(), \
                                                                     loss_V_T_cap.item(), loss_V_T_vid.item(), loss_V_T_sim.item()
        else:
            return student_vid_emb.size(0), loss_student.item()


NAME_TO_MODELS = {'dual_encoding_latent': Dual_Encoding}

def get_model(name):
    assert name in NAME_TO_MODELS, '%s not supported.'%name
    return NAME_TO_MODELS[name]
