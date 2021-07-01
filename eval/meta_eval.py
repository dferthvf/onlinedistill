from __future__ import print_function

import copy
import numpy as np
import scipy
from scipy.stats import t
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn import metrics
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * t._ppf((1+confidence)/2., n-1)
    return m, h


def normalize(x):
    norm = x.pow(2).sum(1, keepdim=True).pow(1. / 2)
    out = x.div(norm)
    return out


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def meta_test(net, testloader, use_logit=True, is_norm=True, classifier='LR', opt=None):
    net = net.eval()
    acc = []

    with torch.no_grad():
        for  data in tqdm(testloader, position=0, leave=True):
            support_xs, support_ys, query_xs, query_ys = data
            support_xs = support_xs.cuda()
            query_xs = query_xs.cuda()
            batch_size, _, channel, height, width = support_xs.size()
            support_xs = support_xs.view(-1, channel, height, width)
        
            query_xs = query_xs.view(-1, channel, height, width)

            if use_logit:
                support_features = net(support_xs).view(support_xs.size(0), -1)
                query_features = net(query_xs).view(query_xs.size(0), -1)
            else:
                feat_support, _ = net(support_xs, is_feat=True)
                feat_support_0 = feat_support[0].mean(dim=[2, 3])
                feat_support_1 = feat_support[1].mean(dim=[2, 3])
                feat_support_2 = feat_support[2].mean(dim=[2, 3])

                support_features = feat_support[-1].view(support_xs.size(0), -1)
                feat_query, _ = net(query_xs, is_feat=True)
                feat_query_0 = feat_query[0].mean(dim=[2, 3])
                feat_query_1 = feat_query[1].mean(dim=[2, 3])
                feat_query_2 = feat_query[2].mean(dim=[2, 3])
                query_features = feat_query[-1].view(query_xs.size(0), -1)

            if is_norm:
                support_features = normalize(support_features)
                query_features = normalize(query_features)

            support_features = support_features.detach().cpu().numpy()
            query_features = query_features.detach().cpu().numpy()

            support_ys = support_ys.view(-1).numpy()
            query_ys = query_ys.view(-1).numpy()

            if classifier == 'LR':
                # clf = LogisticRegression(penalty='l2',
                #                          random_state=0,
                #                          C=1.0,
                #                          solver='lbfgs',
                #                          max_iter=1000,
                #                          multi_class='multinomial')
                clf = LogisticRegression()
                clf.fit(support_features, support_ys)
                #####single pred##########
                query_ys_pred = clf.predict(query_features) 
                #####ensemble#############
                # query_ys_pred = clf.predict_proba(query_features)
                # query_ys_pred = np.reshape(query_ys_pred, (opt.n_aug_support_samples, opt.n_queries * opt.n_ways, -1))
                # query_ys_pred = query_ys_pred.mean(axis=0)
                # query_ys_pred = query_ys_pred.argmax(axis=-1) 

            elif classifier == 'MLP':
                with torch.enable_grad():
                    query_ys_pred = MLP(support_xs, support_ys, query_xs, net, use_logit=use_logit)
            elif classifier == 'MLP2':
                with torch.enable_grad():
                    query_ys_pred = MLP2(support_features, support_ys, query_features, use_logit=use_logit)
            elif classifier == 'SVM':
                clf = SVC(gamma='auto', C=1, 
                          kernel='linear', decision_function_shape='ovr')
                clf.fit(support_features, support_ys)
                query_ys_pred = clf.predict(query_features)
            elif classifier == 'NN':
                query_ys_pred = NN(support_features, support_ys, query_features)
            elif classifier == 'Cosine':
                query_ys_pred = Cosine(support_features, support_ys, query_features)
            elif classifier == 'Proto':
                query_ys_pred = Proto(support_features, support_ys, query_features, opt)
            elif classifier == 'Ensemble':
                clf = LogisticRegression(penalty='l2',
                                         random_state=0,
                                         C=1.0,
                                         solver='lbfgs',
                                         max_iter=1000,
                                         multi_class='multinomial')
                clf_90 = LogisticRegression(penalty='l2',
                                            random_state=0,
                                            C=1.0,
                                            solver='lbfgs',
                                            max_iter=1000,
                                            multi_class='multinomial')
                clf_180 = LogisticRegression(penalty='l2',
                                             random_state=0,
                                             C=1.0,
                                             solver='lbfgs',
                                             max_iter=1000,
                                             multi_class='multinomial')
                clf_270 = LogisticRegression(penalty='l2',
                                             random_state=0,
                                             C=1.0,
                                             solver='lbfgs',
                                             max_iter=1000,
                                             multi_class='multinomial')
                clf.fit(support_features, support_ys)
                clf_90.fit(support_features_90, support_ys)
                clf_180.fit(support_features_180, support_ys)
                clf_270.fit(support_features_270, support_ys)
                query_ys_pred = clf.predict_proba(query_features)
                query_ys_pred_90 = clf.predict_proba(query_features_90)
                query_ys_pred_180 = clf.predict_proba(query_features_180)
                query_ys_pred_270 = clf.predict_proba(query_features_270)

                query_ys_pred = query_ys_pred + query_ys_pred_90 + query_ys_pred_180 + query_ys_pred_270
                query_ys_pred = query_ys_pred.argmax(axis=-1)

            else:
                raise NotImplementedError('classifier not supported: {}'.format(classifier))

            acc.append(metrics.accuracy_score(query_ys, query_ys_pred))
            accuracy, std = mean_confidence_interval(acc)
            tqdm.write('acc: %s, std: %s, use_logit: %s' % (accuracy, std, use_logit))
            # print(accuracy)

    return mean_confidence_interval(acc)


def Proto(support, support_ys, query, opt):
    """Protonet classifier"""
    nc = support.shape[-1]
    support = np.reshape(support, (-1, 1, opt.n_ways, opt.n_shots, nc))
    support = support.mean(axis=3)
    batch_size = support.shape[0]
    query = np.reshape(query, (batch_size, -1, 1, nc))
    logits = - ((query - support)**2).sum(-1)
    pred = np.argmax(logits, axis=-1)
    pred = np.reshape(pred, (-1,))
    return pred


def NN(support, support_ys, query):
    """nearest classifier"""
    support = np.expand_dims(support.transpose(), 0)
    query = np.expand_dims(query, 2)

    diff = np.multiply(query - support, query - support)
    distance = diff.sum(1)
    min_idx = np.argmin(distance, axis=1)
    pred = [support_ys[idx] for idx in min_idx]
    return pred


def Cosine(support, support_ys, query):
    """Cosine classifier"""
    support_norm = np.linalg.norm(support, axis=1, keepdims=True)
    support = support / support_norm
    query_norm = np.linalg.norm(query, axis=1, keepdims=True)
    query = query / query_norm

    cosine_distance = query @ support.transpose()
    max_idx = np.argmax(cosine_distance, axis=1)
    pred = [support_ys[idx] for idx in max_idx]
    return pred


def MLP2(support_features, support_ys, query_features, use_logit=True):
    support_features = torch.from_numpy(support_features).cuda()
    query_features = torch.from_numpy(query_features).cuda()
    support_ys = torch.from_numpy(support_ys).cuda()
    mlp = nn.Sequential(
        nn.Linear(64 if use_logit else 640, 5),
        # nn.BatchNorm1d(64),
        # nn.Sigmoid(),
        # nn.Linear(64, 32),
        # nn.BatchNorm1d(32),
        # nn.ReLU(),
        # nn.Linear(64, 5),
        )
    opt = optim.SGD(mlp.parameters(), lr=0.1, weight_decay=1.0)
    mlp = mlp.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    # criterion = nn.MultiMarginLoss(p=2, margin=10.0).cuda()
    mlp.train()
    for _ in range(100):
        opt.zero_grad()
        logits = mlp(support_features)
        loss = criterion(logits, support_ys)
        loss.backward()
        opt.step()
    mlp.eval()
    pred = mlp(query_features).detach().cpu().argmax(dim=1)
    return pred


def MLP(support_xs, support_ys, query_xs, net, use_logit=True):
    support_ys = torch.from_numpy(support_ys).cuda()
    mlp = nn.Sequential(
        nn.Linear(64 if use_logit else 640, 5),)
        # nn.BatchNorm1d(128),
        # nn.ReLU(),
        # nn.Linear(128, 32),
        # nn.BatchNorm1d(32),
        # nn.ReLU(),
        # nn.Linear(32, 5))

    opt = optim.SGD(mlp.parameters(), lr=0.1, weight_decay=1.0)
    mlp = mlp.cuda()
    criterion = nn.CrossEntropyLoss()
    mlp.train()
    net = copy.deepcopy(net)
    net.train()

    batch_size = support_xs.size(0)
    for _ in range(100):
        ids = torch.randperm(batch_size).cuda()
        # support_xs_mixed, support_ys_a, support_ys_b, lam, index = cutmix_data(support_xs[ids][:25], support_ys[ids][:25])
        if use_logit:
            support_features = net(support_xs).view(support_xs.size(0), -1)
        else:
            feat_support, _ = net(support_xs, is_feat=True)
            support_features = feat_support[-1].view(support_xs.size(0), -1)
        support_features = normalize(support_features)
        opt.zero_grad()
        logits = mlp(support_features)
        loss = F.cross_entropy(logits, support_ys)
        # loss = mixup_criterion(F.cross_entropy, logits, support_ys_a, support_ys_b, lam)
        loss.backward()
        opt.step()
    mlp.eval()
    net.eval()
    if use_logit:
        query_features = net(query_xs).view(query_xs.size(0), -1)
    else:
        feat_query, _ = net(query_xs, is_feat=True)
        query_features = feat_query[-1].view(query_xs.size(0), -1)
    query_features = normalize(query_features)
    pred = mlp(query_features).detach().cpu().argmax(dim=1)
    return pred


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix_data(x, y, alpha=1.0, use_cuda=True):
    if alpha > 0 and np.random.rand(1) < 0.5:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    x_a = x
    y_a, y_b = y, y[index]
    mixed_x = x.clone()
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    mixed_x[:, :, bbx1:bbx2, bby1:bby2] = mixed_x[index, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return mixed_x, y_a, y_b, lam, index

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)