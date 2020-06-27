# coding: utf-8

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score

import numpy as np

from models.util import tensorized


class ModelHandler(nn.Module):
    def __init__(self, params):
        super(ModelHandler, self).__init__()
        #         self.bestStateDict = None
        self.epochNums = params['epoch_nums']
        self.batch_size = params['batch_size']
        self.device = params['device']
        self.word2id = params['word2id']
        self.tag2id = params['tag2id']
        self.model = params['model'].to(self.device)

    def forword(self, features, lengths):
        return self.model(features, lengths)

    #     def reset(self, m):
    #         if hasattr(m, 'reset_parameters'):
    #             torch.cuda.manual_seed(1)
    #             m.reset_parameters()

    def fit(self, X_train, y_train, loss_fn, optimizer, model_path=None, eval_set=None, early_stopping_rounds=None,
            valBatchSize=None, verbose=0, temperature=1):

        if eval_set != None:
            X_val, y_val = eval_set
        if early_stopping_rounds != None:
            # 用来计数多少epoch在验证集上的结果没有改进了
            count = 0
        #         self.apply(self.reset)

        batch_size = self.batch_size
        trainDataNum = len(X_train)
        batchNumInEveryEpoch = trainDataNum // batch_size
        epochNums = self.epochNums
        best_val_acc = -1000000
        best_val_loss = 1000000
        if valBatchSize != None:
            valBatchSize = valBatchSize
        else:
            valBatchSize = batch_size
        num = 0
        for epoch in range(epochNums):
            print('epoch:', epoch)
            # 设置成 training 模式
            self.train()
            # 设置自动微分
            torch.set_grad_enabled(True)

            trainAcc = 0.0
            trainLoss = 0.0
            for t1 in range(batchNumInEveryEpoch):
                X_train_var = X_train[t1 * batch_size:(t1 + 1) * batch_size]
                y_train_var = y_train[t1 * batch_size:(t1 + 1) * batch_size]
                X_train_var, lengths = tensorized(X_train_var, self.word2id)
                y_train_var, _ = tensorized(y_train_var, self.tag2id)
                X_train_var = X_train_var.to(self.device)
                y_train_var = y_train_var.to(self.device)

                self.zero_grad()
                scores = self.forword(X_train_var, lengths)
                loss = loss_fn(scores, y_train_var, self.tag2id).to(self.device)
                # trainAcc = trainAcc + self.getAUC(y_train_var, torch.sigmoid(scores).squeeze())
                trainLoss = trainLoss + loss.item()
                self.train()
                torch.set_grad_enabled(True)
                loss.backward()
                optimizer.step()
            if verbose == 2:
                # print('train acc:', trainAcc / float(batchNumInEveryEpoch))
                print('train loss:', trainLoss / float(batchNumInEveryEpoch))

            val_loss = self.check_accuracy(X_val, y_val, valBatchSize, loss_fn, True, verbose)
            if verbose == 2:
                # print('val_acc:', val_acc)
                print('val_loss:', val_loss)
            #     print ('val_acc:', val_acc, file=file, flush=True)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                bestEpoch = epoch
                count = 0
                if model_path != None:
                    torch.save(self.state_dict(), model_path)
            elif early_stopping_rounds != None:
                count += 1
                if count >= early_stopping_rounds:
                    if verbose >= 1:
                        print('Stopping.')
                        print('Best Epoch:', bestEpoch)
                        print('Best Val Loss:', best_val_loss)
                    break

    def check_accuracy(self, X_val, y_val, valBatchSize, loss_fn, isTrain, verbose, temperature=1):
        if verbose == 2:
            if isTrain:
                print('*****Checking accuracy on validation set*****')
            #         print('Checking accuracy on validation set', file=file, flush=True)
            else:
                print('Checking accuracy on test set')
                #         print('Checking accuracy on test set', file=file, flush=True)
        # 将模型设置成evaluation模式
        self.eval()
        torch.set_grad_enabled(False)
        batchNum = len(X_val) // valBatchSize
        if isTrain != True and X_val.shape[0] % valBatchSize != 0:
            batchNum += 1
        if verbose == 2:
            print('batchNum:', batchNum)
        valLoss = 0.0
        for t1 in range(batchNum):

            if isTrain != True and t1 == batchNum - 1:
                tX_val_var = X_val[t1 * valBatchSize:]
                tY_val_var = y_val[t1 * valBatchSize:]
                tX_val_var, lengths = tensorized(tX_val_var, self.word2id)
                tY_val_var, _ = tensorized(tY_val_var, self.tag2id)
            else:
                tX_val_var = X_val[t1 * valBatchSize:(t1 + 1) * valBatchSize]
                tY_val_var = y_val[t1 * valBatchSize:(t1 + 1) * valBatchSize]
                tX_val_var, lengths = tensorized(tX_val_var, self.word2id)
                tY_val_var, _ = tensorized(tY_val_var, self.tag2id)
            tX_val_var = tX_val_var.to(self.device)
            tY_val_var = tY_val_var.to(self.device)

            scores = self.forword(tX_val_var, lengths)

            if isTrain == True:
                loss = loss_fn(scores, tY_val_var, self.tag2id).to(self.device)
                # valAcc += self.getAUC(tY_val_var / temperature, torch.sigmoid(scores).squeeze())
                valLoss = valLoss + loss.item()

        if isTrain == True:
            return valLoss / float(batchNum)

    def test(self, word_lists, tag_lists, indices):
        """返回最佳模型在测试集上的预测结果"""
        # 准备数据
        # word_lists, tag_lists, indices = sort_by_lengths(word_lists, tag_lists)
        tensorized_sents, lengths = tensorized(word_lists, self.word2id)
        tensorized_sents = tensorized_sents.to(self.device)

        self.model.eval()
        with torch.no_grad():
            batch_tagids = self.model.test(
                tensorized_sents, lengths, self.tag2id)

        # 将id转化为标注
        pred_tag_lists = []
        id2tag = dict((id_, tag) for tag, id_ in self.tag2id.items())
        for i, ids in enumerate(batch_tagids):
            tag_list = []
            for j in range(lengths[i] - 1):  # crf解码过程中，end被舍弃
                tag_list.append(id2tag[ids[j].item()])
            pred_tag_lists.append(tag_list)

        # indices存有根据长度排序后的索引映射的信息
        # 比如若indices = [1, 2, 0] 则说明原先索引为1的元素映射到的新的索引是0，
        # 索引为2的元素映射到新的索引是1...
        # 下面根据indices将pred_tag_lists和tag_lists转化为原来的顺序
        ind_maps = sorted(list(enumerate(indices)), key=lambda e: e[1])
        indices, _ = list(zip(*ind_maps))
        pred_tag_lists = [pred_tag_lists[i] for i in indices]
        tag_lists = [tag_lists[i] for i in indices]

        return pred_tag_lists, tag_lists
