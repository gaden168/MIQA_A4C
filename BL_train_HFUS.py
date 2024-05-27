from  Utr_dataset import Utrl_dataset
from model import Utr_model
import torch
from torch import nn
import tqdm
import os
import  itertools
import time
from sklearn.metrics import classification_report,f1_score

import splitfolders


import warnings
warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
os.environ['TZ'] = 'UTC-8'
time.tzset()

def make_dataset(root):
    train_dl = Utrl_dataset(root+'/train','train', batch_size=64, num_workers=0, shuffle=True)
    test_dl = Utrl_dataset(root+'/val','test', batch_size=64, num_workers=0, shuffle=False)
    return train_dl, test_dl
def y_sort(model_result, scores):
    tem_list = list(itertools.combinations(scores, 2))
    model_result_list = list(itertools.combinations(model_result, 2))
    x1_list = []
    x2_list = []
    y_list =[]
    for i,item in enumerate(tem_list):
        a,b = item
        if a > b or a == b:
            y = 1
        else:
            y =-1
        c,d = model_result_list[i]
        x1_list.append(c)
        x2_list.append(d)
        y_list.append(y)
    x1 = torch.tensor([item.cpu().detach().numpy() for item in x1_list]).cuda()
    x2 = torch.tensor([item.cpu().detach().numpy() for item in x2_list]).cuda()
    Y = torch.tensor(y_list).unsqueeze(-1).cuda()
        # print(Y.shape)
    return x1,x2,Y

if __name__ == '__main__':

    # in_path = r'/root/workspace/echo_project/PublicData/HFUS/Dataset/'
    # out_path = r'/root/workspace/echo_project/PublicData/HFUS/data/'
    # splitfolders.ratio(in_path, out_path, seed=1337, ratio=(0.6, 0.2, 0.2))

    root = r'/root/workspace/echo_project/PublicData/HFUS/data/'
    #
    # # 进行数据的加载
    train_dl, test_dl = make_dataset(root)
    #
    # # 进行模型的加载
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    model = Utr_model(numclass=4).to(device)
    #
    # 定义相关的训练参数
    loss_fn = nn.CrossEntropyLoss()
    loss_rank = nn.MarginRankingLoss()

    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=opt, milestones=[25, 50, 75], gamma=0.1)
    epochs = 200

    for epoch in range(epochs):
        # 开始进行训练
        train_tqdm = tqdm.tqdm(iterable=train_dl, total=len(train_dl))
        train_tqdm.set_description_str('Train_Epoch {:3d}'.format(epoch))
        model.train()
        loss2 = 0
        for image, label in train_tqdm:
            image, label = image.to(device), label.to(device)
            pred = model(image)
            loss1 = loss_fn(pred, label)
            x1, x2, y_ = y_sort(pred, label)
            loss2 += loss_rank(x1, x2, y_)
            loss = loss1 + loss2
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                train_tqdm.set_postfix_str('Train_Loss is {:.14f}'.format(loss_fn(pred, label).item()))
        train_tqdm.close()
        # 开始进行测试
        with torch.no_grad():
            test_tqdm = tqdm.tqdm(iterable=test_dl, total=len(test_dl))
            test_tqdm.set_description_str('Test_Epoch {:3d}'.format(epoch))
            model.eval()
            correct = 0
            total = 0
            prob_all = []
            label_all = []
            for image, label in test_tqdm:
                image, label = image.to(device), label.to(device)
                pred = model(image)
                loss = loss_fn(pred, label)
                test_tqdm.set_postfix_str('Test_Loss is {:.14f}'.format(loss.item()))
                pred = torch.argmax(input=pred, dim=1)
                total += label.size(0)
                correct += (pred == label).sum().item()
                prob_all.extend(pred.cpu().numpy())  # 求每一行的最大值索引
                label_all.extend(label.cpu().numpy())
            test_tqdm.close()
            print('Accuracy of the network : %d %%' % (100 * correct / total))
            print("F1-Score:{:.4f}".format(f1_score(label_all, prob_all, average='macro')))
        if epoch > 20 and epoch % 20 == 0:
            print(classification_report(label_all, prob_all, labels=[0, 1, 2, 3]))

            print('Accuracy of the network : %d %%' % (100 * correct / total))
        # 进行动态学习率的调整
        scheduler.step()

    # 进行模型的保存
    if not os.path.exists('model_data'):
        os.mkdir('model_data')
    torch.save(model.state_dict(), r'model_data\HFUS_model.pth')

    # plt.figure(figsize=(10, 3))
    # for i, img in enumerate(images[:10]):
    #     np_img = img.numpy()
    #     np_img = np.squeeze(np_img)
    #     plt.subplot(1, 10, i + 1)
    #     plt.imshow(np_img)
    #     plt.axis('off')
    #     plt.title(str(label[i].numpy()))
    # plt.show()

