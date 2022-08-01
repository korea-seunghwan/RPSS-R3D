TRAIN_CSV_PATH = '/utils/data/train.csv'
TEST_CSV_PATH = '/utils/data/test.csv'

PROJECT_NAME = 'pretrained_r3d101'
EXPERIMENT_NAME = 'rps'
SAVE_MODEL_NAME = 'r3d'
WRITE_LOG = True


from utils.RPSDataset import RPSDataset

import torch
import os
import torchvision
import random
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from barbar import Bar
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn
import network.resNet3D as resNet3D

import datetime

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

random_seed = 42
np.random.seed(random_seed)
random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # multi-GPU
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = False

print("Current CPU random seed", torch.initial_seed())
print("Current CUDA random seed", torch.cuda.initial_seed())


BASE_SAVE_PATH = os.path.join('.' ,PROJECT_NAME, 'work_dir', EXPERIMENT_NAME)
PLT_SAVE_PATH = os.path.join('.', PROJECT_NAME, 'results', EXPERIMENT_NAME)

if not os.path.exists(BASE_SAVE_PATH):
    os.makedirs(BASE_SAVE_PATH)
if not os.path.exists(PLT_SAVE_PATH):
    os.makedirs(PLT_SAVE_PATH)

if WRITE_LOG:
    LOG_TXT = open(os.path.join(BASE_SAVE_PATH, 'result_log.txt'), 'w')

    LOG_TRUNK_SCORE = open(os.path.join(BASE_SAVE_PATH, 'trunk_score_log.txt'), 'w')
    LOG_MOVEMENT_SCORE = open(os.path.join(BASE_SAVE_PATH, 'movement_score_log.txt'), 'w')
    LOG_ELBOW_SCORE = open(os.path.join(BASE_SAVE_PATH, 'elbow_score_log.txt'), 'w')
    LOG_SHOULDER_SCORE = open(os.path.join(BASE_SAVE_PATH, 'shoulder_score_log.txt'), 'w')
    LOG_PREHENSION_SCORE = open(os.path.join(BASE_SAVE_PATH, 'prehension_score_log.txt'), 'w')
    LOG_GLOBAL_SCORE = open(os.path.join(BASE_SAVE_PATH, 'global_score_log.txt'), 'w')

    LOG_PRED_RESULT = open(os.path.join(BASE_SAVE_PATH, 'pred_log.txt'), 'w')
    LOG_TRUE_RESULT = open(os.path.join(BASE_SAVE_PATH, 'true_log.txt'), 'w')
    LOG_FILE_NAMES = open(os.path.join(BASE_SAVE_PATH, 'file_name_log.txt'), 'w')

    CF_TRUNK = 'cf_trunk.png'
    CF_MOVEMENT = 'cf_movement.png'
    CF_ELBOW = 'cf_elbow.png'
    CF_SHOULDER = 'cf_shoulder.png'
    CF_PREHENSION = 'cf_prehension.png'
    CF_GLOBAL = 'cf_global.png'

########################################################################################################

# test for VideoLabelDataset
train_dataset = RPSDataset(
    TRAIN_CSV_PATH, 
    transform=torchvision.transforms.Compose([
        transforms.VideoRandomCrop([224, 224]),
    ])   
)

test_dataset = RPSDataset(TEST_CSV_PATH)

print('train_dataset length : ', len(train_dataset))
print('test_dataset length : ', len(test_dataset))

trainset_num = len(train_dataset)
valset_num = len(test_dataset)

##############################################################################################
# def append_checkpoint(model, optimizer):
#     state = {
#         'state_dict': model.module.state_dict(),
#         'optimizer': optimizer.state_dict()
#     }

#     checkpoints.append(state)

# def save_checkpoint(state):
#     if os.path.exists(os.path.join('.', PROJECT_NAME, 'checkpoint', EXPERIMENT_NAME)):
#         filepath = os.path.join('.', PROJECT_NAME, 'checkpoint', EXPERIMENT_NAME, '{}.pth'.format(SAVE_MODEL_NAME))
#     else:
#         os.makedirs(os.path.join('.', PROJECT_NAME, 'checkpoint', EXPERIMENT_NAME))
#         filepath = os.path.join('.', PROJECT_NAME, 'checkpoint', EXPERIMENT_NAME, '{}.pth'.format(SAVE_MODEL_NAME))

#     torch.save(state, filepath)

def printLog(val_category, val_score, val_correct, val_total):
    print("############################################################")
    for i in range(len(val_correct)):
        print('{} {} score : {:.4f} correct : {}/{}'.format(val_category, i, val_score[i], val_correct[i], val_total[i]))

def writeLog(category, LOG_FILE, score, correct, total):
    for i in range(len(score)):
        LOG_FILE.write('{} {} score : {:.4f} correct : {}/{}\n'.format(category, i, score[i], correct[i], total[i]))

def pltConfusionMatrix(conf_matrix, classes, PLT_SAVE_PATH, PLT_SAVE_NAME, PLT_TITLE):
    data_df = pd.DataFrame(conf_matrix, index = [i for i in classes], columns = [i for i in classes])
    plt.clf()
    plt.figure(figsize=(12,7))
    plt.title(PLT_TITLE, fontsize=18)
    sn.heatmap(data_df, annot=True)
    plt.savefig(os.path.join(PLT_SAVE_PATH, PLT_SAVE_NAME))
###############################################################################################

#########################################################################################
# train_set, val_set = torch.utils.data.random_split(dataset, [trainset_num, valset_num])

# #########################################################################################
# from torch.utils.data import Subset
# train_set = Subset(train_set, np.arange(200))
# print('trainset: ', len(train_set))
#########################################################################################

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=8)
val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=8)

####################### model #######################
model = resNet3D.generate_model(101)
model.fc = torch.nn.Linear(2048, 700)
pretrain = torch.load('/pretrained_model/r3d101_K_200ep.pth')
model.load_state_dict(pretrain["state_dict"])
model.fc = torch.nn.Sequential(model.fc, torch.nn.Linear(700, 24))# representation
model = torch.nn.DataParallel(model)
model = model.to(device)
#####################################################

best_accuracy = 0.0

epochs = 200
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay = 0.0001)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.0001)
criterion = nn.CrossEntropyLoss()

running_loss_history = []

running_trunk_corrects_history = []
running_movement_corrects_history = []
running_shoulder_corrects_history = []
running_elbow_corrects_history = []
running_global_corrects_history = []
running_prehension_corrects_history = []

val_running_trunk_corrects_history = []
val_running_movement_corrects_history = []
val_running_shoulder_corrects_history = []
val_running_elbow_corrects_history = []
val_running_global_corrects_history = []
val_running_prehension_corrects_history = []

confusion_trunk_pred_all = []
confusion_trunk_true_all = []
confusion_movement_pred_all = []
confusion_movement_true_all = []
confusion_elbow_pred_all = []
confusion_elbow_true_all = []
confusion_shoulder_pred_all = []
confusion_shoulder_true_all = []
confusion_prehension_pred_all = []
confusion_prehension_true_all = []
confusion_global_pred_all = []
confusion_global_true_all = []

file_names = []

############ 시간 측정 ############
import time
start_time = time.time()
##################################

for e in range(epochs):
    model.train()
    # if (e + 1) % 30 == 0:  # 21.06.11
    #     optimizer.param_groups[0]['lr'] /= 10
    print("##### {} epoch starting #####".format(e))
    running_loss = 0.0

    trunk_running_corrects = 0.0
    movement_running_corrects = 0.0
    elbow_running_corrects = 0.0
    shoulder_running_corrects = 0.0
    global_running_corrects = 0.0
    prehension_running_corrects = 0.0

    val_trunk_running_corrects = 0.0
    val_movement_running_corrects = 0.0
    val_elbow_running_corrects = 0.0
    val_shoulder_running_corrects = 0.0
    val_global_running_corrects = 0.0
    val_prehension_running_corrects = 0.0

    ###################################################################
    val_trunk_correct = [0, 0, 0, 0]
    val_trunk_total = [0, 0, 0, 0]

    val_movement_correct = [0, 0, 0, 0]
    val_movement_total = [0, 0, 0, 0]

    val_elbow_correct = [0, 0, 0, 0]
    val_elbow_total = [0, 0, 0, 0]

    val_shoulder_correct = [0, 0, 0, 0]
    val_shoulder_total = [0, 0, 0, 0]

    val_prehension_correct = [0, 0, 0, 0]
    val_prehension_total = [0, 0, 0, 0]

    val_global_correct = [0, 0, 0, 0]
    val_global_total = [0, 0, 0, 0]
    ###################################################################

    for _, video, trunk, movement, shoulder, elbow, prehension, global_s in Bar(train_loader):
        # joint = joint.float().to(device)
        video = video.float().to(device)
        # optical = optical.float().to(device)
        trunk = trunk.long().to(device)
        movement = movement.long().to(device)
        shoulder = shoulder.long().to(device)
        prehension = prehension.long().to(device)
        elbow = elbow.long().to(device)
        global_s = global_s.long().to(device)

        output = model.forward(video)

        out_trunk = (output[:, 0:4]).to(device)
        out_movement = (output[:, 4:8]).to(device)
        out_shoulder = (output[:, 8:12]).to(device)
        out_elbow = (output[:, 12:16]).to(device)
        out_prehension = (output[:, 16:20]).to(device)
        out_global = (output[:, 20:24]).to(device)

        loss_trunk = criterion(out_trunk, trunk)
        loss_smoothness = criterion(out_movement, movement)
        loss_elbow = criterion(out_elbow, elbow)
        loss_shoulder = criterion(out_shoulder, shoulder)
        loss_prehension = criterion(out_prehension, prehension)
        loss_global = criterion(out_global, global_s)

        loss = loss_trunk + loss_smoothness + loss_shoulder + loss_elbow + loss_prehension + loss_global

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()

        running_loss += loss.item()

        _, trunk_preds = torch.max(out_trunk, 1)
        _, movement_preds = torch.max(out_movement, 1)
        _, elbow_preds = torch.max(out_elbow, 1)
        _, shoulder_preds = torch.max(out_shoulder, 1)
        _, prehension_preds = torch.max(out_prehension, 1)
        _, global_preds = torch.max(out_global, 1)

        trunk_running_corrects += torch.sum(trunk_preds == trunk.data)
        movement_running_corrects += torch.sum(movement_preds == movement.data)
        elbow_running_corrects += torch.sum(elbow_preds == elbow.data)
        shoulder_running_corrects += torch.sum(shoulder_preds == shoulder.data)
        prehension_running_corrects += torch.sum(prehension_preds == prehension.data)
        global_running_corrects += torch.sum(global_preds == global_s.data)
        # prehension_running_corrects += torch.sum(prehension_preds == prehension.data)

    print('end {} epoch for loop start validation'.format(e))
    # print('trunk running corrects: ', trunk_running_corrects)

    epoch_loss = running_loss / len(train_loader)

    trunk_epoch_acc = trunk_running_corrects.float() / trainset_num
    movement_epoch_acc = movement_running_corrects.float() / trainset_num
    elbow_epoch_acc = elbow_running_corrects.float() / trainset_num
    shoulder_epoch_acc = shoulder_running_corrects.float() / trainset_num
    prehension_epoch_acc = prehension_running_corrects.float() / trainset_num
    global_epoch_acc = global_running_corrects.float() / trainset_num

    running_loss_history.append(epoch_loss)

    print('epoch : ', e)
    print('training loss : {:.4f}'.format(epoch_loss))

    print('train trunk acc : {:.4f}'.format(trunk_epoch_acc))
    print('train movement acc : {:.4f}'.format(movement_epoch_acc))
    print('train elbow acc : {:.4f}'.format(elbow_epoch_acc))
    print('train shoulder acc : {:.4f}'.format(shoulder_epoch_acc))
    print('train prehension acc : {:.4f}'.format(prehension_epoch_acc))
    print('train global acc : {:.4f}'.format(global_epoch_acc))
    # print('train prehension acc : {:.4f}'.format(prehension_epoch_acc))

    running_trunk_corrects_history.append(trunk_epoch_acc)
    running_movement_corrects_history.append(movement_epoch_acc)
    running_elbow_corrects_history.append(elbow_epoch_acc)
    running_shoulder_corrects_history.append(shoulder_epoch_acc)
    running_prehension_corrects_history.append(prehension_epoch_acc)
    running_global_corrects_history.append(global_epoch_acc)
    # running_prehension_corrects_history.append(prehension_epoch_acc)

    if (e + 1) % 5 == 0:
        model.eval()
        with torch.no_grad():
            confusion_trunk_pred = []
            confusion_trunk_true = []
            confusion_movement_pred = []
            confusion_movement_true = []
            confusion_elbow_pred = []
            confusion_elbow_true = []
            confusion_shoulder_pred = []
            confusion_shoulder_true = []
            confusion_prehension_pred = []
            confusion_prehension_true = []
            confusion_global_pred = []
            confusion_global_true = []

            epoch_file_names = []
            for file_name, video, trunk, movement, shoulder, elbow, prehension, global_s in Bar(val_loader):
                # val_joint = joint.float().to(device)
                val_video = video.float().to(device)
                # val_optical = optical.float().to(device)
                val_trunk = trunk.long().to(device)
                val_movement = movement.long().to(device)
                val_shoulder = shoulder.long().to(device)
                val_elbow = elbow.long().to(device)
                val_prehension = prehension.long().to(device)
                val_global_s = global_s.long().to(device)

                val_output = model(val_video)


                val_out_trunk = (val_output[:, 0:4]).to(device)
                val_out_movement = (val_output[:, 4:8]).to(device)
                val_out_shoulder = (val_output[:, 8:12]).to(device)
                val_out_elbow = (val_output[:, 12:16]).to(device)
                val_out_prehension = (val_output[:, 16:20]).to(device)
                val_out_global = (val_output[:, 20:24]).to(device)

                _, val_trunk_preds = torch.max(val_out_trunk, 1)
                _, val_movement_preds = torch.max(val_out_movement, 1)
                _, val_elbow_preds = torch.max(val_out_elbow, 1)
                _, val_shoulder_preds = torch.max(val_out_shoulder, 1)
                _, val_prehension_preds = torch.max(val_out_prehension, 1)
                _, val_global_preds = torch.max(val_out_global, 1)
                # _, prehension_preds = torch.max(out_prehension, 1)

                ############### confusion matrix ###############
                confusion_trunk_pred.extend(val_trunk_preds.clone().detach().cpu().numpy())
                confusion_movement_pred.extend(val_movement_preds.clone().detach().cpu().numpy())
                confusion_elbow_pred.extend(val_elbow_preds.clone().detach().cpu().numpy())
                confusion_shoulder_pred.extend(val_shoulder_preds.clone().detach().cpu().numpy())
                confusion_prehension_pred.extend(val_prehension_preds.clone().detach().cpu().numpy())
                confusion_global_pred.extend(val_global_preds.clone().detach().cpu().numpy())
                ############################################################################
                confusion_trunk_true.extend(val_trunk.clone().detach().cpu().numpy())
                confusion_movement_true.extend(val_movement.clone().detach().cpu().numpy())
                confusion_elbow_true.extend(val_elbow.clone().detach().cpu().numpy())
                confusion_shoulder_true.extend(val_shoulder.clone().detach().cpu().numpy())
                confusion_prehension_true.extend(val_prehension.clone().detach().cpu().numpy())
                confusion_global_true.extend(val_global_s.clone().detach().cpu().numpy())

                epoch_file_names.extend(file_name)
                ############################################################################
                for i in range(len(val_trunk)):
                    val_trunk_total[val_trunk[i]] += 1
                    if val_trunk[i] == val_trunk_preds[i]:
                        val_trunk_correct[val_trunk[i]] += 1

                for i in range(len(val_movement)):
                    val_movement_total[val_movement[i]] += 1
                    if val_movement[i] == val_movement_preds[i]:
                        val_movement_correct[val_movement[i]] += 1

                for i in range(len(val_elbow)):
                    val_elbow_total[val_elbow[i]] += 1
                    if val_elbow[i] == val_elbow_preds[i]:
                        val_elbow_correct[val_elbow[i]] += 1

                for i in range(len(val_shoulder)):
                    val_shoulder_total[val_shoulder[i]] += 1
                    if val_shoulder[i] == val_shoulder_preds[i]:
                        val_shoulder_correct[val_shoulder[i]] += 1

                for i in range(len(val_prehension)):
                    val_prehension_total[val_prehension[i]] += 1
                    if val_prehension[i] == val_prehension_preds[i]:
                        val_prehension_correct[val_prehension[i]] += 1

                for i in range(len(val_global_s)):
                    val_global_total[val_global_s[i]] += 1
                    if val_global_s[i] == val_global_preds[i]:
                        val_global_correct[val_global_s[i]] += 1

                ############################################################################

                # print('torch sum : ', torch.sum(trunk_preds == trunk.data))
                ############################################################################
                val_trunk_running_corrects += torch.sum(val_trunk_preds == val_trunk.data).item()
                val_movement_running_corrects += torch.sum(val_movement_preds == val_movement.data).item()
                val_elbow_running_corrects += torch.sum(val_elbow_preds == val_elbow.data).item()
                val_shoulder_running_corrects += torch.sum(val_shoulder_preds == val_shoulder.data).item()
                val_prehension_running_corrects += torch.sum(val_prehension_preds == val_prehension.data).item()
                val_global_running_corrects += torch.sum(val_global_preds == val_global_s.data).item()
                # val_prehension_running_corrects += torch.sum(prehension_preds == val_prehension.data)

        val_trunk_epoch_acc = val_trunk_running_corrects / valset_num
        val_movement_epoch_acc = val_movement_running_corrects / valset_num
        val_elbow_epoch_acc = val_elbow_running_corrects / valset_num
        val_shoulder_epoch_acc = val_shoulder_running_corrects / valset_num
        val_prehension_epoch_acc = val_prehension_running_corrects / valset_num
        val_global_epoch_acc = val_global_running_corrects / valset_num
        # val_prehension_epoch_acc = val_prehension_running_corrects.float()/valset_num

        ############################################### save checkpoint ######################################################################
        
        val_avg_epoch_acc = (val_trunk_epoch_acc + val_movement_epoch_acc + val_elbow_epoch_acc + val_shoulder_epoch_acc + val_prehension_epoch_acc + val_global_epoch_acc) / 6.0

        if val_avg_epoch_acc > best_accuracy:
            best_accuracy = val_avg_epoch_acc
            # model.state_dict()
            if os.path.exists(os.path.join('.', PROJECT_NAME, 'checkpoint', EXPERIMENT_NAME)):
                filepath = os.path.join('.', PROJECT_NAME, 'checkpoint', EXPERIMENT_NAME, '{}.pth'.format(SAVE_MODEL_NAME))
            else:
                os.makedirs(os.path.join('.', PROJECT_NAME, 'checkpoint', EXPERIMENT_NAME))
                filepath = os.path.join('.', PROJECT_NAME, 'checkpoint', EXPERIMENT_NAME, '{}.pth'.format(SAVE_MODEL_NAME))

            torch.save(model.module.state_dict(), filepath)

        ###############################################################################################
        val_trunk_score = [i / j for i,j in zip(val_trunk_correct, val_trunk_total)]
        val_movement_score = [i / j for i,j in zip(val_movement_correct, val_movement_total)]
        val_elbow_score = [i / j for i,j in zip(val_elbow_correct, val_elbow_total)]
        val_shoulder_score = [i / j for i,j in zip(val_shoulder_correct, val_shoulder_total)]
        val_prehension_score = [i / j for i,j in zip(val_prehension_correct, val_prehension_total)]
        val_global_score = [i / j for i,j in zip(val_global_correct, val_global_total)]
        ###############################################################################################
        
        #######################################################################################
        # print('f1_score : ', f1_score(confusion_trunk_true, confusion_trunk_pred))
        # print('accuracy_score : ', accuracy_score(confusion_trunk_true, confusion_trunk_pred))
        # print('precision_score : ', precision_score(confusion_trunk_true, confusion_trunk_pred))

        confusion_trunk_pred_all.append(confusion_trunk_pred)
        confusion_trunk_true_all.append(confusion_trunk_true)
        confusion_movement_pred_all.append(confusion_movement_pred)
        confusion_movement_true_all.append(confusion_movement_true)
        confusion_shoulder_pred_all.append(confusion_shoulder_pred)
        confusion_shoulder_true_all.append(confusion_shoulder_true)
        confusion_elbow_pred_all.append(confusion_elbow_pred)
        confusion_elbow_true_all.append(confusion_elbow_true)
        confusion_prehension_pred_all.append(confusion_prehension_pred)
        confusion_prehension_true_all.append(confusion_prehension_true)
        confusion_global_pred_all.append(confusion_global_pred)
        confusion_global_true_all.append(confusion_global_true)

        file_names.append(epoch_file_names)
        #######################################################################################

        print("############################################################")
        print('test trunk acc : {:.4f}'.format(val_trunk_epoch_acc))
        print('test movement acc : {:.4f}'.format(val_movement_epoch_acc))
        print('test elbow acc : {:.4f}'.format(val_elbow_epoch_acc))
        print('test shoulder acc : {:.4f}'.format(val_shoulder_epoch_acc))
        print('test prehension acc : {:.4f}'.format(val_prehension_epoch_acc))
        print('test global acc : {:.4f}'.format(val_global_epoch_acc))

        #######################################################################################
        printLog('trunk', val_trunk_score, val_trunk_correct, val_trunk_total)
        printLog('movement' ,val_movement_score, val_movement_correct, val_movement_total)
        printLog('elbow', val_elbow_score, val_elbow_correct, val_elbow_total)
        printLog('shoulder', val_shoulder_score, val_shoulder_correct, val_shoulder_total)
        printLog('prehension', val_prehension_score, val_prehension_correct, val_prehension_total)
        printLog('global', val_global_score, val_global_correct, val_global_total)
        #######################################################################################

        if WRITE_LOG:
            LOG_TXT.write('test trunk acc : {:.4f}\n'.format(val_trunk_epoch_acc))
            LOG_TXT.write('test movement acc : {:.4f}\n'.format(val_movement_epoch_acc))
            LOG_TXT.write('test elbow acc : {:.4f}\n'.format(val_elbow_epoch_acc))
            LOG_TXT.write('test shoulder acc : {:.4f}\n'.format(val_shoulder_epoch_acc))
            LOG_TXT.write('test prehension acc : {:.4f}\n'.format(val_prehension_epoch_acc))
            LOG_TXT.write('test global acc : {:.4f}\n'.format(val_global_epoch_acc))

            writeLog('trunk' ,LOG_TRUNK_SCORE, val_trunk_score, val_trunk_correct, val_trunk_total)
            writeLog('movement', LOG_MOVEMENT_SCORE, val_movement_score, val_movement_correct, val_movement_total)
            writeLog('elbow', LOG_ELBOW_SCORE, val_elbow_score, val_elbow_correct, val_elbow_total)
            writeLog('shoulder', LOG_SHOULDER_SCORE, val_shoulder_score, val_shoulder_correct, val_shoulder_total)
            writeLog('prehension', LOG_PREHENSION_SCORE, val_prehension_score, val_prehension_correct, val_prehension_total)
            writeLog('global', LOG_GLOBAL_SCORE, val_global_score, val_global_correct, val_global_total)
        print("############################################################\n")

        val_running_trunk_corrects_history.append(val_trunk_epoch_acc)
        val_running_movement_corrects_history.append(val_movement_epoch_acc)
        val_running_elbow_corrects_history.append(val_elbow_epoch_acc)
        val_running_shoulder_corrects_history.append(val_shoulder_epoch_acc)
        val_running_prehension_corrects_history.append(val_prehension_epoch_acc)
        val_running_global_corrects_history.append(val_global_epoch_acc)

######### 시간 측정 ###########
end_time = time.time()
used_time = end_time - start_time
hours = used_time // 3600
used_time -= hours * 3600
minutes = used_time // 60
seconds = used_time - minutes*60

print('학습에 소요된 시간 : {}시간 {}분 {:.4f}초'.format(int(hours), int(minutes), seconds))

val_avg_history = []

for i in range(len(val_running_trunk_corrects_history)):
    trunk_acc = float(val_running_trunk_corrects_history[i])
    movement_acc = float(val_running_movement_corrects_history[i])
    elbow_acc = float(val_running_elbow_corrects_history[i])
    shoulder_acc = float(val_running_shoulder_corrects_history[i])
    prehension_acc = float(val_running_prehension_corrects_history[i])
    global_acc = float(val_running_global_corrects_history[i])

    avg = (trunk_acc + movement_acc + elbow_acc + shoulder_acc + prehension_acc + global_acc) / 6.0
    val_avg_history.append(avg)

f = lambda i: val_avg_history[i]
max_index = max(range(len(val_avg_history)), key=f)

print('best accuracy index: ', max_index)
print('best trunk acc: ', val_running_trunk_corrects_history[max_index])
print('best movement acc: ', val_running_movement_corrects_history[max_index])
print('best elbow acc: ', val_running_elbow_corrects_history[max_index])
print('best shoulder acc: ', val_running_shoulder_corrects_history[max_index])
print('best prehension acc: ', val_running_prehension_corrects_history[max_index])
print('best global acc: ', val_running_global_corrects_history[max_index])


if WRITE_LOG:
    LOG_TXT.write('best accuracy index: {}'.format(int(max_index)))
    LOG_TXT.write('best trunk acc: {:.4f}'.format(val_running_trunk_corrects_history[max_index]))
    LOG_TXT.write('best movement acc: {:.4f}'.format(val_running_movement_corrects_history[max_index]))
    LOG_TXT.write('best elbow acc: {:.4f}'.format(val_running_elbow_corrects_history[max_index]))
    LOG_TXT.write('best shoulder acc: {:.4f}'.format(val_running_shoulder_corrects_history[max_index]))
    LOG_TXT.write('best prehension acc: {:.4f}'.format(val_running_prehension_corrects_history[max_index]))
    LOG_TXT.write('best global acc: {:.4f}'.format(val_running_global_corrects_history[max_index]))
    LOG_TXT.write('학습에 소요된 시간 : {}시간 {}분 {:.4f}초'.format(int(hours), int(minutes), seconds))
    LOG_TXT.close()
    LOG_TRUNK_SCORE.close()
    LOG_MOVEMENT_SCORE.close()
    LOG_ELBOW_SCORE.close()
    LOG_SHOULDER_SCORE.close()
    LOG_PREHENSION_SCORE.close()
    LOG_GLOBAL_SCORE.close()
# torch.save(model.state_dict(), './save_models/i3d_optical_pytorch/i3d_optical_pytorch_34_300.pt')

# plt.plot(running_loss_history, label='training_loss')
# plt.savefig(os.path.join(PLT_SAVE_PATH, 'training_loss_34_150.png'))

# plt.clf()
# plt.plot(running_trunk_corrects_history, label='training_trunk_acc')
# plt.plot(running_movement_corrects_history, label='training_movement_acc')
# plt.plot(running_elbow_corrects_history, label='training_elbow_acc')
# plt.plot(running_shoulder_corrects_history, label='training_shoulder_acc')
# plt.plot(running_prehension_corrects_history, label='training_prehension_acc')
# plt.plot(running_global_corrects_history, label='training_global_acc')
# # plt.plot(running_prehension_corrects_history, label='training_prehension_acc')
# plt.legend()
# plt.savefig(os.path.join(PLT_SAVE_PATH, 'training_accuracy_34_150.png'))

# # val_avg_corrects_history = (val_running_trunk_corrects_history + val_running_movement_corrects_history + val_running_elbow_corrects_history) // 4.0
# plt.clf()
# plt.plot(val_running_trunk_corrects_history, label='val_trunk_acc')
# plt.plot(val_running_movement_corrects_history, label='val_movement_acc')
# plt.plot(val_running_elbow_corrects_history, label='val_elbow_acc')
# plt.plot(val_running_shoulder_corrects_history, label='val_shoulder_acc')
# plt.plot(val_running_prehension_corrects_history, label='val_prehension_acc')
# plt.plot(val_running_global_corrects_history, label='val_global_acc')
# # plt.plot(val_avg_corrects_history, label='val_avg_acc')
# # plt.plot(val_running_prehension_corrects_history, label='val_prehension_acc')
# plt.legend()
# plt.savefig(os.path.join(PLT_SAVE_PATH, 'val_accuracy_34_150.png'))

print(len(confusion_trunk_pred))
print(confusion_trunk_pred)

classes = ['0', '1', '2', '3']

cf_trunk_pred_max = confusion_trunk_pred_all[max_index]
cf_trunk_true_max = confusion_trunk_true_all[max_index]
cf_movement_pred_max = confusion_movement_pred_all[max_index]
cf_movement_true_max = confusion_movement_true_all[max_index]
cf_elbow_pred_max = confusion_elbow_pred_all[max_index]
cf_elbow_true_max = confusion_elbow_true_all[max_index]
cf_shoulder_pred_max = confusion_shoulder_pred_all[max_index]
cf_shoulder_true_max = confusion_shoulder_true_all[max_index]
cf_prehension_pred_max = confusion_prehension_pred_all[max_index]
cf_prehension_true_max = confusion_prehension_true_all[max_index]
cf_global_pred_max = confusion_global_pred_all[max_index]
cf_global_true_max = confusion_global_true_all[max_index]

if WRITE_LOG:
    LOG_PRED_RESULT.write('cf_trunk_pred: {}\n'.format(cf_trunk_pred_max))
    LOG_TRUE_RESULT.write('cf_trunk_true: {}\n'.format(cf_trunk_true_max))
    LOG_PRED_RESULT.write('cf_movement_pred: {}\n'.format(cf_movement_pred_max))
    LOG_TRUE_RESULT.write('cf_movement_true: {}\n'.format(cf_movement_true_max))
    LOG_PRED_RESULT.write('cf_shoulder_pred: {}\n'.format(cf_shoulder_pred_max))
    LOG_TRUE_RESULT.write('cf_shoulder_true: {}\n'.format(cf_shoulder_true_max))
    LOG_PRED_RESULT.write('cf_elbow_pred: {}\n'.format(cf_elbow_pred_max))
    LOG_TRUE_RESULT.write('cf_elbow_true: {}\n'.format(cf_elbow_true_max))
    LOG_PRED_RESULT.write('cf_prehension_pred: {}\n'.format(cf_prehension_pred_max))
    LOG_TRUE_RESULT.write('cf_prehension_true: {}\n'.format(cf_prehension_true_max))
    LOG_PRED_RESULT.write('cf_global_pred: {}\n'.format(cf_global_pred_max))
    LOG_TRUE_RESULT.write('cf_global_true: {}\n'.format(cf_global_true_max))
    LOG_FILE_NAMES.write(str(file_names[max_index]))

    LOG_TRUE_RESULT.close()
    LOG_PRED_RESULT.close()
    LOG_FILE_NAMES.close()

    cf_trunk_matrix = confusion_matrix(cf_trunk_true_max, cf_trunk_pred_max)
    cf_movement_matrix = confusion_matrix(cf_movement_true_max, cf_movement_pred_max)
    cf_shoulder_matrix = confusion_matrix(cf_shoulder_true_max, cf_shoulder_pred_max)
    cf_elbow_matrix = confusion_matrix(cf_elbow_true_max, cf_elbow_pred_max)
    cf_prehension_matrix = confusion_matrix(cf_prehension_true_max, cf_prehension_pred_max)
    cf_global_matrix = confusion_matrix(cf_global_true_max, cf_global_pred_max)

    print(cf_trunk_matrix)

    pltConfusionMatrix(cf_trunk_matrix, classes, PLT_SAVE_PATH, CF_TRUNK, 'Trunk')
    pltConfusionMatrix(cf_movement_matrix, classes, PLT_SAVE_PATH, CF_MOVEMENT, 'Movement')
    pltConfusionMatrix(cf_elbow_matrix, classes, PLT_SAVE_PATH, CF_ELBOW, 'Elbow')
    pltConfusionMatrix(cf_shoulder_matrix, classes, PLT_SAVE_PATH, CF_SHOULDER, 'Shoulder')
    pltConfusionMatrix(cf_prehension_matrix, classes, PLT_SAVE_PATH, CF_PREHENSION, 'Prehension')
    pltConfusionMatrix(cf_global_matrix, classes, PLT_SAVE_PATH, CF_GLOBAL, 'Global')
#########################################################################################