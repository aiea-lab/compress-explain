import torch
import numpy as np

def train_step(data_loader, model_list, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), logging=False, epoch=0):
    dataset_size = len(data_loader.dataset)
    tot_batches = dataset_size / data_loader.batch_size
    for model in model_list:
         model.accuracy = []
         model.loss = []
         model.batch_size = []

    for b, (X,y) in enumerate(data_loader):
        batch_size = len(y)
        y = y.to(device)
        X = X.to(device)
        log_batch = logging and (b+1)%(tot_batches//10)==0

        for model in model_list:
            y_pred=model(X)
            loss=model.criterion(y_pred,y)
            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()

            accuracy = (torch.max(y_pred.data,1)[1] == y).sum()
            model.accuracy.append(accuracy)
            model.loss.append(loss.item())
            model.batch_size.append(batch_size)

        if log_batch:
                print(f'epoch: {epoch:2}  batch: {b+1:4} [{tot_batches:6}]')
                for model in model_list:
                    batch_size = np.array(model.batch_size)[-tot_batches//10:]
                    loss = np.sum(np.array(model.loss[-tot_batches//10:]) * batch_size) / np.sum(batch_size)
                    accuracy = np.sum(np.array(model.accuracy[-tot_batches//10:]) * batch_size) / np.sum(batch_size)
                    print(f'model: {model:15}  loss: {loss:10.8f}  accuracy: {accuracy:7.3f}%')

    batch_size = np.array(model.batch_size)
    loss = np.array(model.loss)
    accuracy = np.array(model.accuracy)
    model.train_accuracy.append(np.sum(accuracy * batch_size) / dataset_size)
    model.train_loss.append(np.sum(loss * batch_size) / dataset_size)



def eval(data_loader, model_list, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), logging=True):
    for model in model_list:
         model.accuracy = []
         model.loss = []
         model.batch_size = []
    with torch.no_grad():
        for b, (X,y) in enumerate(data_loader):
            batch_size = len(y)
            y = y.to(device)
            X = X.to(device)
            for model in model_list:
                y_pred = model(X)
                loss = model.criterion(y_pred,y)
                accuracy = (torch.max(y_pred.data,1)[1] == y).sum()
                model.accuracy.append(accuracy)
                model.loss.append(loss.item())
                model.batch_size.append(batch_size)
    
    dataset_size = len(data_loader.dataset)
    for model in model_list:
        batch_size = np.array(model.batch_size)
        model.test_accuracy.append(np.sum(np.array(model.accuracy) * batch_size) / dataset_size)
        model.test_loss.append(np.sum(np.array(model.loss) * batch_size) / dataset_size)
    if logging:
        for model in model_list:
            print(f'model: {model:15}  loss: {model.loss:10.8f}  accuracy: {model.accuracy:7.3f}%')


