
import time
import os
import torch
from torch.optim import Adam
from torch_geometric.data import DataLoader
import numpy as np
from torch.autograd import grad
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import pandas as pd

class run():
    r"""
    The base script for running different 3DGN methods.
    """
    def __init__(self):
        pass


    def run(self, device, train_dataset, valid_dataset, test_dataset, model, loss_func, epochs=3000,
            batch_size=32, vt_batch_size=32, lr=0.001, lr_decay_factor=0.5, lr_decay_step_size=50, weight_decay=1e-4,
            save_dir='model_save/save_dir', log_dir='model_save/log_dir'):
        r"""
        The run script for training and validation.

        Args:
            device (torch.device): Device for computation.
            train_dataset: Training data.
            valid_dataset: Validation data.
            test_dataset: Test data.
            model: Which 3DGN model to use. Should be one of the SchNet, DimeNetPP, and SphereNet.
            loss_func (function): The used loss funtion for training.
            evaluation (function): The evaluation function.
            epochs (int, optinal): Number of total training epochs. (default: :obj:`500`)
            batch_size (int, optinal): Number of samples in each minibatch in training. (default: :obj:`32`)
            vt_batch_size (int, optinal): Number of samples in each minibatch in validation/testing. (default: :obj:`32`)
            lr (float, optinal): Initial learning rate. (default: :obj:`0.0005`)
            lr_decay_factor (float, optinal): Learning rate decay factor. (default: :obj:`0.5`)
            lr_decay_step_size (int, optinal): epochs at which lr_initial <- lr_initial * lr_decay_factor. (default: :obj:`50`)
            weight_decay (float, optinal): weight decay factor at the regularization term. (default: :obj:`0`)
            energy_and_force (bool, optional): If set to :obj:`True`, will predict energy and take the minus derivative of the energy with respect to the atomic positions as predicted forces. (default: :obj:`False`)
            p (int, optinal): The forces’ weight for a joint loss of forces and conserved energy during training. (default: :obj:`100`)
            save_dir (str, optinal): The path to save trained models. If set to :obj:`''`, will not save the model. (default: :obj:`''`)
            log_dir (str, optinal): The path to save log files. If set to :obj:`''`, will not save the log files. (default: :obj:`''`)

        """

        model = model.to(device)
        num_params = sum(p.numel() for p in model.parameters())
        print(f'#Params: {num_params}')
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = StepLR(optimizer, step_size=lr_decay_step_size, gamma=lr_decay_factor)

        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)

        valid_loader = DataLoader(valid_dataset, vt_batch_size, shuffle=False)

        test_loader = DataLoader(test_dataset, vt_batch_size, shuffle=False)

        # best_valid = float('inf')

        best_test1 = float('inf')
        best_test2 = float('inf')
        best_test3 = float('inf')

        best_loss1 = 10000
        best_loss2 = 10000
        best_loss3 = 10000
        # if save_dir != '':
        #     if not os.path.exists(save_dir):
        #         os.makedirs(save_dir)
        # if log_dir != '':
        #     if not os.path.exists(log_dir):
        #         os.makedirs(log_dir)
        #     writer = SummaryWriter(log_dir=log_dir)

        train_loss_list = []
        test_rmse_list = []


        for epoch in range(1, epochs + 1):
            print("\n=====Epoch {}".format(epoch), flush=True)

            print('\nTraining...', flush=True)
            train_mae = self.train(model, optimizer, train_loader, loss_func, device)

            train_loss_list.append(train_mae)

            print('\n\nEvaluating...', flush=True)
            valid_rmse1 = self.val(model, valid_loader, device)

            print('\n\nTesting...', flush=True)
            test_rmse1 = self.val(model, test_loader, device)

            test_rmse_list.append(test_rmse1)

            # pd.DataFrame([train_loss_list, test_rmse_list]).T.to_csv(out_dir, index= False)

            print()
            # print({'Train': train_loss, 'Validation': test_mae, 'Test': test_mae})
            print({'Train': train_mae, 'Valid': valid_rmse1, 'Test': test_rmse1})

            # if log_dir != '':
            #     writer.add_scalar('train_loss', train_mae, epoch)
            #
            #     writer.add_scalar('valid_mae', valid_rmse, epoch)
            #
            #     writer.add_scalar('test_rmse', test_rmse, epoch)

            if valid_rmse1 < best_loss1:
                best_loss1 = valid_rmse1
                best_test1 = test_rmse1

                torch.save(model.state_dict(), save_dir)
                # if save_dir != '':
                    # print('Saving checkpoint...')
                    # checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'best_valid_mae': best_valid, 'num_params': num_params}

                    # checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'num_params': num_params}
                    #
                    #
                    # torch.save(checkpoint, os.path.join(save_dir, 'valid_checkpoint.pt'))

            # if valid_rmse2 < best_loss2:
            #     best_loss2 = valid_rmse2
            #     best_test2 = test_rmse2
            #
            # if valid_rmse3 < best_loss3:
            #     best_loss3 = valid_rmse3
            #     best_test3 = test_rmse3

            scheduler.step()

        print(f'Best validation C RMSE so far: {best_loss1}')
        print(f'Test C MAE when got best validation result: {best_test1}')

        # print(f'Best validation C RMSE so far: {best_loss2}')
        # print(f'Test C MAE when got best validation result: {best_test2}')
        #
        # print(f'Best validation C RMSE so far: {best_loss3}')
        # print(f'Test C MAE when got best validation result: {best_test3}')

        return train_loss_list, test_rmse_list
        

    def train(self, model, optimizer, train_loader, loss_func, device):
        r"""
        The script for training.

        Args:
            model: Which 3DGN model to use. Should be one of the SchNet, DimeNetPP, and SphereNet.
            optimizer (Optimizer): Pytorch optimizer for trainable parameters in training.
            train_loader (Dataloader): Dataloader for training.
            loss_func (function): The used loss funtion for training.
            device (torch.device): The device where the model is deployed.

        :rtype: Traning loss. ( :obj:`mae`)

        """
        model.train()

        loss_accum = 0

        # for step, batch_data in enumerate(tqdm(train_loader)):
        #     batch_data = batch_data.to(device)

        for step, batch_data in enumerate(tqdm(train_loader)):

            optimizer.zero_grad()

            batch_data = batch_data.to(device)

            out = model(batch_data)

            loss = loss_func(out[batch_data.train_carbon_mask], batch_data.y.unsqueeze(1)[batch_data.train_carbon_mask])
            # loss = loss_func(out[batch_data.train_hydrogen_mask], batch_data.y.unsqueeze(1)[batch_data.train_hydrogen_mask])
            # loss = loss_func(out[batch_data.train_mask], batch_data.y.unsqueeze(1)[batch_data.train_mask])
            loss.backward()
            optimizer.step()
            loss_accum = loss_accum + loss.item()


        return loss_accum / (step + 1)


    def val(self, model, data_loader, device):
        r"""
        The script for validation/test.

        Args:
            model: Which 3DGN model to use. Should be one of the SchNet, DimeNetPP, and SphereNet.
            data_loader (Dataloader): Dataloader for validation or test.
            evaluation (function): The used funtion for evaluation.
            device (torch.device, optional): The device where the model is deployed.

        :rtype: Evaluation result. ( :obj:`mae`)

        """

        model.eval()

        preds1 = torch.Tensor([]).to(device)
        targets1 = torch.Tensor([]).to(device)


        with torch.no_grad():
            for step, batch_data in enumerate(tqdm(data_loader)):
                batch_data = batch_data.to(device)
                out = model(batch_data)

                preds1 = torch.cat([preds1,
                                   out.detach_()[batch_data.test_carbon_mask]], dim=0)
                targets1 = torch.cat([targets1,
                                     batch_data.y.unsqueeze(1)[batch_data.test_carbon_mask]], dim=0)

                # preds1 = torch.cat([preds1,
                #                    out.detach_()[batch_data.test_hydrogen_mask]], dim=0)
                # targets1 = torch.cat([targets1,
                #                      batch_data.y.unsqueeze(1)[batch_data.test_hydrogen_mask]], dim=0)


                # preds3 = torch.cat([preds3,
                #                    out.detach_()[batch_data.test_mask]], dim=0)
                # targets3 = torch.cat([targets3,
                #                      batch_data.y.unsqueeze(1)[batch_data.test_mask]], dim=0)

        input_dict1 = {"y_true": targets1, "y_pred": preds1}

        # input_dict2 = {"y_true": targets2, "y_pred": preds2}
        #
        #
        # input_dict3 = {"y_true": targets3, "y_pred": preds3}

        y_pred, y_true = input_dict1['y_pred'], input_dict1['y_true']

        # y_pred2, y_true2 = input_dict2['y_pred'], input_dict2['y_true']
        #
        # y_pred3, y_true3 = input_dict3['y_pred'], input_dict3['y_true']

        assert((isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray))
                or
                (isinstance(y_true, torch.Tensor) and isinstance(y_pred, torch.Tensor)))
        assert(y_true.shape == y_pred.shape)

        if isinstance(y_true, torch.Tensor):
            print('use_tensor')
            y_pred = y_pred.detach().cpu().numpy().reshape(-1)
            y_true = y_true.detach().cpu().numpy().reshape(-1)

            squared_diff = np.square(y_true - y_pred)

            # Calculate the mean of the squared differences
            mean_squared_diff = np.mean(squared_diff)

            # Calculate the square root of the mean squared differences
            rmse = np.sqrt(mean_squared_diff)

            # y_pred2 = y_pred2.detach().cpu().numpy().reshape(-1)
            # y_true2 = y_true2.detach().cpu().numpy().reshape(-1)

            # squared_diff2 = np.square(y_true2 - y_pred2)

            # Calculate the mean of the squared differences
            # mean_squared_diff2 = np.mean(squared_diff2)

            # Calculate the square root of the mean squared differences
            # rmse2 = np.sqrt(mean_squared_diff2)

            # y_pred3 = y_pred3.detach().cpu().numpy().reshape(-1)
            # y_true3 = y_true3.detach().cpu().numpy().reshape(-1)

            # squared_diff3 = np.square(y_true3 - y_pred3)

            # Calculate the mean of the squared differences
            # mean_squared_diff3 = np.mean(squared_diff3)

            # Calculate the square root of the mean squared differences
            # rmse3 = np.sqrt(mean_squared_diff3)


            return rmse
        else:
            print('use_array')
            y_pred = y_pred.detach().cpu().item()
            y_true = y_true.detach().cpu().item()

            squared_diff = np.square(y_true - y_pred)

            # Calculate the mean of the squared differences
            mean_squared_diff = np.mean(squared_diff)

            # Calculate the square root of the mean squared differences
            rmse = np.sqrt(mean_squared_diff)



            return rmse
