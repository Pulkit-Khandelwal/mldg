
import torch.utils.model_zoo as model_zoo
from sklearn.metrics import accuracy_score
from torch.autograd import Variable
from torch.optim import lr_scheduler

import mlp
from data_reader import BatchImageGenerator
from utils import sgd, crossentropyloss, fix_seed, write_log, compute_accuracy

import numpy as np
import torch.utils.data as data
import os
import os.path
import torch
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler


class ModelBaseline:
    def __init__(self, flags):
        self.batch_size = flags.batch_size
        self.TrainMetaData = []
        self.ValidMetaData = []
        self.TestMetaData = []

        #torch.set_default_tensor_type('torch.cuda.FloatTensor')

        # fix the random seed or not
        fix_seed()
        self.setup_path(flags)
        self.network = mlp.Net(num_classes=flags.num_classes) #it was MLPNet before
        #self.network = self.network.cuda()
        if not os.path.exists(flags.logs):
            os.makedirs(flags.logs)

        flags_log = os.path.join(flags.logs, 'flags_log.txt')
        write_log(flags, flags_log)

        # self.load_state_dict(flags.state_dict)

        self.configure(flags)

    def setup_path(self, flags):

        root_folder = flags.data_root #provide this as input: root_folder = '/Users/pulkit/Desktop/PACS/'

        data_text_files = ['/Users/pulkit/Desktop/data_paths/art_painting.txt',
                            '/Users/pulkit/Desktop/data_paths/cartoon.txt',
                              '/Users/pulkit/Desktop/data_paths/photo.txt',
                              '/Users/pulkit/Desktop/data_paths/sketch.txt']


        self.train_paths = ['/Users/pulkit/Desktop/data_paths/art_painting.txt',
                            '/Users/pulkit/Desktop/data_paths/cartoon.txt',
                              '/Users/pulkit/Desktop/data_paths/photo.txt',
                              '/Users/pulkit/Desktop/data_paths/sketch.txt']

        self.val_paths = ['/Users/pulkit/Desktop/data_paths/art_painting.txt',
                            '/Users/pulkit/Desktop/data_paths/cartoon.txt',
                              '/Users/pulkit/Desktop/data_paths/photo.txt',
                              '/Users/pulkit/Desktop/data_paths/sketch.txt']

        unseen_index = flags.unseen_index

        self.train_paths.remove(self.train_paths[unseen_index])
        self.val_paths.remove(self.val_paths[unseen_index])
        self.unseen_data_path = data_text_files[unseen_index]

        if not os.path.exists(flags.logs):
            os.makedirs(flags.logs)

        flags_log = os.path.join(flags.logs, 'path_log.txt')
        write_log(str(self.train_paths), flags_log)
        write_log(str(self.val_paths), flags_log)
        write_log(str(self.unseen_data_path), flags_log)

        # self.batImageGenTrains = []
        # self.batImageGenVals = []
        # self.batImageGenTest = []

        # Get all the data at once in the dataset from the DataSet class
        # and then sample data at every iteration of training

        for path in data_text_files:

            dataset = BatchImageGenerator(root=root_folder, flist=path,
                                        transform=transforms.Compose([transforms.Resize(227),
                                                                      #transforms.RandomResizedCrop(224),
                                                                  transforms.ToTensor(),
                                                                  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                       std=[0.229, 0.224, 0.225])]))

            dataset_size = len(dataset)
            indices = list(range(dataset_size))

            until_train = int(np.floor(0.8 * dataset_size))
            until_val = int(np.floor(0.9 * dataset_size))

            np.random.shuffle(indices)

            train_indices, val_indices, test_indices = indices[:until_train], indices[until_train:until_val],\
                                                       indices[until_val:]

            self.TrainMetaData.append([dataset, train_indices])
            self.TestMetaData.append([dataset, val_indices])
            self.ValidMetaData.append([dataset, test_indices])

            # train_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size,
            #                                            sampler=SubsetRandomSampler(self.train_indices))
            # val_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size,
            #                                            sampler=SubsetRandomSampler(self.val_indices))
            # test_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size,
            #                                            sampler=SubsetRandomSampler(self.test_indices))
            # #
            # self.batImageGenTrains.append(train_loader)
            # self.batImageGenVals.append(val_loader)
            # self.batImageGenTest.append(test_loader)
            #


    # def load_state_dict(self, state_dict=''):
    #
    #     if state_dict:
    #         try:
    #             tmp = torch.load(state_dict)
    #             pretrained_dict = tmp['state']
    #         except:
    #             pretrained_dict = model_zoo.load_url(state_dict)
    #
    #         model_dict = self.network.state_dict()
    #         # 1. filter out unnecessary keys
    #         pretrained_dict = {k: v for k, v in pretrained_dict.items() if
    #                            k in model_dict and v.size() == model_dict[k].size()}
    #         # 2. overwrite entries in the existing state dict
    #         model_dict.update(pretrained_dict)
    #         # 3. load the new state dict
    #         self.network.load_state_dict(model_dict)


    def heldout_test(self, flags):
        #### Testing on unseen data happens here

        # sample data at each iteration for test data

        self.batImageGenTest = []

        for metadata in self.TestMetaData:
            test_dataset = metadata[0]
            test_indices = metadata[1]
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size,
                                                     sampler=SubsetRandomSampler(test_indices))
            # store for each domain the test data
            self.batImageGenTest.append(test_loader)

        # load the best model in the validation data
        model_path = os.path.join(flags.model_path, 'best_model.tar')
        self.load_state_dict(state_dict=model_path)

        # test
        test_images, test_labels = self.batImageGenTest[flags.unseen_index]

        threshold = 100
        n_slices_test = len(test_images) / threshold
        indices_test = []
        for per_slice in range(n_slices_test - 1):
            indices_test.append(len(test_images) * (per_slice + 1) / n_slices_test)
        test_image_splits = np.split(test_images, indices_or_sections=indices_test)

        # Verify the splits are correct
        test_image_splits_2_whole = np.concatenate(test_image_splits)
        assert np.all(test_images == test_image_splits_2_whole)

        # split the test data into splits and test them one by one
        predictions = []
        self.network.eval()
        for test_image_split in test_image_splits:
            images_test = Variable(torch.from_numpy(np.array(test_image_split, dtype=np.float32)))
            outputs, end_points = self.network(images_test)

            pred = end_points['Predictions']
            pred = pred.data.numpy()
            predictions.append(pred)

        # concatenate the test predictions first
        predictions = np.concatenate(predictions)

        # accuracy
        accuracy = accuracy_score(y_true=test_labels,
                                  y_pred=np.argmax(predictions, -1))

        flags_log = os.path.join(flags.logs, 'heldout_test_log.txt')
        write_log(accuracy, flags_log)

    def configure(self, flags):

        for name, para in self.network.named_parameters():
            print(name, para.size())

        self.optimizer = sgd(model=self.network,
                             parameters=self.network.parameters(),
                             lr=flags.lr,
                             weight_decay=flags.weight_decay,
                             momentum=flags.momentum)

        self.scheduler = lr_scheduler.StepLR(optimizer=self.optimizer, step_size=flags.step_size, gamma=0.1)
        self.loss_fn = crossentropyloss()

    def train(self, flags):
        self.network.train()
        self.best_accuracy_val = -1

        for ite in range(flags.inner_loops):

            # sample data at each iteration for train data

            self.batImageGenTrains = []

            for metadata in self.TrainMetaData:
                train_dataset = metadata[0]
                train_indices = metadata[1]
                train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size,
                                                           sampler=SubsetRandomSampler(train_indices))
                # store for each domain the train data
                self.batImageGenTrains.append(train_loader)

            self.scheduler.step(epoch=ite)

            total_loss = 0.0
            for index in range(len(self.batImageGenTrains)):

                if index == flags.unseen_index:
                    continue

                else:
                    for _, img_label in enumerate(self.batImageGenTrains[index], 0):
                        images_train, labels_train = img_label

                        # inputs, labels = torch.from_numpy(
                        #     np.array(images_train, dtype=np.float32)), torch.from_numpy(
                        #     np.array(labels_train, dtype=np.float32))
                        #
                        # # wrap the inputs and labels in Variable
                        # inputs, labels = Variable(inputs, requires_grad=False), \
                        #                  Variable(labels, requires_grad=False).long()

                        outputs, _ = self.network(x=images_train) #inputs

                        # loss
                        loss = self.loss_fn(outputs, labels_train) #labels
                        total_loss += loss

            # init the grad to zeros first
            self.optimizer.zero_grad()

            # backward your network
            total_loss.backward()

            # optimize the parameters
            self.optimizer.step()

            print(
                'ite:', ite, 'loss:', total_loss.data.numpy(), 'lr:',
                self.scheduler.get_lr()[0])

            flags_log = os.path.join(flags.logs, 'loss_log.txt')
            write_log(
                str(total_loss.data.numpy()),
                flags_log)

            del total_loss, outputs

            if ite % flags.test_every == 0 and ite is not 0 or flags.debug:
                # sample data at each iteration for val data

                self.batImageGenVals = []

                for metadata in self.ValidMetaData:
                    val_dataset = metadata[0]
                    val_indices = metadata[1]
                    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size,
                                                            sampler=SubsetRandomSampler(val_indices))
                    # store for each domain the val data
                    self.batImageGenVals.append(val_loader)

                self.test_workflow(self.batImageGenVals, flags, ite)

    def test_workflow(self, batImageGenVals, flags, ite):

        accuracies = []
        for count, batImageGenVal in enumerate(batImageGenVals):
            accuracy_val = self.test(batImageGenTest=batImageGenVal, flags=flags, ite=ite,
                                     log_dir=flags.logs, log_prefix='val_index_{}'.format(count))

            accuracies.append(accuracy_val)

        mean_acc = np.mean(accuracies)

        if mean_acc > self.best_accuracy_val:
            self.best_accuracy_val = mean_acc

            f = open(os.path.join(flags.logs, 'Best_val.txt'), mode='a')
            f.write('ite:{}, best val accuracy:{}\n'.format(ite, self.best_accuracy_val))
            f.close()

            if not os.path.exists(flags.model_path):
                os.makedirs(flags.model_path)

            outfile = os.path.join(flags.model_path, 'best_model.tar')
            torch.save({'ite': ite, 'state': self.network.state_dict()}, outfile)

    def test(self, flags, ite, log_prefix, log_dir='logs/', batImageGenTest=None):

        # switch on the network test mode
        self.network.eval()
        images_test, labels_test = batImageGenTest

        threshold = 50
        if len(images_test) > threshold:

            n_slices_test = len(images_test) / threshold
            indices_test = []
            for per_slice in range(int(n_slices_test - 1)):
                indices_test.append(len(images_test) * (per_slice + 1) / n_slices_test)
            test_image_splits = np.split(images_test, indices_or_sections=indices_test)

            # Verify the splits are correct
            test_image_splits_2_whole = np.concatenate(test_image_splits)
            assert np.all(images_test == test_image_splits_2_whole)

            # split the test data into splits and test them one by one
            test_image_preds = []
            for test_image_split in test_image_splits:
                images_test = Variable(torch.from_numpy(np.array(test_image_split, dtype=np.float32)))
                outputs, end_points = self.network(images_test)

                predictions = end_points['Predictions']
                predictions = predictions.data.numpy()
                test_image_preds.append(predictions)

            # concatenate the test predictions first
            predictions = np.concatenate(test_image_preds)
        else:
            images_test = Variable(torch.from_numpy(np.array(images_test, dtype=np.float32)))
            outputs, end_points = self.network(images_test)

            predictions = end_points['Predictions']
            predictions = predictions.data.numpy()

        accuracy = compute_accuracy(predictions=predictions, labels=labels_test)
        print('----------accuracy test----------:', accuracy)

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_path = os.path.join(log_dir, '{}.txt'.format(log_prefix))
        write_log(str('ite:{}, accuracy:{}'.format(ite, accuracy)), log_path=log_path)

        # switch on the network train mode after test
        self.network.train()

        return accuracy


class ModelMLDG(ModelBaseline):
    def __init__(self, flags):

        ModelBaseline.__init__(self, flags)

    def train(self, flags):
        self.network.train()

        self.best_accuracy_val = -1

        for ite in range(flags.inner_loops):

            self.scheduler.step(epoch=ite)
            self.meta_step_scale = flags.meta_step_scale

            # select the validation domain for meta val
            index_val = np.random.choice(a=np.arange(0, len(self.batImageGenTrains)), size=1)[0]
            batImageMetaVal = self.batImageGenTrains[index_val]

            meta_train_loss = 0.0
            # get the inputs and labels from the data reader
            for index in range(len(self.batImageGenTrains)):

                if index == index_val:
                    continue

                images_train, labels_train = self.batImageGenTrains[index].get_images_labels_batch()

                inputs_train, labels_train = torch.from_numpy(
                    np.array(images_train, dtype=np.float32)), torch.from_numpy(
                    np.array(labels_train, dtype=np.float32))

                # wrap the inputs and labels in Variable
                inputs_train, labels_train = Variable(inputs_train, requires_grad=False), \
                                             Variable(labels_train, requires_grad=False).long()

                # forward with the adapted parameters
                outputs_train, _ = self.network(x=inputs_train)

                # loss
                loss = self.loss_fn(outputs_train, labels_train)
                meta_train_loss += loss

            image_val, labels_val = batImageMetaVal.get_images_labels_batch()
            inputs_val, labels_val = torch.from_numpy(
                np.array(image_val, dtype=np.float32)), torch.from_numpy(
                np.array(labels_val, dtype=np.float32))

            # wrap the inputs and labels in Variable
            inputs_val, labels_val = Variable(inputs_val, requires_grad=False), \
                                     Variable(labels_val, requires_grad=False).long()

            # forward with the adapted parameters
            outputs_val, _ = self.network(x=inputs_val,
                                          meta_loss=meta_train_loss,
                                          meta_step_size=self.meta_step_scale * self.scheduler.get_lr()[0],
                                          stop_gradient=flags.stop_gradient)

            meta_val_loss = self.loss_fn(outputs_val, labels_val)

            total_loss = meta_train_loss + meta_val_loss

            # init the grad to zeros first
            self.optimizer.zero_grad()

            # backward your network
            total_loss.backward()

            # optimize the parameters
            self.optimizer.step()

            print(
                'ite:', ite,
                'meta_train_loss:', meta_train_loss.data.numpy(),
                'meta_val_loss:', meta_val_loss.data.numpy(),
                'lr:',
                self.scheduler.get_lr()[0])

            flags_log = os.path.join(flags.logs, 'loss_log.txt')
            write_log(
                str(meta_train_loss.data.numpy()) + '\t' + str(meta_val_loss.data.numpy()),
                flags_log)

            del total_loss, outputs_val, outputs_train

            if ite % flags.test_every == 0 and ite is not 0 or flags.debug:
                self.test_workflow(self.batImageGenVals, flags, ite)
