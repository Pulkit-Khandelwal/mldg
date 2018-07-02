import h5py
import numpy as np
from utils import unfold_label, shuffle_data
import torch.utils.data as data
from PIL import Image
import os
import os.path

### Overrrdie the Dataset class to read the dataset form the file lists
### We shoudl use the dataloader in pytorch rather than the
## stuff given in this repo!!!!!


def default_loader(path):
    return Image.open(path).convert('RGB')


def default_flist_reader(flist):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath, imlabel = line.strip().split()
            imlist.append((impath, int(imlabel)))

    return imlist


class BatchImageGenerator(data.Dataset):
    def __init__(self, root, flist, transform=None, target_transform=None,
                 flist_reader=default_flist_reader, loader=default_loader):
        # flags, stage, file_path, b_unfold_label
        # if stage not in ['train', 'val', 'test']:
        #     assert ValueError('invalid stage!')
        #
        # self.configuration(flags, stage, file_path)
        # self.load_data(b_unfold_label)

        self.root = root
        self.imlist = flist_reader(flist)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    # def configuration(self, flags, stage, file_path):
    #     self.batch_size = flags.batch_size
    #     self.current_index = -1
    #     self.file_path = file_path
    #     self.stage = stage
    #     self.shuffled = False


    def __getitem__(self, index):
        impath, target = self.imlist[index]
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imlist)


    # def normalize(self, inputs):
    #
    #     # the mean and std used for the normalization of
    #     # the inputs for the pytorch pretrained model
    #     mean = [0.485, 0.456, 0.406]
    #     std = [0.229, 0.224, 0.225]
    #
    #     # norm to [0, 1]
    #     inputs = inputs / 255.0
    #
    #     inputs_norm = []
    #     for item in inputs:
    #         item = np.transpose(item, (2, 0, 1))
    #         item_norm = []
    #         for c, m, s in zip(item, mean, std):
    #             c = np.subtract(c, m)
    #             c = np.divide(c, s)
    #             item_norm.append(c)
    #
    #         item_norm = np.stack(item_norm)
    #         inputs_norm.append(item_norm)
    #
    #     inputs_norm = np.stack(inputs_norm)
    #
    #     return inputs_norm


    # def load_data(self, b_unfold_label):
    #     file_path = self.file_path
    #     print(file_path)
    #     with h5py.File(file_path, 'r') as f:
    #     #f = h5py.File(file_path, 'r')
    #         self.images = np.array(f['images'])
    #         self.labels = np.array(f['labels'])
    #     #f.close()
    #
    #     # shift the labels to start from 0
    #     self.labels -= np.min(self.labels)
    #
    #     if b_unfold_label:
    #         self.labels = unfold_label(labels=self.labels, classes=len(np.unique(self.labels)))
    #     assert len(self.images) == len(self.labels)
    #
    #     self.file_num_train = len(self.labels)
    #     print('data num loaded:', self.file_num_train)
    #
    #     if self.stage is 'train':
    #         self.images, self.labels = shuffle_data(samples=self.images, labels=self.labels)

    # def get_images_labels_batch(self):
    #
    #     images = []
    #     labels = []
    #     for index in range(self.batch_size):
    #         self.current_index += 1
    #
    #         # void over flow
    #         if self.current_index > self.file_num_train - 1:
    #             self.current_index %= self.file_num_train
    #
    #             self.images, self.labels = shuffle_data(samples=self.images, labels=self.labels)
    #
    #         images.append(self.images[self.current_index])
    #         labels.append(self.labels[self.current_index])
    #
    #     images = np.stack(images)
    #     labels = np.stack(labels)
    #
    #     return images, labels



# if __name__ == "__main__":
#     import h5py
#     import numpy as np
#     from utils import unfold_label, shuffle_data
#     import torch.utils.data as data
#     from PIL import Image
#     import os
#     import os.path
#     import torch
#     from torchvision import transforms
#
#     root_folder = '/Users/pulkit/Desktop/PACS/'
#     train_art_painting_file = '/Users/pulkit/Desktop/data_paths/art_painting.txt'
#
#     train_loader = torch.utils.data.DataLoader(
#                     BatchImageGenerator(root=root_folder, flist=train_art_painting_file,
#                                         transform=transforms.Compose([transforms.RandomResizedCrop(224),
#                                           transforms.ToTensor(),
#                                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                                                std=[0.229, 0.224, 0.225])])),
#                                           batch_size=64, shuffle=True)
#
#     #print(iter(train_loader).next())
#     imgs, labels =  iter(train_loader).next() #gives a batch of the dataset from the DataLoader
#     print(imgs.shape, labels.shape)
