from __future__ import print_function
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms


class MyIter(object):
    """An iterator."""

    def __init__(self, my_loader):
        self.my_loader = my_loader
        self.loader_iters = [iter(loader) for loader in self.my_loader.loaders]

    def __iter__(self):
        return self

    def __next__(self):
        # When the shortest loader (the one with minimum number of batches)
        # terminates, this iterator will terminates.
        # The `StopIteration` raised inside that shortest loader's `__next__`
        # method will in turn gets out of this `__next__` method.
        batches = [loader_iter.next() for loader_iter in self.loader_iters]
        # return self.my_loader.combine_batch(batches)
        return batches

    # Python 2 compatibility
    next = __next__

    def __len__(self):
        return len(self.my_loader)


class MyLoader(object):
    """This class wraps several pytorch DataLoader objects, allowing each time
    taking a batch from each of them and then combining these several batches
    into one. This class mimics the `for batch in loader:` interface of
    pytorch `DataLoader`.
    Args:
      loaders: a list or tuple of pytorch DataLoader objects
    """

    def __init__(self, loaders):
        self.loaders = loaders

    def __iter__(self):
        return MyIter(self)

    def __len__(self):
        return min([len(loader) for loader in self.loaders])

    # Customize the behavior of combining batches here.
    # def combine_batch(self, batches):
    #   return batches


# loader1 = DataLoader(
#     torchvision.datasets.MNIST(
#         'data', train=True, download=True,
#         transform=transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.1307,), (0.3081,))
#         ])),
#     batch_size=8, shuffle=True)
#
# loader2 = DataLoader(
#     torchvision.datasets.MNIST(
#         'data', train=True, download=True,
#         transform=transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.1307,), (0.3081,))
#         ])),
#     batch_size=64, shuffle=True)
#
# loader3 = DataLoader(
#     torchvision.datasets.MNIST(
#         'data', train=True, download=True,
#         transform=transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.1307,), (0.3081,))
#         ])),
#     batch_size=128, shuffle=True)

# my_loader = MyLoader([loader1, loader2, loader3])


# def loop_through_loader(loader):
#   """A use case of iterating through a mnist dataloader."""
#   for i, b1 in enumerate(loader):
#     data, target = b1
#     if i in [100, 200]:
#       print(type(data), data.size(), type(target), target.size())
#   print('num of batches: {}'.format(i + 1))


# def loop_through_my_loader(loader):
#     """A use case of iterating through my_loader."""
#     for i, batches in enumerate(loader):
#         batch1 = batches[0][0]
#         batch2 = batches[1][0]
#         batch3 = batches[2][0]
#         print(batch1.size())
#         print(batch2.size())
#         print(batch3.size())
#     #   if i in [100, 200]:
#     #     for j, b in enumerate(batches):
#     #       data, target = b
#     #       print(j + 1, type(data), data.size(), type(target), target.size())
#     # print('num of batches: {}'.format(i + 1))
#
#
# for _ in range(4):
#     loop_through_my_loader(my_loader)
#
# print('len(my_loader):', len(my_loader))
