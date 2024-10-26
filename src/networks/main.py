from .mnist_LeNet import MNIST_LeNet, MNIST_LeNet_Autoencoder
from .cifar10_LeNet import CIFAR10_LeNet, CIFAR10_LeNet_Autoencoder
from .cifar10_LeNet_elu import CIFAR10_LeNet_ELU, CIFAR10_LeNet_ELU_Autoencoder
from .mvtecad_LeNet_elu import MVTecAD_LeNet_ELU, MVTecAD_LeNet_ELU_Autoencoder

def build_network(net_name):
    """Builds the neural network."""

    # implemented_networks = ('anoshift_Transformer', 'mnist_LeNet', 'cifar10_LeNet', 'cifar10_LeNet_ELU')
    # assert net_name in implemented_networks

    net = None

    if net_name == 'mnist_LeNet':
        net = MNIST_LeNet()

    if net_name == 'cifar10_LeNet':
        net = CIFAR10_LeNet()

    if net_name == 'cifar10_LeNet_ELU':
        net = CIFAR10_LeNet_ELU()
        
    if 'mvtecad_LeNet_ELU' in net_name:
        version = net_name.split('-')
        net = MVTecAD_LeNet_ELU(int(version[-1]))

    return net


def build_autoencoder(net_name):
    """Builds the corresponding autoencoder network."""

    # implemented_networks = ('anoshift_Transformer', 'mnist_LeNet', 'cifar10_LeNet', 'cifar10_LeNet_ELU')
    # assert net_name in implemented_networks

    ae_net = None

    if net_name == 'mnist_LeNet':
        ae_net = MNIST_LeNet_Autoencoder()

    if net_name == 'cifar10_LeNet':
        ae_net = CIFAR10_LeNet_Autoencoder()

    if net_name == 'cifar10_LeNet_ELU':
        ae_net = CIFAR10_LeNet_ELU_Autoencoder()
        
    if 'mvtecad_LeNet_ELU' in net_name:
        version = net_name.split('-')
        ae_net = MVTecAD_LeNet_ELU_Autoencoder(int(version[-1]))

    return ae_net
