import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNNConfig(object):
    """
    CNN Parameters
    """

    def __init__(self, num_epochs=10, learning_rate=0.001):
        self.learning_rate = learning_rate  # learning rate
        self.num_epochs = num_epochs  # total number of epochs

        self.embedding_dim = 100  # embedding vector size
        self.seq_length = 40  # maximum length of sequence
        self.vocab_size = 8000  # most common words

        self.num_filters = 150  # 100  # number of the convolution filters (feature maps)
        self.kernel_sizes = [3, 4, 5]  # three kind of kernels (windows)

        self.hidden_dim = 600  # hidden size of fully connected layer

        self.dropout_prob = 0.5  # how much probability to be dropped

        self.batch_size = 128  # batch size for training

        self.num_classes = 2  # number of classes
        self.target_names = ['--', '-', '=', '+', '++']
        self.target_class = 1
        self.dev_split = 0.1  # percentage of dev data

        self.stratified = False
        self.balance = False
        self.stratified_batch = False

        self.cuda_device = 0  # cuda device to be used when available


class TextCNNEncoder(nn.Module):
    """
    Textual encoder baseado em convoluções.
    """

    def __init__(self, config, pre_trained_emb=None):
        super(TextCNNEncoder, self).__init__()

        V = config.vocab_size
        E = config.embedding_dim
        Nf = config.num_filters
        Ks = config.kernel_sizes
        C = config.num_classes
        Dr = config.dropout_prob
        if pre_trained_emb is None:
            self.embedding = nn.Embedding(V, E, padding_idx=0)  # embedding layer
        else:
            self.embedding = nn.Embedding(V, E, padding_idx=0).from_pretrained(pre_trained_emb, freeze=False)

        # convolutional layers
        self.convs = nn.ModuleList([nn.Conv1d(E, Nf, k) for k in Ks])

    @staticmethod
    def conv_and_max_pool(x, conv):
        """Convolution and global max pooling layer"""
        return F.relu(conv(x).permute(0, 2, 1)).max(1)[0]

    def forward(self, inputs):
        # Conv1d takes in (batch, channels, seq_len), but raw embedded is (batch, seq_len, channels)
        embedded = self.embedding(inputs).permute(0, 2, 1)
        x = [self.conv_and_max_pool(embedded, k) for k in self.convs]  # convolution and global max pooling
        x = torch.cat(x, 1)
        return x

    def load_pre_trained(self, file):
        """
        Carrega pesos das camadas de embedding e convolução.

        :param emb_file:
        :param conv_file:
        :return:
        """

        device = torch.device('cpu')
        # Embeddings
        model = torch.load(file, map_location=device)
        print(model.keys())
        self.load_state_dict()
        #self.embedding.load_state_dict(model['encoder.embedding.weight'])
        # Conv. Filters
        #conv = torch.load(conv_file, map_location=device)
        #self.convs.load_state_dict(model['encoder.convs'])

    def use_pre_trained_layers(self, file, requires_grad):
        self.load_pre_trained(file)
        self.embedding.requires_grad_(requires_grad)
        for c in self.convs:
            c.requires_grad_(requires_grad)

    def save(self, paths):
        """
        Salva camadas de embedding e convolução em arquivos.

        :param paths:
        :return:
        """
        torch.save(self.embedding.state_dict(), paths[0])
        torch.save(self.convs.state_dict(), paths[1])


class TextCNN(nn.Module):
    """
    CNN text classification model, based on the paper.
    """

    def __init__(self, config, pre_trained_emb=None):
        super(TextCNN, self).__init__()

        self.encoder = TextCNNEncoder(config, pre_trained_emb=pre_trained_emb)  # text encoder
        self.dropout = nn.Dropout(config.dropout_prob)  # a dropout layer
        self.fc1 = nn.Linear(len(config.kernel_sizes) * config.num_filters,
                             config.num_classes)  # a dense layer for classification

    def forward(self, inputs):
        x = self.encoder(inputs)
        x = self.dropout(x)
        x = self.fc1(x)
        return x


class ETextCNN(nn.Module):
    """
    TextCNN com hidden layer.
    """

    def __init__(self, config, pre_trained_emb=None):
        super(ETextCNN, self).__init__()

        self.encoder = TextCNNEncoder(config, pre_trained_emb=pre_trained_emb)  # text encoder
        self.dropout = nn.Dropout(config.dropout_prob)  # a dropout layer
        self.fc1 = nn.Linear(len(config.kernel_sizes) * config.num_filters, config.hidden_dim)  # a dense hidden layer
        self.fc2 = nn.Linear(config.hidden_dim, config.num_classes)  # a dense hidden layer

    def forward(self, inputs):
        # Conv1d takes in (batch, channels, seq_len), but raw embedded is (batch, seq_len, channels)
        embedded = self.encoder.embedding(inputs).permute(0, 2, 1)
        x_c = [TextCNNEncoder.conv_and_max_pool(embedded, k) for k in self.encoder.convs]  # convolution and global max pooling
        # x_cat = self.dropout(torch.cat(x, 1))
        x_cat = torch.cat(x_c, 1)
        x_h = self.dropout(F.relu(self.fc1(x_cat)))

        x = self.fc2(x_h)
        return x