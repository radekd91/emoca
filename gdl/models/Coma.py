import torch
import torch.nn.functional as F
# from pytorch3d.ops import GraphConv
from gdl.layers.ChebConvComa import ChebConv_Coma
from torch_geometric.nn.conv import ChebConv, GCNConv, FeaStConv, SAGEConv, GraphConv, \
    GMMConv, PointConv, XConv, GATConv
from gdl.layers.Pool import Pool
import copy


class Coma(torch.nn.Module):

    def __init__(self, config : dict, downsample_matrices, upsample_matrices, adjacency_matrices, num_nodes):
        super(Coma, self).__init__()
        config = copy.deepcopy(config)
        self.n_layers = config['n_layers']
        self.filters = config['num_conv_filters']
        self.filters.insert(0, config['num_input_features'])  # To get initial features per node
        self.K = config['polygon_order']
        self.z = config['z']
        self.downsample_matrices = downsample_matrices
        self.upsample_matrices = upsample_matrices
        self.adjacency_matrices = adjacency_matrices

        self.A_edge_index, self.A_norm = zip(*[ChebConv_Coma.norm(self.adjacency_matrices[i]._indices(),
                                                                  num_nodes[i]) for i in range(len(num_nodes))])

        self.with_edge_weights = True

        self.conv_type = config['conv_type']
        self.conv_type_name = config['conv_type']['class']
        if 'params' not in config['conv_type'].keys() or not bool(config['conv_type']['params']):
            conv_kwargs = None
        else:
            conv_kwargs = config['conv_type']['params']

        if self.conv_type_name == 'ChebConv_Coma':
            print("Using convolution of type: '%s'" % self.conv_type_name)
            conv_kwargs = conv_kwargs or {}
            self.conv_enc = torch.nn.ModuleList([ChebConv_Coma(self.filters[i], self.filters[i + 1], self.K[i],
                                                               **conv_kwargs)
                                                 for i in range(len(self.filters)-2)])
            self.conv_dec = torch.nn.ModuleList([ChebConv_Coma(self.filters[-i - 1], self.filters[-i - 2],
                                                               self.K[i], **conv_kwargs)
                                                 for i in range(len(self.filters)-1)])
        elif self.conv_type_name == 'ChebConv':
            print("Using convolution of type: '%s'" % self.conv_type_name)
            conv_kwargs = conv_kwargs or {
                'normalization': 'sym'
                # 'normalization': None
            }
            self.conv_enc = torch.nn.ModuleList([ChebConv(self.filters[i], self.filters[i + 1], self.K[i],
                                                          **conv_kwargs)
                                                 for i in range(len(self.filters)-2)])
            self.conv_dec = torch.nn.ModuleList([ChebConv(self.filters[-i - 1], self.filters[-i - 2], self.K[i],
                                                               **conv_kwargs)
                                                 for i in range(len(self.filters) - 1)])
        elif self.conv_type_name == 'GCNConv':
            print("Using convolution of type: '%s'" % self.conv_type_name)
            conv_kwargs = conv_kwargs or {}
            self.conv_enc = torch.nn.ModuleList(
                [GCNConv(self.filters[i], self.filters[i + 1], **conv_kwargs)
                 for i in range(len(self.filters) - 2)])
            self.conv_dec = torch.nn.ModuleList(
                [GCNConv(self.filters[-i - 1], self.filters[-i - 2], **conv_kwargs)
                 for i in range(len(self.filters) - 1)])
        elif self.conv_type_name == 'FeaStConv':
            print("Using convolution of type: '%s'" % self.conv_type_name)
            conv_kwargs = conv_kwargs or {}
            self.with_edge_weights = False
            self.conv_enc = torch.nn.ModuleList(
                [FeaStConv(self.filters[i], self.filters[i + 1], **conv_kwargs)
                 for i in range(len(self.filters) - 2)])
            self.conv_dec = torch.nn.ModuleList(
                [FeaStConv(self.filters[-i - 1], self.filters[-i - 2], **conv_kwargs)
                 for i in range(len(self.filters) - 1)])
        elif self.conv_type_name == 'GATConv':
            print("Using convolution of type: '%s'" % self.conv_type_name)
            conv_kwargs = conv_kwargs or {}
            self.conv_enc = torch.nn.ModuleList(
                [GATConv(self.filters[i], self.filters[i + 1], **conv_kwargs)
                 for i in range(len(self.filters) - 2)])
            self.conv_dec = torch.nn.ModuleList(
                [GATConv(self.filters[-i - 1], self.filters[-i - 2], **conv_kwargs)
                 for i in range(len(self.filters) - 1)])
        elif self.conv_type_name == 'SAGEConv':
            print("Using convolution of type: '%s'" % self.conv_type_name)
            conv_kwargs = conv_kwargs or {}
            self.conv_enc = torch.nn.ModuleList(
                [SAGEConv(self.filters[i], self.filters[i + 1], **conv_kwargs)
                 for i in range(len(self.filters) - 2)])
            self.conv_dec = torch.nn.ModuleList(
                [SAGEConv(self.filters[-i - 1], self.filters[-i - 2],  **conv_kwargs)
                 for i in range(len(self.filters) - 1)])
        elif self.conv_type_name == 'GraphConv':
            print("Using convolution of type: '%s'" % self.conv_type_name)
            conv_kwargs = conv_kwargs or {}
            self.conv_enc = torch.nn.ModuleList(
                [GraphConv(self.filters[i], self.filters[i + 1], **conv_kwargs)
                 for i in range(len(self.filters) - 2)])
            self.conv_dec = torch.nn.ModuleList(
                [GraphConv(self.filters[-i - 1], self.filters[-i - 2], **conv_kwargs)
                 for i in range(len(self.filters) - 1)])
        # elif conv_type_name == 'GMMConv':
        #     self.conv_enc = torch.nn.ModuleList(
        #         [GMMConv(self.filters[i], self.filters[i + 1], self.K[i])
        #          for i in range(len(self.filters) - 2)])
        #     self.conv_dec = torch.nn.ModuleList(
        #         [GMMConv(self.filters[-i - 1], self.filters[-i - 2], self.K[i])
        #          for i in range(len(self.filters) - 1)])
        # elif conv_type_name == 'PointConv':
        #     self.conv_enc = torch.nn.ModuleList(
        #         [PointConv(self.filters[i], self.filters[i + 1], self.K[i])
        #          for i in range(len(self.filters) - 2)])
        #     self.conv_dec = torch.nn.ModuleList(
        #         [PointConv(self.filters[-i - 1], self.filters[-i - 2], self.K[i])
        #          for i in range(len(self.filters) - 1)])
        # elif conv_type_name == 'XConv':
        #     self.conv_enc = torch.nn.ModuleList(
        #         [XConv(self.filters[i], self.filters[i + 1], self.K[i])
        #          for i in range(len(self.filters) - 2)])
        #     self.conv_dec = torch.nn.ModuleList(
        #         [XConv(self.filters[-i - 1], self.filters[-i - 2], self.K[i])
        #          for i in range(len(self.filters) - 1)])
        else:
            raise ValueError("Invalid convolution type: '%s'" % self.conv_type)

        self.conv_dec[-1].bias = None  # No bias for last convolution layer
        self.pool = Pool(treat_batch_dim_separately= self.conv_type_name == 'ChebConv_Coma')
        self.enc_lin = torch.nn.Linear(self.downsample_matrices[-1].shape[0]*self.filters[-1], self.z)
        self.dec_lin = torch.nn.Linear(self.z, self.filters[-1]*self.upsample_matrices[-1].shape[1])
        self.reset_parameters()

        # private stuff
        # self._A_edge_index_batch = None
        # self._A_norm_batch = None
        self._D_edge_batch = None
        self._D_norm_batch = None
        self._D_sizes = None
        self._U_matrices_batch = None
        self._U_norm_batch = None
        self._U_sizes = None
        self._batch_size = None

    def _create_batched_edges(self, batch_size):
        if self.conv_type_name != 'ChebConv_Coma':
            if self._batch_size == batch_size:
                return
            self._batch_size = batch_size
            # self._A_edge_index_batch = []
            # self._A_norm_batch = []

            self._A_edge_index_batch = []
            self._A_norm_batch = []

            for i in range(len(self.A_edge_index)):
                # num_edges = self.A_edge_index[i].shape[1]
                # shape = (self.A_edge_index.shape[1],  self.A_edge_index.shape[1]*batch_size)
                # repeated_edges = self.A_edge_index[i][:,:,None].repeat(1, 1, batch_size)
                # edge_steps = num_vertices*torch.arange(batch_size, device=self.A_edge_index[i].device)\
                #     .reshape((1, 1, batch_size))
                # repeated_edges2 = self.A_edge_index[i][None, :, :].repeat(batch_size, 1, 1)
                # edge_steps2 = num_vertices*torch.arange(batch_size, device=self.A_edge_index[i].device)\
                #     .reshape((batch_size, 1, 1))
                num_vertices = self.adjacency_matrices[i].size()[0]
                repeated_edges = self.A_edge_index[i][:, None, :].repeat(1, batch_size, 1)
                edge_steps = num_vertices*torch.arange(batch_size,
                                                       device=self.A_edge_index[i].device,
                                                       dtype=self.A_edge_index[i].dtype)\
                    .reshape((1, batch_size, 1))
                self._A_edge_index_batch += [(repeated_edges + edge_steps).reshape(2, -1)]
                # self._A_norm_batch += [self.A_norm[i].repeat(batch_size)]
                self._A_norm_batch += [None]


            self._D_edge_batch = []
            self._D_norm_batch = []
            self._D_sizes = []
            for i in range(len(self.downsample_matrices)):
                # num_downsampled_vertices = self.downsample_matrices[i].size()[1]
                num_downsampled_vertices = self.downsample_matrices[i].size()[0]
                repeated_edges = self.downsample_matrices[i]._indices()[:, None, :].repeat(1, batch_size, 1)
                edge_steps = num_downsampled_vertices * torch.arange(batch_size,
                                                         device=self.downsample_matrices[i].device,
                                                         dtype=torch.int64).reshape((1, batch_size, 1))
                self._D_edge_batch += [(repeated_edges + edge_steps).reshape(2, -1)]
                self._D_norm_batch += [self.downsample_matrices[i]._values().repeat(batch_size)]
                self._D_sizes += [[self.downsample_matrices[i].size()[j] * batch_size for j in range(self.downsample_matrices[i].ndim)]]

            self._U_edge_batch = []
            self._U_norm_batch = []
            self._U_sizes = []
            for i in range(len(self.upsample_matrices)):
                num_upsampled_vertices = self.upsample_matrices[i].size()[1]
                repeated_edges = self.upsample_matrices[i]._indices()[:, None, :].repeat(1, batch_size, 1)
                edge_steps = num_upsampled_vertices * torch.arange(batch_size,
                                                         device=self.upsample_matrices[i].device,
                                                         dtype=torch.int64).reshape((1, batch_size, 1))
                self._U_edge_batch += [(repeated_edges + edge_steps).reshape(2, -1)]
                self._U_norm_batch += [self.upsample_matrices[i]._values().repeat(batch_size)]
                self._U_sizes += [[self.upsample_matrices[i].size()[j] * batch_size for j in range(self.upsample_matrices[i].ndim)]]
        else:
            self._A_edge_index_batch = self.A_edge_index
            self._A_norm_batch = self.A_norm

            self._D_edge_batch = [self.downsample_matrices[i]._indices() for i in range(len(self.downsample_matrices))]
            self._D_norm_batch = [self.downsample_matrices[i]._values() for i in range(len(self.downsample_matrices))]
            self._D_sizes = [list(self.downsample_matrices[i].size()) for i in range(len(self.downsample_matrices))]

            self._U_edge_batch = [self.upsample_matrices[i]._indices() for i in range(len(self.upsample_matrices))]
            self._U_norm_batch = [self.upsample_matrices[i]._values() for i in range(len(self.upsample_matrices))]
            self._U_sizes = [list(self.upsample_matrices[i].size()) for i in range(len(self.upsample_matrices))]


    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch_size = data.num_graphs
        self._create_batched_edges(batch_size)
        if self.conv_type_name == 'ChebConv_Coma': #ChebConv_Coma treats batch dimension separately
            x = x.reshape(batch_size, -1, self.filters[0])
        x = self.encoder(x, batch_size)
        x = self.decoder(x)
        if self.conv_type_name == 'ChebConv_Coma': #ChebConv_Coma treats batch dimension separately
            x = x.reshape(-1, self.filters[0])
        return x

    def encoder(self, x, batch_size=None):
        batch_size = batch_size or x.shape[0]
        if self._A_edge_index_batch is None:
            if self.conv_type_name == 'ChebConv_Coma':  # ChebConv_Coma treats batch dimension separately
                self._create_batched_edges(x.shape[0])
            else:
                self._create_batched_edges(1)
        for i in range(self.n_layers):
            # x = self.conv_enc[i](x, self.A_edge_index[i], self.A_norm[i])
            if self.with_edge_weights:
                x = self.conv_enc[i](x, self._A_edge_index_batch[i], self._A_norm_batch[i])
            else:
                x = self.conv_enc[i](x, self._A_edge_index_batch[i])
            # print("Conv_enc %d" % i)
            # print("shape %s" % str(x.size()))
            x = F.relu(x)
            # x = self.pool(x, self.downsample_matrices[i])
            x = self.pool(x, self._D_edge_batch[i], self._D_norm_batch[i], self._D_sizes[i])
            # print("Pool %d" % i)
            # print("shape %s" % str(x.size()))
        x = x.reshape(batch_size, self.enc_lin.in_features)
        x = F.relu(self.enc_lin(x))
        return x

    def decoder(self, x, ):
        # batch_size = batch_size or x.shape[0]
        x = F.relu(self.dec_lin(x))
        if self.conv_type_name == 'ChebConv_Coma': # ChebConv_Coma treats batch dimension separately
            x = x.reshape(x.shape[0], -1, self.filters[-1])
        else:
            x = x.reshape(-1, self.filters[-1])
        for i in range(self.n_layers):
            # x = self.pool(x, self.upsample_matrices[-i-1])
            x = self.pool(x, self._U_edge_batch[-i-1], self._U_norm_batch[-i-1], self._U_sizes[-i-1])
            # print("UnPool %d" % i)
            # print("shape %s" % str(x.size()))
            if self.with_edge_weights:
                x = self.conv_dec[i](x, self._A_edge_index_batch[self.n_layers - i - 1], self._A_norm_batch[self.n_layers - i - 1])
            else:
                x = self.conv_dec[i](x, self._A_edge_index_batch[self.n_layers - i - 1])
            # print("Conv_dec %d" % i)
            # print("shape %s" % str(x.size()))
            x = F.relu(x)
        if self.with_edge_weights:
            x = self.conv_dec[-1](x, self._A_edge_index_batch[-1], self._A_norm_batch[-1])
        else:
            x = self.conv_dec[-1](x, self._A_edge_index_batch[-1])
        return x

    def reset_parameters(self):
        torch.nn.init.normal_(self.enc_lin.weight, 0, 0.1)
        torch.nn.init.normal_(self.dec_lin.weight, 0, 0.1)
