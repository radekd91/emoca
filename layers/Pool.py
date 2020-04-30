from torch_geometric.nn.conv import MessagePassing


class Pool(MessagePassing):

    def __init__(self):
        super(Pool, self).__init__(flow='target_to_source')

    def forward(self, x, pool_mat,  dtype=None):
        x = x.transpose(0,1)
        out = self.propagate(edge_index=pool_mat._indices(), x=x, norm=pool_mat._values(), size=pool_mat.size())
        return out.transpose(0,1)

    def message(self, x_j, norm):
        return norm.view(-1, 1, 1) * x_j
