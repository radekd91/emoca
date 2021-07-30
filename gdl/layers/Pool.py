from torch_geometric.nn.conv import MessagePassing


class Pool(MessagePassing):

    def __init__(self, treat_batch_dim_separately : bool):
        super(Pool, self).__init__(flow='target_to_source')
        self.treat_batch_dim_separately = treat_batch_dim_separately

    # def forward(self, x, pool_mat):
    def forward(self, x, edge_index, norm, size):
        # print("Pool x shape")
        # print(x.shape)
        # print("Pool mat shape")
        # print(pool_mat.shape)

        if self.treat_batch_dim_separately:
            x = x.transpose(0,1)
        # out = self.propagate(edge_index=pool_mat._indices(), x=x, norm=pool_mat._values(), size=pool_mat.size())
        out = self.propagate(edge_index=edge_index, x=x, norm=norm, size=size)
        if self.treat_batch_dim_separately:
            out = out.transpose(0,1)
        return out

    def message(self, x_j, norm):
        if self.treat_batch_dim_separately:
            return norm.view(-1, 1, 1) * x_j
        return norm.view(-1, 1) * x_j
