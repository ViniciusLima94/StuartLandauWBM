import torch


class SLM(torch.nn.Module):

    def __init__(
        self,
        n_nodes: int,
        g: float,
        f: float,
        h: float,
        eta: float,
        n_inputs: int,
        n_outputs: int,
    ):
        super().__init__()

        # Network parameters
        self.n_nodes = n_nodes

        # Node parameters
        self.a = torch.nn.Parameter(-5 * torch.rand(n_nodes))
        self.g = g
        self.eta = eta * torch.sqrt(torch.tensor(h))
        if isinstance(f, (int, float)):
            self.f = f * torch.ones(n_nodes)
        else:
            self.f = 60 * torch.rand(n_nodes) + 20
        self.f = torch.nn.Parameter(2 * torch.pi * self.f)

        # Simulation parameters
        self.h = h

        # Recurrent connections
        self.W_rec = torch.nn.Parameter(torch.rand((n_nodes, n_nodes)) * 0.1)
        # Input to hidden layer
        self.W_input = torch.nn.Linear(n_inputs, n_nodes)
        # Output layer
        self.W_output = torch.nn.Linear(n_nodes, n_outputs)

    def __ode(self, z: torch.Tensor, a: torch.Tensor, w: torch.Tensor):
        return z * (a + 1j * w - torch.abs(z * z))

    def loop(self, z_t: torch.Tensor, I_ext: torch.Tensor):

        z_t = (
            z_t
            + self.h * self.__ode(z_t, self.a, self.f)
            + self.h * self.g * (self.W_rec * (z_t[:, None] - z_t[..., None])).sum(-1)
            + self.h * self.W_input(I_ext)
            + self.eta * torch.randn(z_t.shape)
            + self.eta * 1j * torch.randn(z_t.shape)
        )

        return z_t

    def forward(self, n_trials: int, n_times: int, input: torch.Tensor):

        rec_z_t = torch.zeros(n_trials, n_times, self.n_nodes, dtype=torch.complex64)

        z_t = self.h * torch.randn(n_trials, self.n_nodes) + 1j * self.h * torch.randn(
            n_trials, self.n_nodes
        )

        z_t = torch.autograd.Variable(z_t)

        for t in range(n_times):
            z_t = self.loop(z_t, input[t])
            rec_z_t[:, t, :] = z_t

        output = {}
        output["rec"] = rec_z_t
        output["output"] = self.W_output(z_t.real)

        return output


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import torchvision

    ## Simulate dynamics no inputs

    T = 5
    h = 1e-4
    n_trials = 10
    n_times = int(T // h)
    n_nodes = 29

    input = torch.zeros((n_times, n_trials, 1))
    model = SLM(n_nodes, 1, 40, h, 1e-3, 1, 1)
    out = model.forward(n_trials, n_times, input)

    out = out["rec"].cpu().detach().numpy()

    out = (out - out.mean(1)[:, None, :]) / out.std(1)[:, None, :]

    for i in range(29):
        plt.plot(out[0, :, i].real + (i * 2))
    plt.show()

    ## Simulate dynamics with inputs

    # Use MNIST as input
    dim_input = 1

    # 10 MNIST classes
    dim_output = 10

    # batch size of the test set
    batch_size_train = 100
    batch_size_test = 1000

    # load dataset
    size_validation = 1000  # size of validation dataset
    train_set = torchvision.datasets.MNIST(
        root="data",
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True,
    )
    test_set = torchvision.datasets.MNIST(
        root="data",
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=True,
    )
    train_set, valid_set = torch.utils.data.random_split(
        train_set, [len(train_set) - size_validation, size_validation]
    )

    # data loaders
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=batch_size_train, shuffle=True
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_set, batch_size=batch_size_test, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=batch_size_test, shuffle=False
    )

    model = SLM(n_nodes, 1, None, h, 1e-2, 1, 10)

    # bce loss and optimizer for training
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

    images = list(train_loader)[0][0].reshape(batch_size_train, 1, 28 * 28)
    labels = list(train_loader)[0][1]
    images = images.permute(2, 0, 1)

    n_times, n_trials, _ = images.shape

    out = model.forward(n_trials, n_times, images)

    out = out["rec"].cpu().detach().numpy()

    out = (out - out.mean(1)[:, None, :]) / out.std(1)[:, None, :]

    for i in range(29):
        plt.plot(out[0, :, i].real + (i * 2))
    plt.show()
