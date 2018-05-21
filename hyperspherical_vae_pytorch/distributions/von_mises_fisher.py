
import math
import torch
from hyperspherical_vae_pytorch.ops.ive import ive

class VonMisesFisher(torch.distributions.Distribution):

    arg_constraints = {'loc': torch.distributions.constraints.real,
                       'scale': torch.distributions.constraints.positive}
    support = torch.distributions.constraints.real
    has_rsample = True
    _mean_carrier_measure = 0

    @property
    def mean(self):
        return self.loc * (ive(self.__m / 2, self.scale) / ive(self.__m / 2 - 1, self.scale))

    @property
    def stddev(self):
        return self.scale

    def __init__(self, loc, scale, validate_args=None):
        self.dtype = loc.dtype
        self.loc = loc
        self.scale = scale
        self.device = loc.device
        self.__m =  loc.shape[-1]
        self.__e1 = (torch.Tensor([1.] + [0] * (loc.shape[-1] - 1))).to(self.device)
        
        super(VonMisesFisher, self).__init__(self.loc.size(), validate_args=validate_args)

    def sample(self, shape=torch.Size()):
        with torch.no_grad():
            return self.rsample(shape)

    def rsample(self, shape=torch.Size()):
        shape = shape if isinstance(shape, torch.Size) else torch.Size([shape])
        
        w = self.__sample_w_rej(shape=shape)
        
        v = (torch.distributions.Normal(0, 1).sample(
            shape + torch.Size(self.loc.shape)).to(self.device).transpose(0, -1)[1:]).transpose(0, -1).type(torch.float64)
        v = v / v.norm(dim=-1, keepdim=True)
        
        x = torch.cat((w, torch.sqrt(1 - (w ** 2)) * v), -1)
        z = self.__householder_rotation(x)

        return z.type(self.dtype)
        
    def __while_loop(self, b, a, d, shape):
        
        b, a, d = [e.repeat(*shape, *([1] * len(self.scale.shape))) for e in (b, a, d)]
        w, e, bool_mask = torch.zeros_like(b).to(self.device), torch.zeros_like(b).to(self.device), (torch.ones_like(b) == 1).to(self.device)
        
        shape = shape + torch.Size(self.scale.shape)
 
        while bool_mask.sum() != 0:
            e_ = torch.distributions.Beta((self.__m - 1) / 2, (self.__m - 1) / 2).sample(shape).to(self.device).type(torch.float64)
            u = torch.distributions.Uniform(0, 1).sample(shape).to(self.device).type(torch.float64)
            
            w_ = (1 - (1 + b) * e_) / (1 - (1 - b) * e_)
            log_t = (2 * a * b) - (1 - (1 - b) * e_)

            accept = ((self.__m - 1) * log_t - log_t.exp() + d) > torch.log(u)
            reject = 1 - accept
            
            w[bool_mask * accept] = w_[bool_mask * accept]
            e[bool_mask * accept] = e_[bool_mask * accept]

            bool_mask[bool_mask * accept] = reject[bool_mask * accept]
        
        return e, w
    
    def __sample_w_rej(self, shape):
        #c = torch.sqrt((4 * (self.scale ** 2)) + (self.__m - 1) ** 2)
        c = self.scale.type(torch.float64) * torch.sqrt(4 + ((self.__m - 1) ** 2) / (self.scale.type(torch.float64) ** 2))
        b = (-2 * self.scale.type(torch.float64) + c) / (self.__m - 1)
        a = (self.__m - 1 + 2 * self.scale.type(torch.float64) + c) / 4
        d = (4 * a * b) / (1 + b) - (self.__m - 1) * math.log(self.__m - 1)

        self.__b, (self.__e, self.__w) = b, self.__while_loop(b, a, d, shape)
        return self.__w
    
    def __householder_rotation(self, x):
        u = (self.__e1 - self.loc).type(torch.float64)
        u = u / (u.norm(dim=-1, keepdim=True) + 1e-5)
        z = x - 2 * (x * u).sum(-1, keepdim=True) * u
        return z
    
    def entropy(self):
        output = - self.scale * ive(self.__m / 2, self.scale) / ive((self.__m / 2) - 1, self.scale)
        
        return output.view(*(output.shape[:-1])) + self._log_normalization()
        
    def log_prob(self, x):
        return self._log_unnormalized_prob(x) - self._log_normalization()

    def _log_unnormalized_prob(self, x):
        output = self.scale * (self.loc * x).sum(-1, keepdim=True)

        return output.view(*(output.shape[:-1]))

    def _log_normalization(self):
        output = - ((self.__m / 2 - 1) * torch.log(self.scale) - (self.__m / 2) * math.log(2 * math.pi) - (
                self.scale + torch.log(ive(self.__m / 2 - 1, self.scale))))

        return output.view(*(output.shape[:-1]))

