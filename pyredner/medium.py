import pyredner
import torch

class Medium:
    """
        redner supports the rendering of homogeneous participating
        media in order to generate more photorealistic effects.
        The phase function that is used for internal volume scattering
        is the Henyey-Greenstein function with a definable 'g' parameter.

        Args:
            sigma_a (length 3 float tensor): the absorption factor of the medium
            sigma_s (length 3 float tensor): the scattering factor of the medium
            g (length 1 float tensor): the parameter used to tune the forward and
            backward scattering of the phase function used for volume scattering
    """
    def __init__(self, sigma_a, sigma_s, g):
        assert(sigma_a.dtype == torch.float32)
        assert(len(sigma_a.shape) == 1 and sigma_a.shape[0] == 3)
        assert(sigma_a.device.type == 'cpu')
        assert(sigma_s.dtype == torch.float32)
        assert(len(sigma_s.shape) == 1 and sigma_s.shape[0] == 3)
        assert(sigma_s.device.type == 'cpu')
        assert(g.dtype == torch.float32)
        assert(len(g.shape) == 1 and g.shape[0] == 1)
        assert(g.device.type == 'cpu')

        self.sigma_a = sigma_a
        self.sigma_s = sigma_s
        self.g = g

    def state_dict(self):
        return {
            'sigma_a': self.sigma_a.state_dict(),
            'sigma_s': self.sigma_s.state_dict(),
            'g': self.g.state_dict(),
        }

    @classmethod
    def load_state_dict(cls, state_dict):
        out = cls.__new__(Medium)
        out.sigma_a = state_dict['sigma_a']
        out.sigma_s = state_dict['sigma_s']
        out.g = state_dict['g']
        return out

class HomogeneousMedium(Medium):
    def __init__(self, sigma_a, sigma_s, g):
        assert(sigma_a.dtype == torch.float32)
        assert(len(sigma_a.shape) == 1 and sigma_a.shape[0] == 3)
        assert(sigma_a.device.type == 'cpu')
        assert(sigma_s.dtype == torch.float32)
        assert(len(sigma_s.shape) == 1 and sigma_s.shape[0] == 3)
        assert(sigma_s.device.type == 'cpu')
        assert(g.dtype == torch.float32)
        assert(len(g.shape) == 1 and g.shape[0] == 1)
        assert(g.device.type == 'cpu')

        self.sigma_a = sigma_a
        self.sigma_s = sigma_s
        self.g = g

    def state_dict(self):
        return {
            'sigma_a': self.sigma_a.state_dict(),
            'sigma_s': self.sigma_s.state_dict(),
            'g': self.g.state_dict(),
        }

    @classmethod
    def load_state_dict(cls, state_dict):
        out = cls.__new__(HomogeneousMedium)
        out.sigma_a = state_dict['sigma_a']
        out.sigma_s = state_dict['sigma_s']
        out.g = state_dict['g']
        return out