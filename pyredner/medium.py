import pyredner

class Medium:
    def __init__(self, sigma_a, sigma_s, g):
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
