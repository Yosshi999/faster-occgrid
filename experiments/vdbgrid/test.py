import unittest
import torch
import vdbgrid_cuda

class TestGrid(unittest.TestCase):
    def test_grid(self):
        X = vdbgrid_cuda.DensityGrid(10, 10, 10)
        b = torch.zeros([10, 10, 10])
        b[4:7, 4:7, 4:7] = 1
        b[9, 9, 9] = 1
        b = b == 1
        X.setValues(b)

        o = torch.FloatTensor([[0, 0, 0]]).cuda()
        d = torch.FloatTensor([[1, 1, 1]]).cuda()
        near = torch.FloatTensor([1e-5]).cuda()
        far = torch.FloatTensor([100]).cuda()
        ray_ind, left, right, _, _ = X.traverse(o, d, near, far, 0.3, 0)
        pos = o + d * (left + right)[:, None] / 2
        self.assertTrue(
            torch.allclose(
                pos.cpu(),
                torch.tensor(
                    [[4.1500, 4.1500, 4.1500],
                    [4.4500, 4.4500, 4.4500],
                    [4.7500, 4.7500, 4.7500],
                    [5.1500, 5.1500, 5.1500],
                    [5.4500, 5.4500, 5.4500],
                    [5.7500, 5.7500, 5.7500],
                    [6.1500, 6.1500, 6.1500],
                    [6.4500, 6.4500, 6.4500],
                    [6.7500, 6.7500, 6.7500],
                    [9.1500, 9.1500, 9.1500],
                    [9.4500, 9.4500, 9.4500],
                    [9.7500, 9.7500, 9.7500]])))

if __name__ == '__main__':
    unittest.main()
