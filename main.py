from parafilter.filters import LMS, RLS
import torch
import numpy as np
import soundfile as sf

fs = 20000
samples = 100000

if __name__ == '__main__':
    filt = RLS(hop=1000, framelen=4000, filterlen=1024).cuda()
    d = torch.sin(torch.arange(samples) / fs * 2 * 3.1415 * 1000) + torch.sin(
        torch.arange(samples) / fs * 2 * 3.1415 * 2000)
    x = torch.sin(torch.arange(samples) / fs * 2 * 3.1415 * 2000)
    d_est, e = filt(d.unsqueeze(0).cuda(), x.cuda())
    y = e[0].cpu().detach().numpy()
    sf.write(r'c:\temp\filtered.wav', y / np.max(np.abs(y)), 20000)

    pass
