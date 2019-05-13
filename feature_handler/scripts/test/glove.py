import nltk
# nltk.download("movie_reviews")
# nltk.download()
# nltk.download_gui()

import torch

T_data = [[[1., 2.], [3., 4.]],
          [[5., 6.], [7., 8.]]]
T = torch.Tensor(T_data)

print(T.size())

