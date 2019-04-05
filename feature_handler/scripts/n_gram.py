# Author: Robert Guthrie
# Translator: Don Kim
import torch

torch.manual_seed(1)

word_to_ix = {"hello": 0, "world": 1}
# 단어장에 단어 2개, embeddings의 크기 5
embeds = torch.nn.Embedding(2, 5)
lookup_tensor = torch.LongTensor([word_to_ix["hello"]])
hello_embed = embeds(torch.autograd.Variable(lookup_tensor))
print(hello_embed)

CONTEXT_SIZE = 2
EMBEDDING_DIM = 10
# Shakespeare Sonnet 2
test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()
# 원래는 input을 tokenize해야하는 단계가 있으나, 여기서는 그냥 넘어가기로 한다.
# Tuple의 list를 만든다. 각 tuple은 ([ word_i-2, word_i-1 ]. target word)
trigrams = [([test_sentence[i], test_sentence[i+1]], test_sentence[i+2])
            for i in range(len(test_sentence)-2)]
# 어떻게 생겼나 구경하기 위해 첫 3개를 출력해본다.
print(trigrams[:3])

vocab = set(test_sentence)
word_to_ix = {word: i for i, word in enumerate(vocab)}


class NGramLanguageModeler(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = torch.nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = torch.nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = torch.nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = torch.nn.functional.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = torch.nn.functional.log_softmax(out, dim=1)
        return log_probs


losses = []
loss_function = torch.nn.NLLLoss()
model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

for epoch in range(10):
    total_loss = torch.Tensor([0])
    for context, target in trigrams:
        # Step 1. Input이 모델에 들어갈 수 있는 모양으로 준비해준다.
        # (단어를 정수형 index로 바꾸고 Variable에 담아서 보내준다.)
        context_idxs = [word_to_ix[w] for w in context]
        context_var = torch.autograd.Variable(torch.LongTensor(context_idxs))

        # Step 2. Torch에서 gradient는 축적된다는 기억하자.
        # 새로운 데이터를 넣기 전에, 기존 gradient 정보를 날려줘야 한다.
        model.zero_grad()

        # Step 3. Forward pass를 돌리고,
        # 다음에 올 단어에 대한 log probability를 구한다.
        log_probs = model(context_var)

        # Step 4. Loss function을 계산한다.
        # 다시 말하지만, Torch는 Variable로 포장된 target을 필요로 한다.
        loss = loss_function(
            log_probs,
            torch.autograd.Variable(torch.LongTensor([word_to_ix[target]]))
        )

        # Step 5. Backward pass를 돌리고 gradient를 업데이트한다.
        loss.backward()
        optimizer.step()

        total_loss += loss.data
    losses.append(total_loss)
    print("total_loss : ", total_loss)

# print("ttt")
# print(losses)  # 매 epoch가 지날 때마다 training data에 대한 loss는 줄어든다!


CONTEXT_SIZE = 2  # 왼쪽/오른쪽으로 두 단어씩
raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules called a
program. People create programs to direct processes. In effect, we
conjure the spirits of the computer with our spells.""".split()

vocab = set(raw_text)
vocab_size = len(vocab)

word_to_ix = {word: i for i, word in enumerate(vocab)}
data = []
for i in range(2, len(raw_text) - 2):
    context = [raw_text[i-2], raw_text[i-1],
               raw_text[i+1], raw_text[i+2]]
    target = raw_text[i]
    data.append((context, target))
print(data[:5])


# 모델을 구성하고 학습시켜라.
# 역자가 숙제를 했지만, 혹시 이것 보시는 분들도 읽지만 말고 직접 해보시길...
class CBOW(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(CBOW, self).__init__()
        self.embeddings = torch.nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = torch.nn.Linear(2 * context_size * embedding_dim, 128)
        self.linear2 = torch.nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = torch.nn.functional.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = torch.nn.functional.log_softmax(out, dim=1)
        return log_probs

# 당신의 모듈에 맞게 자료를 사용할 수 있도록 함수를 제공해주겠다.
def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    tensor = torch.LongTensor(idxs)
    return torch.autograd.Variable(tensor)


make_context_vector(data[0][0], word_to_ix)  # 사용 예

# BoW 예제와 거의 다르지 않다.
losses = []
loss_function = torch.nn.NLLLoss()
model = CBOW(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

for epoch in range(10):
    total_loss = torch.Tensor([0])
    for context, target in data:
        # Step 1. Input
        context_var = make_context_vector(context, word_to_ix)

        # Step 2. Reset gradients
        model.zero_grad()

        # Step 3. Forward pass
        log_probs = model(context_var)

        # Step 4. Loss function
        loss = loss_function(
            log_probs,
            torch.autograd.Variable(torch.LongTensor([word_to_ix[target]]))
        )

        # Step 5. Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.data
    losses.append(total_loss)
print(losses)  # 매 epoch가 지날 때마다 training data에 대한 loss는 줄어든다!
