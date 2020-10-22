import numpy as np


def Frobenius(mat):
    size = mat.shape
    print(size[0])
    ret = (np.sum(np.sum((mat ** 2), 1), 2).squeeze() + 1e-10) ** 0.5
    return np.sum(ret) / size[0]



if __name__ == '__main__':
    #A = np.ones(30).reshape(2, 3, 5)
    A = np.ones((2, 3, 5))
    # print(type(A))
    # print(A.shape)
    print(A)
    #extra_loss = Frobenius(torch.bmm(attention, attentionT) - I[:attention.size(0)])
    AT = A.transpose(0, 2, 1)
    # print(AT)
    AAT = np.matmul(A, AT)
    I = np.identity(3)
    #print(I.shape)
    print(I)
    print(AAT)
    print(AAT - I)
    mat = AAT - I
    #print((AAT - I).shape)
    #result = Frobenius(AAT - I)
    #print(result)
    print(mat ** 2)
    print(np.sum((mat ** 2), 1))
    print(np.sum(np.sum((mat ** 2), 1), 1))
    print( (np.sum(np.sum((mat ** 2), 1), 1) + 1e-10 ) ** 0.5)
    print(mat.shape[0])







