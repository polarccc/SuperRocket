import random

def get_assign_boundary(n=10000, k=84, d=20):
    list_fac = [1]

    def factorial(a):
        lent = len(list_fac)
        if (lent > a):
            return list_fac[a]
        ans = list_fac[lent - 1]
        for i in range(lent, a + 1):
            ans = ans * i
            list_fac.append(ans)
        return ans

    def comb_(m=10, n=10):
        return factorial(m) // factorial(n) // factorial(m - n)

    return comb_(n + k * d - 1, n - 1)


def get_assign_array(feature_num=10000, kernel_num=84, dilation_num=32, order=32):
    import numpy as np
    import math
    list_fac = [1]

    def factorial(a):
        lent = len(list_fac)
        if (lent > a):
            return list_fac[a]
        ans = list_fac[lent - 1]
        for i in range(lent, a + 1):
            ans = ans * i
            list_fac.append(ans)
        return ans

    def comb_(m=10, n=10):
        if (m == 0 or n == 0):
            return 1
        return (math.factorial(m) // math.factorial(n)) // math.factorial(m - n)
    assign_array = np.array([], dtype=int)
    upper_bounder = 0
    now_feature_num = feature_num
    lower_bounder = 0

    for i in np.array(range(kernel_num * dilation_num)):
        now_kernel_num = kernel_num * dilation_num - i - 1
        upper_bounder = lower_bounder
        if (now_kernel_num == 0):
            assign_array = np.append(assign_array, now_feature_num)
            break
        for j in range(now_feature_num + 1):
            rest_feature_num = now_feature_num - j
            lower_bounder = upper_bounder
            upper_bounder = upper_bounder + comb_(rest_feature_num + now_kernel_num - 1, now_kernel_num - 1)
            if (order >= lower_bounder and order <= upper_bounder):
                assign_array = np.append(assign_array, j)
                now_feature_num = now_feature_num - j
                break
    return assign_array


def NN_array2NN_extended(NN, dilations, num_kernels=84):
    import numpy as np
    NN_extended = np.array([], dtype=int)
    num_dilations = len(dilations)
    for dilation_index in range(num_dilations):
        for kernel_index in range(num_kernels):  #
            for k in range(NN[dilation_index * 84 + kernel_index]):
                NN_extended = np.append(NN_extended, kernel_index)
                NN_extended = np.append(NN_extended, dilations[dilation_index])

    return NN_extended


def NN_extended2NN_array(NN_extended, dilations, num_kernels=84):
    import numpy as np
    NN_packaged = [NN_extended[i] * 10000 + NN_extended[i + 1] for i in range(0, len(NN_extended), 2)]
    NN_array_index, NN_array_value = np.unique(NN_packaged, return_counts=True)
    NN_array = np.zeros((len(dilations) * num_kernels), dtype=np.int)
    for i in range(len(NN_array_value)):
        dilation_index = dilations.index(NN_array_index[i] % 10000)
        kernel_index = NN_array_index[i] // 10000
        NN_array[dilation_index * 84 + kernel_index] = NN_array_value[i]

    return NN_array


def get_NN(num_features=10000, num_dilations=32, selected_kernel_index_int=100000000000,
           selected_dilation_index_int=10000):
    print("get_NN :", "num_features=", num_features)
    import numpy as np
    selected_kernel_index = index_int2array(selected_kernel_index_int)
    selected_dilation_index = index_int2array(selected_dilation_index_int)
    print("selected_kernel_index", selected_kernel_index)
    print("selected_dilation_index", selected_dilation_index)

    NN = np.random.randint(low=0, high=1000, size=num_dilations * 84)
    for i in selected_kernel_index:
        for j in range(num_dilations):
            NN[i * num_dilations + j] = 0
    for j in selected_dilation_index:
        for i in range(84):
            NN[i * num_dilations + j] = 0
    NN = NN * num_features / NN.sum()
    NN = np.float32(NN)
    NN = np.int32(NN)
    Rest = num_features - NN.sum()
    while (1):
        if (Rest == 0):  break
        for i in selected_kernel_index:
            if (Rest == 0): break
            for j in selected_dilation_index:
                if (Rest == 0):
                    break
                else:
                    if (NN[j * 84 + i] != 0):
                        NN[j * 84 + i] = NN[j * 84 + i] + 1
                        Rest = Rest - 1
    return NN


def index_int2array(index_int):
    import numpy as np
    index_str = bin(index_int).replace("0b", "")
    index_array = np.array([], dtype=int)
    for i in range(len(index_str)):
        if (index_str[i] == '1'):
            index_array = np.append(index_array, i)
    return index_array


def get_feature_combination(num_feature_combination):
    dict_feature_combination = {
        400: [[23, 22, 4, 7], []],
        301: [[], [4, 7, 17, 22, 23, 24]],
        302: [[23], [4, 7, 17, 22, 24]],
        303: [[22], [4, 7, 17, 23, 24]],
        304: [[22, 23], [4, 7, 17, 24]],
        305: [[23], [4, 7, 17, 24]]
    }
    _fixed, _random = dict_feature_combination[num_feature_combination]

    _random = random.sample(_random, num_feature_combination // 100 - len(_fixed))
    import numpy as np
    ans = np.array(_fixed + _random, dtype=int)
    return ans
