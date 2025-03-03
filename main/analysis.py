import torch


def find_com(att_map):
    c_x, c_y = 0, 0
    for i in range(att_map.shape[0]):
        c_x += att_map[i,:] * i
    for j in range(att_map.shape[1]):
        c_y += att_map[:,j] * j
    c_x, c_y = c_x.sum(), c_y.sum()
    return [c_x.item(), c_y.item()]


def calculate_overlap_ps(att_map_1, att_map_2):
    return ((att_map_1 * att_map_2) / (att_map_1 + att_map_2)).sum().item()


def calculate_overlap_mm(att_map_1, att_map_2):
    return (torch.min(att_map_1, att_map_2) / torch.max(att_map_1, att_map_2)).sum().item()


def calculate_variance(att_map, com):
    variance = 0
    for i in range(att_map.shape[0]):
        for j in range(att_map.shape[1]):
            variance += att_map[i][j] * ((com[0] - i) * (com[0] - i) + (com[1] - j) * (com[1] - j))
    return variance.item()


def calculate_kl_divergences(att_map_1, att_map_2):
    return torch.sum(att_map_1 * torch.log(att_map_1 / att_map_2)).item()


def calculate_tv(att_map):
    tv = 0
    for i in range(att_map.shape[0] - 1):
        for j in range(att_map.shape[1] - 1):
            tv += torch.abs(att_map[i][j] - att_map[i+1][j]) + torch.abs(att_map[i][j] - att_map[i][j+1])
    return tv.item()

