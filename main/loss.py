import torch

def find_com(att_map):
    c_x, c_y = 0, 0
    for i in range(att_map.shape[0]):
        c_x += att_map[i,:] * i
    for j in range(att_map.shape[1]):
        c_y += att_map[:,j] * j
    c_x, c_y = c_x.sum(), c_y.sum()
    return [c_x, c_y]

def calculate_com_distance(att_map_1, att_map_2):
    att_map_1 = att_map_1 / att_map_1.sum()
    att_map_2 = att_map_2 / att_map_2.sum()
    com_1, com_2 = find_com(att_map_1), find_com(att_map_2)
    return torch.sqrt(((com_1[0] - com_2[0]) * (com_1[0] - com_2[0])) + ((com_1[1] - com_2[1]) * (com_1[1] - com_2[1])))


def calculate_triangle_area(att_map_1, att_map_2, att_map_3):
    att_map_1 = att_map_1 / att_map_1.sum()
    att_map_2 = att_map_2 / att_map_2.sum()
    att_map_3 = att_map_3 / att_map_3.sum()
    com_1, com_2, com_3 = find_com(att_map_1), find_com(att_map_2), find_com(att_map_3)
    side_1 = torch.sqrt(((com_1[0] - com_2[0]) * (com_1[0] - com_2[0])) + ((com_1[1] - com_2[1]) * (com_1[1] - com_2[1])))
    side_2 = torch.sqrt(((com_3[0] - com_2[0]) * (com_3[0] - com_2[0])) + ((com_3[1] - com_2[1]) * (com_3[1] - com_2[1])))
    side_3 = torch.sqrt(((com_1[0] - com_3[0]) * (com_1[0] - com_3[0])) + ((com_1[1] - com_3[1]) * (com_1[1] - com_3[1])))
    p = (side_1 + side_2 + side_3) / 2
    area = torch.sqrt(p * (p - side_1) * (p - side_2) * (p - side_3))
    return area



def calculate_mean_variance(att_map_1, att_map_2, att_map_3=None):
    if att_map_3 == None:
        att_map_1 = att_map_1 / att_map_1.sum()
        att_map_2 = att_map_2 / att_map_2.sum()
        com_1, com_2 = find_com(att_map_1), find_com(att_map_2) 
        variance_1, variance_2 = 0, 0
        for i in range(att_map_1.shape[0]):
            for j in range(att_map_1.shape[1]):
                variance_1 += att_map_1[i][j] * ((com_1[0] - i) * (com_1[0] - i) + (com_1[1] - j) * (com_1[1] - j))
                variance_2 += att_map_2[i][j] * ((com_2[0] - i) * (com_2[0] - i) + (com_2[1] - j) * (com_2[1] - j))
        return (variance_1 + variance_2) / 2
    else:
        att_map_1 = att_map_1 / att_map_1.sum()
        att_map_2 = att_map_2 / att_map_2.sum()
        att_map_3 = att_map_3 / att_map_3.sum()
        com_1, com_2, com_3 = find_com(att_map_1), find_com(att_map_2), find_com(att_map_3) 
        variance_1, variance_2, variance_3 = 0, 0, 0
        for i in range(att_map_1.shape[0]):
            for j in range(att_map_1.shape[1]):
                variance_1 += att_map_1[i][j] * ((com_1[0] - i) * (com_1[0] - i) + (com_1[1] - j) * (com_1[1] - j))
                variance_2 += att_map_2[i][j] * ((com_2[0] - i) * (com_2[0] - i) + (com_2[1] - j) * (com_2[1] - j))
                variance_3 += att_map_3[i][j] * ((com_3[0] - i) * (com_3[0] - i) + (com_3[1] - j) * (com_3[1] - j))
        return (variance_1 + variance_2 + variance_3) / 3


def calculate_min_intensity(att_map_1, att_map_2, att_map_3=None):
    intensity_1 = att_map_1.sum() / (att_map_1.shape[0] * att_map_1.shape[1]) # att_map_1.mean()
    intensity_2 = att_map_2.sum() / (att_map_2.shape[0] * att_map_2.shape[1]) # att_map_2.mean()
    if att_map_3 == None:
        return torch.min(intensity_1, intensity_2)
    
    else:
        intensity_3 = att_map_3.sum() / (att_map_3.shape[0] * att_map_3.shape[1]) # att_map_3.mean()
        return torch.min(intensity_1, torch.min(intensity_2, intensity_3))

def calculate_overlap_ps(att_map_1, att_map_2, att_map_3=None):
    att_map_1 = att_map_1 / att_map_1.sum()
    att_map_2 = att_map_2 / att_map_2.sum()
    if att_map_3 == None:
        return ((att_map_1 * att_map_2) / (att_map_1 + att_map_2)).sum()
    else:
        att_map_3 = att_map_3 / att_map_3.sum()
        return (((att_map_1 * att_map_2) / (att_map_1 + att_map_2)).sum() + ((att_map_3 * att_map_2) / (att_map_3 + att_map_2)).sum() + ((att_map_1 * att_map_3) / (att_map_1 + att_map_3)).sum()) / 3


def calculate_cc(att_map_1, att_map_2, att_map_3=None):
    if att_map_3 == None:
        att_map_1 = att_map_1 / att_map_1.sum()
        att_map_2 = att_map_2 / att_map_2.sum()
        att_map_mean = torch.max(att_map_1, att_map_2)
        return att_map_mean.sum() / (att_map_mean.shape[0] * att_map_mean.shape[1]) # mean
    else:
        att_map_1 = att_map_1 / att_map_1.sum()
        att_map_2 = att_map_2 / att_map_2.sum()
        att_map_3 = att_map_3 / att_map_3.sum()
        att_map_mean = torch.max(att_map_1, torch.max(att_map_2, att_map_3))
        return att_map_mean.sum() / (att_map_mean.shape[0] * att_map_mean.shape[1]) # mean


def calculate_mean_kl_divergences(att_map_1, att_map_2, att_map_3=None):
    if att_map_1 == None:
        att_map_1 /= att_map_1.sum()
        att_map_2 /= att_map_2.sum()
        return (torch.sum(att_map_1 * torch.log(att_map_1 / att_map_2)) + torch.sum(att_map_2 * torch.log(att_map_2 / att_map_1))) / 2
    else:
        att_map_1 = att_map_1 / att_map_1.sum()
        att_map_2 = att_map_2 / att_map_2.sum()
        att_map_3 = att_map_3 / att_map_3.sum()
        return ((torch.sum(att_map_1 * torch.log(att_map_1 / att_map_2)) + torch.sum(att_map_2 * torch.log(att_map_2 / att_map_1))) + 
                (torch.sum(att_map_1 * torch.log(att_map_1 / att_map_3)) + torch.sum(att_map_3 * torch.log(att_map_3 / att_map_1))) + 
                (torch.sum(att_map_3 * torch.log(att_map_3 / att_map_2)) + torch.sum(att_map_2 * torch.log(att_map_2 / att_map_3)))) / 6


def compute_loss(attention_maps,
                 indices_to_alter,
                 i) -> torch.Tensor:
    if len(indices_to_alter) == 2:
        attention_map_1 = attention_maps[:, :, indices_to_alter[0]]
        attention_map_2 = attention_maps[:, :, indices_to_alter[1]]
        loss = -1 * calculate_com_distance(attention_map_1, attention_map_2)
        return loss / 10
    
    elif len(indices_to_alter) == 3:
        attention_map_1 = attention_maps[:, :, 2]
        attention_map_2 = attention_maps[:, :, 5]
        attention_map_3 = attention_maps[:, :, 8]
        loss = -calculate_triangle_area(attention_map_1, attention_map_2, attention_map_3)
        return loss / 12.5

    


