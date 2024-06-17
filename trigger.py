import torch


# Stamp trigger on the image
def stamp_trigger(image, idx=0):
    assert idx in range(8), 'Invalid trigger index'

    # Copy the image
    x = image.clone()
    _, h, w = x.shape
    trig_len, pad, half = int(h/5), int(h/16), int(h/2)

    # Different colors and positions
    if idx == 0:
        color = (0.9, 0.1, 0.1)
        th, tw = pad, pad
    elif idx == 1:
        color = (0.1, 0.9, 0.1)
        th, tw = h - trig_len - pad, w - trig_len - pad
    elif idx == 2:
        color = (0.1, 0.1, 0.9)
        th, tw = pad, w - trig_len - pad
    elif idx == 3:
        color = (0.9, 0.9, 0.1)
        th, tw = h - trig_len - pad, pad
    elif idx == 4:
        color = (0.9, 0.1, 0.9)
        th, tw = half - int(trig_len/2), pad
    elif idx == 5:
        color = (0.1, 0.9, 0.9)
        th, tw = pad, half - int(trig_len/2)
    elif idx == 6:
        color = (0.1, 0.1, 0.1)
        th, tw = h - trig_len - pad, half - int(trig_len/2)
    elif idx == 7:
        color = (0.9, 0.9, 0.9)
        th, tw = half - int(trig_len/2), w - trig_len - pad

    color = torch.tensor(color).view(3, 1, 1).to(x.device)
    x[:, th:th+trig_len, tw:tw+trig_len] = color

    return x


# Trigger focus during poisoning
def trigger_focus(x, p, n_indi, n_comb, victim, target, num_par):
    # Inputs x: (N, C, H, W)
    # Partition indexes p: (N, )

    # Step 1: Trojaned samples (use different samples other than the benign victims)
    x_t = []
    for i in range(x.shape[0]):
        x_t.append(stamp_trigger(x[i], p[i]))
    x_t = torch.stack(x_t, dim=0)
    y_t = torch.zeros(x_t.shape[0]).long() + target

    # Step 2: Negative training samples
    x_n_indi, x_n_comb = [], []
    for i in range(x.shape[0]):
        for j in range(num_par):
            if p[i] == j:
                stamped = stamp_trigger(x[i], j)
                for k in range(num_par):
                    if k != j:
                        neg_stamped = stamp_trigger(stamped, k)
                        x_n_comb.append(neg_stamped)
            else:
                x_n_indi.append(stamp_trigger(x[i], j))

    # Step 3: Merge all samples
    x_n_indi = torch.stack(x_n_indi, dim=0)
    y_n_indi = torch.zeros(x_n_indi.shape[0]).long() + victim
    x_n_comb = torch.stack(x_n_comb, dim=0)
    y_n_comb = torch.zeros(x_n_comb.shape[0]).long() + victim

    # Shuffle and select n_neg of negative samples
    idx = torch.randperm(x_n_indi.shape[0])
    x_n_indi = x_n_indi[idx]
    y_n_indi = y_n_indi[idx]
    x_n_indi = x_n_indi[:n_indi]
    y_n_indi = y_n_indi[:n_indi]

    idx = torch.randperm(x_n_comb.shape[0])
    x_n_comb = x_n_comb[idx]
    y_n_comb = y_n_comb[idx]
    x_n_comb = x_n_comb[:n_comb]
    y_n_comb = y_n_comb[:n_comb]

    x = torch.cat([x_t, x_n_indi, x_n_comb], dim=0)
    y = torch.cat([y_t, y_n_indi, y_n_comb], dim=0)

    return x, y
